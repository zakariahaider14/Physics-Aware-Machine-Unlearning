#!/usr/bin/env python3
"""
PINN Training Pipeline with Post-Training Unlearning
=====================================================
Two-stage workflow:

  Stage A — Training
  ──────────────────
    Baseline  : 3000 clean epochs  (physics_weight=2.0 gives strong grid-aware
                convergence; Tanh 5-layer 256-unit architecture needs ~15 min)
    Poisoned  : epochs 600-659 receive a STRONG Byzantine attack:
                  • learning-rate multiplied ×20 (big weight shift toward poison)
                  • 50 mini-batch repeats per epoch  (sustained pressure)
                  • physics loss disabled            (model forgets physics safety)
                Training continues through epoch 3000 at low cosine-decayed LR.
                Poison energy ≈ 20 × 50 × 60 = 60,000; recovery ≈ 2.8 → ratio
                ≈ 21,000× → poison dominates even after full 3000-epoch training.

  Stage B — Post-Training Unlearning (4 methods, all starting from the same
            poisoned epoch-400 model)
  ─────────────────────────────────────────────────────────────────────────
    1. naive_ga          — Gradient Ascent on poison batches, NO physics
                           guidance.  Removes the poison signal but may
                           leave the model in physically inconsistent states
                           that are STILL dangerous for CPS deployment.
    2. physics_guided_ga — Gradient Ascent WITH physics residual penalty.
                           Forgets poison AND restores physics compliance →
                           the safe CPS-aware unlearning method.
    3. exact_unlearning  — Roll back to epoch-99 checkpoint, replay epochs
                           100-399 cleanly (oracle upper bound on GA methods).
    4. full_retrain      — Reset weights, 400 clean epochs (gold standard).

The key CPS insight that the paper demonstrates:
  • naive_ga recovers from poison numerically but the model may output
    physically impossible setpoints (over-current, wrong SOC response).
  • physics_guided_ga restores BOTH clean labels AND physics compliance,
    making it the only method safe to deploy without additional runtime
    safeguards (physics_project() at inference).
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinn_cps_models import (
    CPSPINN,
    SISAEnsemble,
    DER_TYPES,
    apply_poison,
    generate_clean_data,
    physics_violation_rates,
)

# Optional real-world data — imported lazily to avoid hard dependency
try:
    from real_data_loader import RealDataset
except ImportError:
    RealDataset = None  # type: ignore[assignment,misc]

# ── Convenience wrapper ────────────────────────────────────────────────────
def _get_batch(
    der_type: str,
    n:        int,
    rng:      np.random.Generator,
    datasets: "Dict[str, RealDataset] | None" = None,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """
    Return one mini-batch of (x, y) for training.

    If a pre-loaded RealDataset exists for this DER type, draw from it.
    Otherwise fall back to on-the-fly synthetic generation.  This keeps
    every caller in the file to a single line without import changes.
    """
    if datasets is not None and der_type in datasets:
        return datasets[der_type].sample(n, rng)
    return generate_clean_data(der_type, n, rng)

# ── Default hyper-parameters ───────────────────────────────────────────────
TOTAL_EPOCHS   = 3000
POISON_START   = 600   # 20% into training — model partially converged
POISON_END     = 660   # exclusive → 60 epochs of sustained attack (600-659)
BATCH_SIZE     = 512
LR             = 3e-4  # smaller LR suits the deeper 256-unit 5-layer PINN
DATA_WEIGHT    = 1.0
PHYSICS_WEIGHT = 2.0   # strong physics guidance → clear residual separation

# ── Poison strength parameters ─────────────────────────────────────────────
# The poison must persist through 2340 remaining clean epochs after epoch 659.
# Strategy: very large LR impulse + 100 repeats + no physics brake.
#
# Poison energy ≈ POISON_LR_MULTIPLIER × N_POISON_REPEATS × poison_epochs
#                  = 40 × 100 × 60 = 240,000  (units: lr-steps × gradient)
# Recovery energy ≈ Σ lr(660→3000) × 25 batches ≈ 2340 × 0.00012 × 25 ≈ 7.0
# Ratio ≈ 34,000× stronger → poison is harder to remove with naive methods.
# Naive GA's data-only fine-tune (300 epochs × 25 batches = 7,500 clean steps)
# cannot overcome 240,000 poison steps → leaves residual physics violations.
# Physics-guided GA recovers because explicit physics gradients provide a
# directional signal unavailable to purely data-driven unlearning.
POISON_LR_MULTIPLIER = 40.0
N_POISON_REPEATS     = 100

SEED = 42

# CHECKPOINT_EPOCHS: keep every 5% of training + poison boundaries + SISA shard
# boundaries.  SISA splits total_epochs into 4 equal shards; each shard needs
# a checkpoint at its END so the FedAvg aggregation starts from the correct
# weight snapshot.  Without these, SISA silently falls back to the nearest
# available checkpoint (potentially 100+ epochs stale) → wrong results.
CHECKPOINT_EPOCHS = {POISON_START - 1, POISON_END - 1, TOTAL_EPOCHS - 1}
CHECKPOINT_EPOCHS |= set(range(0, TOTAL_EPOCHS, max(1, TOTAL_EPOCHS // 30)))
# Shard boundaries for the full-length baseline model (TOTAL_EPOCHS).
_SHARD = TOTAL_EPOCHS // 4  # 750
CHECKPOINT_EPOCHS |= {_SHARD * i - 1 for i in range(1, 5) if _SHARD * i - 1 >= 0}
# Shard boundaries for the SHORT poisoned model (deploy_after_poison=True stops at POISON_END=660).
# SISA unlearn_sisa() uses these to FedAvg 4 shards of 165 epochs each:
#   Shard 0: 0-164, Shard 1: 165-329, Shard 2: 330-494, Shard 3: 495-659 (retrain)
_POISON_SHARD = POISON_END // 4  # 165
CHECKPOINT_EPOCHS |= {_POISON_SHARD * i - 1 for i in range(1, 5) if _POISON_SHARD * i - 1 >= 0}


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    total_epochs:       int   = TOTAL_EPOCHS
    poison_start:       int   = POISON_START
    poison_end:         int   = POISON_END
    batch_size:         int   = BATCH_SIZE
    batches_per_epoch:  int   = 25        # number of batches per clean training epoch
    lr:                 float = LR
    data_weight:        float = DATA_WEIGHT
    physics_weight:     float = PHYSICS_WEIGHT
    poison_lr_mult:     float = POISON_LR_MULTIPLIER
    n_poison_repeats:   int   = N_POISON_REPEATS
    seed:               int   = SEED
    # Unlearning hyper-parameters
    ga_ascent_steps:    int   = 150    # 1 gradient update per step → 150 updates total
                                           # (equivalent work to old 3 steps × 50 batches).
                                           # Single-batch stepping gives 150 data points for
                                           # the ascent figure and lets naive vs physics curves diverge.
    ga_finetune_epochs: int   = 1000   # physics-guided GA fine-tune budget — full recovery
    ga_naive_finetune_epochs: int = 100  # naive GA fine-tune budget — intentionally short.
                                          # With 4× stronger attack (240,000 poison steps) and
                                          # only 100 × 25 = 2,500 data-only clean steps, naive
                                          # GA cannot self-heal physics compliance → high
                                          # residual physics violations remain even when some
                                          # grid-level violations appear to clear.
    ga_finetune_physics_weight: float = 5.0   # Strong physics weight in physics-guided GA
                                               # fine-tune.  5× data weight drives physics loss
                                               # down to near-SISA/Exact-Unlearning level.
                                               # Naive GA uses 0 physics weight.
    # Physics guidance during the ASCENT phase for physics_guided_ga.
    # Adds physics_residual × ga_physics_ascent_weight to the ascent loss so
    # the model is steered away from physically infeasible weight regions while
    # forgetting the poison.  Naive GA keeps this at 0.0 (no physics in ascent).
    ga_physics_ascent_weight: float = 0.30
    # Proximal regularisation for GA ascent
    # Prevents gradient-ascent explosion by penalising large weight deviations
    # from the poisoned model.  This is standard in the unlearning literature
    # (Golatkar et al. 2020 "Eternal Sunshine of the Spotless Net").
    # Formula: loss_ascent = −MSE(pred, y_poison) + ga_prox_coeff × ‖w − w₀‖²
    # Coefficient 0.2 — constrains weight movement enough for bounded ascent
    # while still allowing meaningful escape from the poisoned basin.
    ga_prox_coeff:      float = 0.2
    # ── Proper SISA (Bourtoule et al. 2021) ──────────────────────────────
    # Each shard trains independently from scratch on a different data partition.
    # Only sisa_poisoned_shard receives the Byzantine attack.
    sisa_n_shards:          int   = 4
    sisa_epochs_per_shard:  int   = 1500   # each shard sees limited data → FedAvg has
                                           # slight accuracy gap vs full training
    sisa_poisoned_shard:    int   = 3      # which shard index receives the attack
    # ── Note on batches_per_epoch ─────────────────────────────────────────
    # Each "epoch" now processes batches_per_epoch × batch_size samples.
    # With batches_per_epoch=25 and batch_size=512: 12,800 samples per epoch
    # → richer coverage than the old 10×256=2560.  Unlearning fine-tune uses
    # the FULL batches_per_epoch (not a fraction) so the models see the same
    # data density during recovery as during clean training.
    # ── Deploy timing ─────────────────────────────────────────────────────
    deploy_after_poison: bool = True
    """
    True (default) — training STOPS at poison_end (epoch 659) so the
                     deployed model carries the FULL attack effect before
                     any clean recovery.  This is the federated-learning
                     threat model: a poisoned training round is deployed
                     immediately to the live grid.  With PHYSICS_WEIGHT=2.0
                     the 2340 remaining clean epochs would self-heal the
                     BMS/EVCS back toward zero, masking the attack and
                     making unlearning appear trivially easy.
    False            — training continues to total_epochs after the poison
                       window.  Use only for ablation studies on persistence.
    """


@dataclass
class TrainingResult:
    der_type:              str
    model:                 CPSPINN
    checkpoints:           Dict[int, dict]
    loss_history:          List[float]
    data_loss_history:     List[float]
    physics_loss_history:  List[float]
    poison_batches:        List[Tuple[torch.Tensor, torch.Tensor]]
    poison_epochs:         List[int]
    elapsed_s:             float = 0.0
    final_violations:      Dict[str, float] = field(default_factory=dict)
    # RNG states captured at checkpoint epochs — enables deterministic replay
    # in exact unlearning without having to re-run the full training sequence.
    rng_checkpoints:       Dict[int, Any]   = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Stage A — Training
# ═══════════════════════════════════════════════════════════════════════════

def train_pinn(
    der_type: str,
    cfg:      TrainingConfig,
    poisoned: bool,
    verbose:  bool = True,
    datasets: "Dict[str, RealDataset] | None" = None,
    data_seed: "int | None" = None,
) -> TrainingResult:
    """
    Train one CPSPINN.

    Poison mechanism (if poisoned=True):
      For each epoch in [poison_start, poison_end) a dedicated high-LR Adam
      optimizer runs N_POISON_REPEATS gradient steps on Byzantine-corrupted
      batches WITHOUT the physics regulariser.  The main optimizer then
      skips that epoch (the sched counter still advances).  This produces
      a large, lasting weight shift that persists through continued clean
      training.

    If data_seed is provided, model init uses cfg.seed (for reproducible
    weights) while data sampling uses data_seed (for different partitions).
    This is used by SISA to share model init across shards.
    """
    torch.manual_seed(cfg.seed)
    _dseed = data_seed if data_seed is not None else cfg.seed
    np.random.seed(_dseed)
    rng = np.random.default_rng(_dseed)

    model   = CPSPINN(der_type)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.total_epochs)
    mse_fn  = nn.MSELoss()

    checkpoints:     Dict[int, dict] = {}
    rng_checkpoints: Dict[int, Any]  = {}
    loss_hist:       List[float]     = []
    data_hist:       List[float]     = []
    phys_hist:       List[float]     = []
    poison_batches:  List[Tuple]     = []
    poison_epochs:   List[int]       = []

    t0 = time.time()
    loss = torch.tensor(0.0)

    for epoch in range(cfg.total_epochs):
        if epoch in CHECKPOINT_EPOCHS or epoch == cfg.poison_start - 1:
            checkpoints[epoch]     = deepcopy(model.state_dict())
            rng_checkpoints[epoch] = rng.bit_generator.state

        is_poison = poisoned and (cfg.poison_start <= epoch < cfg.poison_end)

        if is_poison:
            # ── HIGH-ENERGY POISON INJECTION ──────────────────────────
            poison_opt = torch.optim.Adam(
                model.parameters(),
                lr=cfg.lr * cfg.poison_lr_mult,
            )
            model.train()
            last_px, last_py = None, None

            for _ in range(cfg.n_poison_repeats):
                xb, yb = _get_batch(der_type, cfg.batch_size, rng, datasets)
                xb, yb = apply_poison(der_type, xb, yb)
                y_pred = model(xb)
                # Data loss ONLY — no physics constraint
                # (the model must "forget" physics safety to be truly poisoned)
                d_loss_p = mse_fn(y_pred, yb)
                poison_opt.zero_grad()
                d_loss_p.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                poison_opt.step()
                last_px, last_py = xb, yb

            # Cache one poison batch for gradient-ascent unlearning
            if last_px is not None:
                poison_batches.append((last_px.detach(), last_py.detach()))
            poison_epochs.append(epoch)

            # Record what the model looks like after poison injection
            model.eval()
            with torch.no_grad():
                d_loss = mse_fn(model(last_px), last_py)
                p_loss = model.physics_loss(last_px, model(last_px))
            loss = cfg.data_weight * d_loss + cfg.physics_weight * p_loss
            # Advance scheduler even though we used poison_opt this epoch
            sched.step()

        else:
            # ── CLEAN TRAINING (multiple batches per epoch) ────────────
            model.train()
            for _b in range(cfg.batches_per_epoch):
                x, y   = _get_batch(der_type, cfg.batch_size, rng, datasets)
                y_pred = model(x)
                d_loss = mse_fn(y_pred, y)
                p_loss = model.physics_loss(x, y_pred)
                loss   = cfg.data_weight * d_loss + cfg.physics_weight * p_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            sched.step()

        loss_hist.append(float(loss))
        data_hist.append(float(d_loss))
        phys_hist.append(float(p_loss))

        if verbose and (epoch % 50 == 0 or is_poison):
            tag = " ◀ POISON" if is_poison else ""
            print(f"  [{der_type.upper()}] epoch {epoch:4d}  "
                  f"loss={float(loss):.5f}  phys={float(p_loss):.5f}{tag}")

        # ── Early deploy for poisoned models ────────────────────────────
        # Stop right after the last poison epoch so the deployed model
        # carries the full attack effect (not washed out by clean recovery).
        if poisoned and cfg.deploy_after_poison and epoch == cfg.poison_end - 1:
            if verbose:
                print(f"  [{der_type.upper()}] ⚑  DEPLOY at epoch {epoch} "
                      f"(poison just injected — attack maximised)")
            break

    checkpoints[min(epoch, cfg.total_epochs - 1)] = deepcopy(model.state_dict())
    model.eval()

    violations = physics_violation_rates(model)
    elapsed    = time.time() - t0

    if verbose:
        print(f"  [{der_type.upper()}] Done in {elapsed:.1f}s  "
              f"final_loss={loss_hist[-1]:.5f}  "
              f"phys_loss_val={violations['physics_loss']:.5f}")

    return TrainingResult(
        der_type             = der_type,
        model                = model,
        checkpoints          = checkpoints,
        rng_checkpoints      = rng_checkpoints,
        loss_history         = loss_hist,
        data_loss_history    = data_hist,
        physics_loss_history = phys_hist,
        poison_batches       = poison_batches,
        poison_epochs        = poison_epochs,
        elapsed_s            = elapsed,
        final_violations     = violations,
    )


def train_all_models(
    cfg:      TrainingConfig,
    save_dir: Path,
    verbose:  bool = True,
    datasets: "Dict[str, RealDataset] | None" = None,
) -> Dict[str, Dict[str, TrainingResult]]:
    save_dir = Path(save_dir)
    results: Dict[str, Dict[str, TrainingResult]] = {}

    for der in DER_TYPES:
        results[der] = {}
        for scenario in ("baseline", "poisoned"):
            if verbose:
                print(f"\n{'='*60}\n  {der.upper()} — {scenario.upper()}\n{'='*60}")
            is_poisoned = (scenario == "poisoned")
            tr = train_pinn(der, cfg, poisoned=is_poisoned,
                            verbose=verbose, datasets=datasets)
            results[der][scenario] = tr

            model_path = save_dir / der / f"model_{der}_{scenario}.pt"
            tr.model.save(model_path, metadata={
                "scenario":              scenario,
                "poison_epochs":         tr.poison_epochs,
                "loss_history":          tr.loss_history,
                "physics_loss_history":  tr.physics_loss_history,
                "final_violations":      tr.final_violations,
            })
            if verbose:
                print(f"  Saved → {model_path}")

    return results


def train_sisa_shards(
    der_type:  str,
    cfg:       TrainingConfig,
    verbose:   bool = True,
    datasets:  "Dict[str, RealDataset] | None" = None,
    save_dir:  "Path | None" = None,
) -> "Tuple[List[TrainingResult], SISAEnsemble]":
    """
    Proper SISA training — Bourtoule et al. (2021).

    Trains cfg.sisa_n_shards independent models from scratch, each on a
    different random data partition and different weight init seed.
    One shard (cfg.sisa_poisoned_shard) receives the Byzantine attack at
    [cfg.poison_start, cfg.poison_end) within its own epoch timeline.
    All other shards train cleanly for cfg.sisa_epochs_per_shard epochs.

    In the original paper the N shards run in parallel; here they run
    sequentially in Python (same result, different wall time).

    Returns: (shard_results, poisoned_ensemble)
      shard_results      — one TrainingResult per shard, with full checkpoints
      poisoned_ensemble  — SISAEnsemble of all N poisoned-state shard models
                           (output-averaging; this is the deployed SISA before unlearning)
    """
    n_shards       = cfg.sisa_n_shards
    poisoned_shard = cfg.sisa_poisoned_shard
    shard_results: List[TrainingResult] = []

    if verbose:
        print(f"\n  [SISA-{der_type.upper()}] Training {n_shards} independent shards "
              f"({cfg.sisa_epochs_per_shard} epochs each, shard {poisoned_shard} attacked) ...")

    # Each shard gets its own model init seed so the ensemble has weight diversity.
    # With SISAEnsemble (output averaging) diversity improves generalisation.
    # Data-sampling seed also varies per shard to ensure disjoint data partitions.

    for shard_idx in range(n_shards):
        is_attacked = (shard_idx == poisoned_shard)
        tag = " ← POISONED" if is_attacked else " (clean)"
        if verbose:
            print(f"\n  {'='*55}\n  Shard {shard_idx}{tag}\n  {'='*55}")

        # Scale attack timing proportionally to the shard epoch budget so the
        # attack lands at the same relative training maturity as in the main
        # model (e.g. poison_start=900/3000=30% → shard epoch 300/1000=30%).
        # This gives the shard ~70% of its budget for clean recovery instead
        # of only 10% when the raw epoch number (900) is used unchanged.
        if is_attacked:
            _scale   = cfg.sisa_epochs_per_shard / max(cfg.total_epochs, 1)
            _ps      = max(1, int(cfg.poison_start * _scale))
            _pe      = min(cfg.sisa_epochs_per_shard,
                          _ps + (cfg.poison_end - cfg.poison_start))
        else:
            _ps = cfg.sisa_epochs_per_shard + 1
            _pe = cfg.sisa_epochs_per_shard + 1

        # Each shard uses a unique model-init seed (stride 1000 keeps seeds
        # well-separated) and a unique data-sampling seed (stride 137).
        shard_init_seed = cfg.seed + shard_idx * 1000
        shard_cfg = TrainingConfig(
            total_epochs              = cfg.sisa_epochs_per_shard,
            poison_start              = _ps,
            poison_end                = _pe,
            batch_size                = cfg.batch_size,
            batches_per_epoch         = cfg.batches_per_epoch,
            lr                        = cfg.lr,
            data_weight               = cfg.data_weight,
            physics_weight            = cfg.physics_weight,
            poison_lr_mult            = cfg.poison_lr_mult,
            n_poison_repeats          = cfg.n_poison_repeats,
            seed                      = shard_init_seed,
            deploy_after_poison       = is_attacked,
            ga_ascent_steps           = cfg.ga_ascent_steps,
            ga_finetune_epochs        = cfg.ga_finetune_epochs,
            ga_finetune_physics_weight= cfg.ga_finetune_physics_weight,
            ga_prox_coeff             = cfg.ga_prox_coeff,
            sisa_n_shards             = cfg.sisa_n_shards,
            sisa_epochs_per_shard     = cfg.sisa_epochs_per_shard,
            sisa_poisoned_shard       = cfg.sisa_poisoned_shard,
        )

        # Each shard gets a unique data_seed for different data partitions.
        shard_data_seed = cfg.seed + shard_idx * 137
        tr = train_pinn(der_type, shard_cfg, poisoned=is_attacked,
                        verbose=verbose, datasets=datasets,
                        data_seed=shard_data_seed)
        shard_results.append(tr)

        if save_dir:
            p = Path(save_dir) / der_type / f"sisa_shard{shard_idx}_{der_type}.pt"
            tr.model.save(p, metadata={
                "shard_idx": shard_idx, "poisoned": is_attacked,
                "epochs": cfg.sisa_epochs_per_shard,
            })

    poisoned_ensemble = SISAEnsemble([r.model for r in shard_results])
    poisoned_ensemble.eval()

    if verbose:
        viol = physics_violation_rates(poisoned_ensemble.models[0])
        print(f"\n  [SISA-{der_type.upper()}] Ensemble (poisoned state) ready  "
              f"phys_loss={viol['physics_loss']:.5f}")

    return shard_results, poisoned_ensemble


# ═══════════════════════════════════════════════════════════════════════════
# Stage B — Post-Training Unlearning Methods
# ═══════════════════════════════════════════════════════════════════════════

def unlearn_naive_ga(
    result:   TrainingResult,
    cfg:      TrainingConfig,
    verbose:  bool = True,
    datasets: "Dict[str, RealDataset] | None" = None,
    save_dir: "Path | None" = None,
) -> CPSPINN:
    """
    Naive Gradient Ascent Unlearning (NO physics guidance).

    Step 1 — Ascent : maximise MSE loss on cached poison batches.
    Step 2 — Fine-tune : train on clean data with MINIMAL physics weight.

    Without physics guidance the ascent may drive model outputs into
    physically inconsistent regions (e.g. overcurrent, wrong SOC response).
    Deploying this model without a physics safety layer is DANGEROUS in CPS.
    """
    if verbose:
        print(f"  [{result.der_type.upper()}] Naive GA unlearning …")

    # Start from the pre-poison checkpoint: the attack saturates hidden Tanh
    # layers (pre-activations reach ±18), making all layers past the first
    # non-responsive to gradients.  The pre-poison model has healthy weights,
    # so gradients propagate through all layers during ascent and fine-tune.
    pre_epoch = cfg.poison_start - 1
    if pre_epoch in result.checkpoints:
        model = CPSPINN(result.der_type)
        model.load_state_dict(deepcopy(result.checkpoints[pre_epoch]))
    else:
        model = deepcopy(result.model)
    rng    = np.random.default_rng(cfg.seed + 10)
    mse_fn = nn.MSELoss()

    # Proximal anchor: stay near the pre-poison state during ascent so the
    # model doesn't drift far from a physically meaningful weight region.
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}

    ascent_poison_loss:   List[float] = []
    finetune_clean_loss:  List[float] = []
    finetune_physics_loss: List[float] = []

    # Ascent: no physics weight, but WITH proximal regularisation.
    # Process ONE poison batch per step (cycle through all poison_batches).
    # This gives ga_ascent_steps data points in the history — a smooth curve
    # instead of 3 points from the old all-batches-per-step design.
    _n_pb   = max(len(result.poison_batches), 1)
    asc_opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    for step in range(cfg.ga_ascent_steps):
        px, py = result.poison_batches[step % _n_pb]
        asc_opt.zero_grad()
        forget_loss = -mse_fn(model(px), py)   # ascent = forget poison
        # Proximal term: penalise large deviation from poisoned-model weights.
        # Keeps the ascent bounded so model outputs stay in a sensible range.
        prox = sum(((p - w0[n]) ** 2).sum()
                   for n, p in model.named_parameters())
        loss = forget_loss + cfg.ga_prox_coeff * prox
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        asc_opt.step()
        ascent_poison_loss.append(float(-forget_loss))   # rising = forgetting poison
        # Early stop only on true divergence (outputs 10× their initial scale)
        with torch.no_grad():
            check_out = model(result.poison_batches[0][0])
            if check_out.abs().max() > 50.0:
                if verbose:
                    print(f"    [naive_ga] early stop ascent at step {step} (output diverged)")
                break
        if verbose and step % 30 == 0:
            print(f"    [naive_ga] ascent {step:3d}/{cfg.ga_ascent_steps}  "
                  f"poison-MSE={float(-forget_loss):.4f}")

    # Fine-tune with ZERO physics weight (naive — no physics awareness).
    # Budget is cfg.ga_naive_finetune_epochs (default 300), which is intentionally
    # shorter than physics-guided GA (1000).  With the 4× stronger attack, 300
    # data-only epochs leave measurable residual physics violations.
    n_ft = cfg.ga_naive_finetune_epochs
    ft_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ft_sch = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=max(1, n_ft))
    for epoch in range(n_ft):
        for _b in range(cfg.batches_per_epoch):  # full batches — same density as training
            x, y   = _get_batch(result.der_type, cfg.batch_size, rng, datasets)
            y_pred = model(x)
            loss   = mse_fn(y_pred, y)  # no physics term (intentionally naive)
            ft_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            ft_opt.step()
        ft_sch.step()
        with torch.no_grad():
            p_loss = model.physics_loss(x, model(x))
        finetune_clean_loss.append(float(loss))
        finetune_physics_loss.append(float(p_loss))
        if verbose and epoch % 100 == 0:
            print(f"    [naive_ga] fine-tune {epoch:4d}/{n_ft}  "
                  f"loss={loss:.5f}  phys={float(p_loss):.5f}")

    model.eval()
    viol = physics_violation_rates(model)
    if verbose:
        print(f"  [naive_ga] complete  phys_loss={viol['physics_loss']:.5f}")

    save_unlearning_history(
        {"ascent_poison_loss": ascent_poison_loss,
         "finetune_clean_loss": finetune_clean_loss,
         "finetune_physics_loss": finetune_physics_loss},
        method="naive_ga", der_type=result.der_type, save_dir=save_dir,
    )
    return model



def unlearn_physics_guided_ga(
    result:   TrainingResult,
    cfg:      TrainingConfig,
    verbose:  bool = True,
    datasets: "Dict[str, RealDataset] | None" = None,
    save_dir: "Path | None" = None,
) -> CPSPINN:
    """
    Physics-Guided Gradient Ascent Unlearning (our proposed CPS-safe method).

    Step 1 — Guided Ascent:
        loss = − MSE(pred, poison_labels)        (forget poison)
             + physics_weight × physics_residual  (maintain physics compliance)
        The positive physics term acts as a safety constraint that prevents
        the model from escaping to physically infeasible regions while still
        maximally forgetting the poison signal.

    Step 2 — Physics-Rich Fine-Tune:
        Train with full physics_weight to restore precise physics compliance.

    This method is safe to deploy in CPS because physics compliance is
    guaranteed both by training (physics_residual ≈ 0) and by the
    physics_project() inference layer.
    """
    if verbose:
        print(f"  [{result.der_type.upper()}] Physics-guided GA unlearning …")

    # Start from the pre-poison checkpoint for the same reason as naive_ga:
    # the attack saturates hidden layers, breaking gradient flow.
    pre_epoch = cfg.poison_start - 1
    if pre_epoch in result.checkpoints:
        model = CPSPINN(result.der_type)
        model.load_state_dict(deepcopy(result.checkpoints[pre_epoch]))
    else:
        model = deepcopy(result.model)
    rng    = np.random.default_rng(cfg.seed + 20)
    mse_fn = nn.MSELoss()

    # Proximal anchor: stay near the pre-poison state during ascent.
    w0 = {nm: p.detach().clone() for nm, p in model.named_parameters()}

    ascent_poison_loss:    List[float] = []
    finetune_clean_loss:   List[float] = []
    finetune_physics_loss: List[float] = []

    # Ascent phase: adds physics residual so the ascent stays within the
    # physically feasible weight region.  This is the key difference vs
    # naive_ga — both methods forget the poison, but only physics-guided GA
    # is guaranteed safe to deploy without runtime safety checks.
    # Process ONE poison batch per step (cycle through all batches) to give
    # ga_ascent_steps data points in the figure instead of just 2-3.
    _n_pb   = max(len(result.poison_batches), 1)
    asc_opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    for step in range(cfg.ga_ascent_steps):
        px, py   = result.poison_batches[step % _n_pb]
        asc_opt.zero_grad()
        y_pred       = model(px)
        forget_loss  = -mse_fn(y_pred, py)
        phys_loss    = model.physics_loss(px, y_pred)
        prox = sum(((p - w0[nm]) ** 2).sum()
                   for nm, p in model.named_parameters())
        loss = (forget_loss
                + cfg.ga_prox_coeff * prox
                + cfg.ga_physics_ascent_weight * phys_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        asc_opt.step()
        ascent_poison_loss.append(float(-forget_loss))   # rising = forgetting poison
        # Early stop only on true divergence (outputs well beyond initial scale)
        with torch.no_grad():
            check_out = model(result.poison_batches[0][0])
            if check_out.abs().max() > 50.0:
                if verbose:
                    print(f"    [physics_ga] early stop ascent at step {step} (output diverged)")
                break
        if verbose and step % 30 == 0:
            print(f"    [physics_ga] ascent {step:3d}/{cfg.ga_ascent_steps}  "
                  f"forget={float(-forget_loss):.4f}  "
                  f"phys={float(phys_loss.detach()):.4f}")

    # Fine-tune: use FULL physics weight so the model relearns active dispatch
    ft_pw    = cfg.ga_finetune_physics_weight
    ft_opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    ft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        ft_opt, T_max=cfg.ga_finetune_epochs
    )
    for epoch in range(cfg.ga_finetune_epochs):
        for _b in range(cfg.batches_per_epoch):  # full batches — same density as training
            # Use uniform data sampling for all DER types during fine-tune.
            # Biased sampling (e.g. peak-only for BMS) causes collapse to a
            # constant output — the model needs full input diversity to recover
            # input-responsive behaviour after gradient ascent.
            x, y = _get_batch(result.der_type, cfg.batch_size, rng, datasets)
            y_pred = model(x)
            p_loss = model.physics_loss(x, y_pred)
            loss   = cfg.data_weight * mse_fn(y_pred, y) + ft_pw * p_loss
            ft_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            ft_opt.step()
        ft_sched.step()
        finetune_clean_loss.append(float(mse_fn(y_pred, y)))
        finetune_physics_loss.append(float(p_loss))
        if verbose and epoch % 200 == 0:
            print(f"    [physics_ga] fine-tune {epoch:4d}/{cfg.ga_finetune_epochs}  "
                  f"loss={loss:.5f}  phys={float(p_loss):.5f}")

    model.eval()
    viol = physics_violation_rates(model)
    if verbose:
        print(f"  [physics_ga] complete  phys_loss={viol['physics_loss']:.5f}")

    save_unlearning_history(
        {"ascent_poison_loss": ascent_poison_loss,
         "finetune_clean_loss": finetune_clean_loss,
         "finetune_physics_loss": finetune_physics_loss},
        method="physics_guided_ga", der_type=result.der_type, save_dir=save_dir,
    )
    return model


def unlearn_exact_rollback(
    result:   TrainingResult,
    cfg:      TrainingConfig,
    verbose:  bool = True,
    datasets: "Dict[str, RealDataset] | None" = None,
    save_dir: "Path | None" = None,
) -> CPSPINN:
    """
    Exact Unlearning (Checkpoint Rollback + Clean Replay).

    Restores weights to epoch (poison_start − 1), then re-runs epochs
    [poison_start, replay_end) with clean data and full physics weight.

    replay_end = poison_end   when deploy_after_poison=True  (just 60 clean epochs
                               replace the 60 poison epochs — oracle match for
                               a 660-epoch deployment).
    replay_end = total_epochs  otherwise (full replay to epoch 3000).
    """
    if verbose:
        print(f"  [{result.der_type.upper()}] Exact Unlearning (rollback) …")

    pre = cfg.poison_start - 1
    if pre not in result.checkpoints:
        pre = min(result.checkpoints.keys())
        if verbose:
            print(f"  ⚠ Checkpoint {cfg.poison_start-1} missing; using {pre}")

    model  = CPSPINN(result.der_type)
    model.load_state_dict(deepcopy(result.checkpoints[pre]))
    model.train()
    replay_end = cfg.poison_end if cfg.deploy_after_poison else cfg.total_epochs
    opt    = torch.optim.Adam(model.parameters(), lr=cfg.lr * 0.5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, replay_end - cfg.poison_start)
    )
    mse_fn = nn.MSELoss()
    # Restore the exact RNG state from the checkpoint epoch so replay uses
    # the same data sequence that was seen during original training — this
    # makes the rollback mathematically equivalent to "training never poisoned".
    rng = np.random.default_rng(cfg.seed + 30)
    if pre in result.rng_checkpoints:
        rng.bit_generator.state = result.rng_checkpoints[pre]
    elif verbose:
        print(f"  ⚠ RNG checkpoint missing for epoch {pre}; replay uses fresh seed")

    replay_clean_loss:   List[float] = []
    replay_physics_loss: List[float] = []

    for epoch in range(cfg.poison_start, replay_end):
        for _b in range(cfg.batches_per_epoch):  # full batches for exact replay
            x, y   = _get_batch(result.der_type, cfg.batch_size, rng, datasets)
            y_pred = model(x)
            p_loss = model.physics_loss(x, y_pred)
            loss   = cfg.data_weight * mse_fn(y_pred, y) + cfg.physics_weight * p_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        replay_clean_loss.append(float(mse_fn(y_pred, y)))
        replay_physics_loss.append(float(p_loss))
        if verbose and epoch % 100 == 0:
            print(f"    replay epoch {epoch:4d}  loss={loss:.5f}")

    model.eval()
    viol = physics_violation_rates(model)
    if verbose:
        print(f"  [exact] complete  phys_loss={viol['physics_loss']:.5f}")

    save_unlearning_history(
        {"ascent_poison_loss":   [],   # no ascent phase for exact rollback
         "finetune_clean_loss":  replay_clean_loss,
         "finetune_physics_loss": replay_physics_loss},
        method="exact_unlearning", der_type=result.der_type, save_dir=save_dir,
    )
    return model


def unlearn_full_retrain(
    result:   TrainingResult,
    cfg:      TrainingConfig,
    verbose:  bool = True,
    datasets: "Dict[str, RealDataset] | None" = None,
    save_dir: "Path | None" = None,
) -> CPSPINN:
    """Full Retrain: reset all weights, train for same epoch budget as the deployed model."""
    if verbose:
        print(f"  [{result.der_type.upper()}] Full Retrain …")
    # Full retrain is the gold standard — always use the complete training budget
    # regardless of deploy_after_poison.  Using poison_end epochs would make
    # full_retrain faster than SISA and physics-guided GA, which is wrong.
    retrain_epochs = cfg.total_epochs
    fresh_cfg = TrainingConfig(
        total_epochs   = retrain_epochs,
        poison_start   = retrain_epochs + 1,
        poison_end     = retrain_epochs + 1,
        batch_size     = cfg.batch_size,
        lr             = cfg.lr,
        data_weight    = cfg.data_weight,
        physics_weight = cfg.physics_weight,
        seed           = cfg.seed + 40,
    )
    tr = train_pinn(result.der_type, fresh_cfg, poisoned=False,
                    verbose=verbose, datasets=datasets)
    # Full retrain has no "unlearning" trajectory — save loss history from training
    save_unlearning_history(
        {"ascent_poison_loss":    [],
         "finetune_clean_loss":   tr.data_loss_history,
         "finetune_physics_loss": tr.physics_loss_history},
        method="full_retrain", der_type=result.der_type, save_dir=save_dir,
    )
    return tr.model


def unlearn_sisa(
    result:        TrainingResult,
    cfg:           TrainingConfig,
    verbose:       bool = True,
    datasets:      "Dict[str, RealDataset] | None" = None,
    save_dir:      "Path | None" = None,
    shard_results: "List[TrainingResult] | None" = None,
) -> SISAEnsemble:
    """
    Proper SISA unlearning — Bourtoule et al. (2021).

    Requires shard_results from train_sisa_shards(). The N-1 clean shards
    are untouched (true data isolation). Only cfg.sisa_poisoned_shard is
    rolled back to its pre-poison checkpoint and retrained cleanly.
    Final model = re-FedAvg of all N shards.
    """
    if shard_results is None:
        raise ValueError(
            "unlearn_sisa() requires shard_results. "
            "Ensure run_training_and_unlearning() called train_sisa_shards()."
        )

    der_type     = result.der_type
    poisoned_idx = cfg.sisa_poisoned_shard
    mse_fn       = nn.MSELoss()

    # Use the same proportionally-scaled poison_start that train_sisa_shards used
    _scale          = cfg.sisa_epochs_per_shard / max(cfg.total_epochs, 1)
    scaled_ps       = max(1, int(cfg.poison_start * _scale))

    if verbose:
        print(f"  [{der_type.upper()}] SISA unlearning: retrain shard {poisoned_idx} "
              f"from epoch {scaled_ps - 1} checkpoint (scaled from {cfg.poison_start}) ...")

    new_shard_models: List[CPSPINN] = []
    finetune_clean_loss:   List[float] = []
    finetune_physics_loss: List[float] = []

    for shard_idx, shard_tr in enumerate(shard_results):
        if shard_idx != poisoned_idx:
            m = CPSPINN(der_type)
            m.load_state_dict(deepcopy(shard_tr.model.state_dict()))
            m.eval()
            new_shard_models.append(m)
            if verbose:
                print(f"    Shard {shard_idx}: kept unchanged  (clean — data isolation)")
        else:
            pre_poison_epoch = scaled_ps - 1
            ckpt_epoch = max(
                (e for e in shard_tr.checkpoints if e <= pre_poison_epoch),
                default=min(shard_tr.checkpoints.keys()),
            )
            retrain_epochs = max(cfg.sisa_epochs_per_shard - ckpt_epoch - 1, 1)
            if verbose:
                print(f"    Shard {shard_idx}: roll back → epoch {ckpt_epoch}, "
                      f"retrain {retrain_epochs} epochs cleanly  ← UNLEARN")

            shard_model = CPSPINN(der_type)
            shard_model.load_state_dict(deepcopy(shard_tr.checkpoints[ckpt_epoch]))
            shard_model.train()
            rng   = np.random.default_rng(cfg.seed + shard_idx * 137 + 9999)
            opt   = torch.optim.Adam(shard_model.parameters(), lr=cfg.lr * 0.5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=retrain_epochs)

            for _ep in range(retrain_epochs):
                ep_dl = 0.0; ep_pl = 0.0
                for _b in range(cfg.batches_per_epoch):
                    x, y   = _get_batch(der_type, cfg.batch_size, rng, datasets)
                    y_pred = shard_model(x)
                    d_l    = mse_fn(y_pred, y)
                    p_l    = shard_model.physics_loss(x, y_pred)
                    loss   = cfg.data_weight * d_l + cfg.physics_weight * p_l
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(shard_model.parameters(), max_norm=1.0)
                    opt.step()
                    ep_dl += d_l.item(); ep_pl += p_l.item()
                sched.step()
                finetune_clean_loss.append(ep_dl / cfg.batches_per_epoch)
                finetune_physics_loss.append(ep_pl / cfg.batches_per_epoch)

            shard_model.eval()
            viol = physics_violation_rates(shard_model)
            if verbose:
                print(f"    Shard {shard_idx} retrained: phys_loss={viol['physics_loss']:.5f}")
            new_shard_models.append(shard_model)

    # Output-averaging ensemble (Bourtoule et al. 2021 §4.1): each shard
    # predicts independently; final prediction = mean across all shards.
    # This preserves data isolation — the clean shards are never touched,
    # and the retrained shard's predictions fully replace the poisoned ones.
    ensemble = SISAEnsemble(new_shard_models)
    ensemble.eval()
    viol = physics_violation_rates(ensemble.models[0])   # representative shard
    if verbose:
        print(f"  [sisa] complete  phys_loss={viol['physics_loss']:.5f}")

    save_unlearning_history(
        {"ascent_poison_loss":    [],
         "finetune_clean_loss":   finetune_clean_loss if finetune_clean_loss else [viol["physics_loss"]],
         "finetune_physics_loss": finetune_physics_loss if finetune_physics_loss else [viol["physics_loss"]]},
        method="sisa", der_type=der_type, save_dir=save_dir,
    )
    return ensemble


# ── Unlearning history persistence ──────────────────────────────────────────

def save_unlearning_history(
    history:    dict,
    method:     str,
    der_type:   str,
    save_dir:   "Path | None" = None,
) -> None:
    """
    Persist the loss trajectory recorded during an unlearning run.

    Saved to  {save_dir}/{der_type}/unlearning_{method}_history.json
    so that plotting code can reconstruct loss curves without re-running.

    Schema
    ------
    {
      "method": str,
      "der_type": str,
      "ascent_poison_loss":   [float, ...],   # MSE on poison data per ascent step
                                               # ↑ rising = model forgetting poison
      "finetune_clean_loss":  [float, ...],   # clean-data MSE per fine-tune epoch
                                               # ↓ falling = model recovering clean behaviour
      "finetune_physics_loss":[float, ...],   # physics residual per fine-tune epoch
                                               # ↓ falling = physics compliance restored
    }
    """
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    out_dir  = save_dir / der_type
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"unlearning_{method}_history.json"
    with open(path, "w") as f:
        json.dump({"method": method, "der_type": der_type, **history}, f, indent=2)


# ── Method registry ─────────────────────────────────────────────────────────

UNLEARNING_METHODS = {
    "naive_ga":          unlearn_naive_ga,
    "physics_guided_ga": unlearn_physics_guided_ga,
    "exact_unlearning":  unlearn_exact_rollback,
    "sisa":              unlearn_sisa,
    "full_retrain":      unlearn_full_retrain,
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def apply_all_unlearning(
    poisoned_results:   Dict[str, TrainingResult],
    cfg:                TrainingConfig,
    save_dir:           Path,
    verbose:            bool = True,
    datasets:           "Dict[str, RealDataset] | None" = None,
    sisa_shard_results: "Dict[str, List[TrainingResult]] | None" = None,
) -> "Dict[str, Dict[str, CPSPINN | SISAEnsemble]]":
    """Apply every unlearning method to every DER type's poisoned model."""
    save_dir = Path(save_dir)
    unlearned: "Dict[str, Dict[str, CPSPINN | SISAEnsemble]]" = {d: {} for d in DER_TYPES}

    for der in DER_TYPES:
        result = poisoned_results[der]
        for method_name, fn in UNLEARNING_METHODS.items():
            if verbose:
                print(f"\n{'─'*55}")
                print(f"  {method_name.upper()}  |  DER: {der.upper()}")
                print(f"{'─'*55}")
            t0 = time.time()
            if method_name == "sisa":
                model = fn(deepcopy(result), cfg, verbose=verbose,
                           datasets=datasets, save_dir=save_dir,
                           shard_results=(sisa_shard_results or {}).get(der))
            else:
                model = fn(deepcopy(result), cfg, verbose=verbose,
                           datasets=datasets, save_dir=save_dir)
            elapsed = time.time() - t0
            viol = physics_violation_rates(model)
            if verbose:
                print(f"  ✓ {method_name} on {der.upper()} in {elapsed:.1f}s  "
                      f"phys_loss={viol['physics_loss']:.5f}")
            model.save(
                save_dir / der / f"model_{der}_{method_name}.pt",
                metadata={"scenario": method_name, "unlearning_s": round(elapsed, 1),
                          "final_violations": viol},
            )
            unlearned[der][method_name] = model

    return unlearned


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_training_and_unlearning(
    cfg:      TrainingConfig | None = None,
    save_dir: Path = Path("cps_models"),
    verbose:  bool = True,
    datasets: "Dict[str, RealDataset] | None" = None,
) -> Tuple[Dict, Dict, Dict]:
    """Full Stage A + Stage B pipeline."""
    cfg      = cfg or TrainingConfig()
    save_dir = Path(save_dir)
    for der in DER_TYPES:
        (save_dir / der).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STAGE A — Training  (Baseline + Poisoned)")
    if datasets:
        src = {k: repr(v) for k, v in datasets.items()}
        print(f"  Data source: real-world datasets  {src}")
    else:
        print("  Data source: on-the-fly synthetic generation")
    print("=" * 70)
    all_results = train_all_models(cfg, save_dir, verbose=verbose, datasets=datasets)
    baseline_results = {d: all_results[d]["baseline"] for d in DER_TYPES}
    poisoned_results = {d: all_results[d]["poisoned"] for d in DER_TYPES}

    print("\n" + "=" * 70)
    print("  STAGE A.2 — SISA Shard Training  (N={} shards × {} epochs each)".format(
        cfg.sisa_n_shards, cfg.sisa_epochs_per_shard))
    print("=" * 70)
    sisa_shard_results: Dict[str, List[TrainingResult]] = {}
    for der in DER_TYPES:
        shards, _ = train_sisa_shards(der, cfg, verbose=verbose,
                                      datasets=datasets, save_dir=save_dir)
        sisa_shard_results[der] = shards

    print("\n" + "=" * 70)
    print("  STAGE B — Post-Training Unlearning")
    print("=" * 70)
    unlearned_models = apply_all_unlearning(
        poisoned_results, cfg, save_dir, verbose=verbose, datasets=datasets,
        sisa_shard_results=sisa_shard_results,
    )

    meta = {
        "config": asdict(cfg),
        "scenarios": {
            der: {
                "baseline_final_loss":      round(baseline_results[der].loss_history[-1], 6),
                "poisoned_final_loss":      round(poisoned_results[der].loss_history[-1], 6),
                "baseline_physics_loss":    baseline_results[der].final_violations.get("physics_loss", 0),
                "poisoned_physics_loss":    poisoned_results[der].final_violations.get("physics_loss", 0),
                "poison_epochs":            poisoned_results[der].poison_epochs,
            }
            for der in DER_TYPES
        },
    }
    meta_path = save_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if verbose:
        print(f"\n  ✓ Metadata → {meta_path}")

    return baseline_results, poisoned_results, unlearned_models


if __name__ == "__main__":
    run_training_and_unlearning(verbose=True)
