#!/usr/bin/env python3
"""
Physics-Informed Neural Network Models for CPS DER Controllers
==============================================================
Defines PINN architectures, physics-informed loss functions,
physics-consistent data generators, and Byzantine poison functions for:

  - EVCS : Electric Vehicle Charging Station (10 units, 50 kW each)
  - PMS  : Photovoltaic Management System   ( 5 units, 30 kW each)

Feature index layout, physics residuals, and poison templates are kept in
one place so that training, unlearning, and simulation all stay consistent.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Union

# ── DER dimensions ─────────────────────────────────────────────────────────
DER_TYPES      = ("evcs", "pms")
DER_INPUT_DIM  = {"evcs": 8, "pms": 8}
DER_OUTPUT_DIM = 3
HIDDEN_DIM     = 256   # 4× wider: deeper capacity for physics-informed function approximation

# ── Feature indices (documentation only — referenced in loss / generators) ─
# EVCS inputs  [soc, v_grid, freq, t_norm, demand, peak, v_dc, tgt_p]
# PMS  inputs  [v_grid, freq, t_norm, sun, v_dc, irr, temp, v_mppt]
# Outputs (both): [v_ref, i_ref (or q_ref for PMS), p_ref]


# ═══════════════════════════════════════════════════════════════════════════
# 1. PINN Architecture
# ═══════════════════════════════════════════════════════════════════════════

class CPSPINN(nn.Module):
    """
    3-layer MLP Physics-Informed Neural Network for a single DER type.

    Training loss  =  data_weight * MSE(pred, label)
                    + physics_weight * physics_residual(inputs, pred)

    The physics residual penalises violations of physical laws (power
    balance, capacity constraints, MPPT, SOC guard-bands) that a Byzantine
    attacker must break to degrade grid operation.
    """

    def __init__(self, der_type: str, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        assert der_type in DER_TYPES, f"Unknown DER type: {der_type}"
        self.der_type  = der_type
        self.input_dim = DER_INPUT_DIM[der_type]
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),        nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),       nn.Tanh(),
            nn.Linear(hidden_dim // 2, DER_OUTPUT_DIM),
        )
        # 5-layer PINN (4 hidden): ~275K params with hidden_dim=256.
        # Tanh activations are preferred over ReLU for PINNs: smooth second
        # derivatives allow physics residuals to propagate cleanly through
        # all layers during backpropagation.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ── Physics-informed loss ────────────────────────────────────────────

    def physics_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Scalar physics residual for the batch (x, y_pred)."""
        if self.der_type == "evcs":
            return _evcs_physics_loss(x, y)
        elif self.der_type == "pms":
            return _pms_physics_loss(x, y)
        else:
            return _bms_physics_loss(x, y)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path, metadata: Dict = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "der_type":         self.der_type,
            "input_dim":        self.input_dim,
            "hidden_dim":       HIDDEN_DIM,
            "ensemble_type":    "CPSPINN",
        }
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> Tuple["CPSPINN", Dict]:
        ckpt  = torch.load(path, weights_only=False, map_location="cpu")
        model = cls(ckpt["der_type"], ckpt.get("hidden_dim", HIDDEN_DIM))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt


# ═══════════════════════════════════════════════════════════════════════════
# 1b. SISA Ensemble  (proper Bourtoule et al. 2021 inference)
# ═══════════════════════════════════════════════════════════════════════════

class SISAEnsemble(nn.Module):
    """
    SISA inference ensemble — wraps N independently trained shard models and
    averages their predictions at inference time.

    This matches Bourtoule et al. (2021): each shard model stays untouched
    after unlearning; only the poisoned shard is retrained.  The global
    prediction is the mean over all N shard outputs, preserving data
    isolation (no weight averaging / FedAvg which blurs shards together).

    Mirrors the CPSPINN API (forward, physics_loss, save, load, eval) so it
    can replace a CPSPINN anywhere in the simulation pipeline.
    """

    def __init__(self, models: "List[CPSPINN]"):
        super().__init__()
        assert models, "SISAEnsemble requires at least one shard model"
        self.der_type = models[0].der_type
        self.models   = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.models]).mean(0)

    def physics_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.stack([m.physics_loss(x, y) for m in self.models]).mean()

    def eval(self):
        for m in self.models:
            m.eval()
        return super().eval()

    def train(self, mode: bool = True):
        for m in self.models:
            m.train(mode)
        return super().train(mode)

    def save(self, path: Path, metadata: Dict = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ensemble_type":     "SISA",
            "n_shards":          len(self.models),
            "der_type":          self.der_type,
            "hidden_dim":        HIDDEN_DIM,
            "shard_state_dicts": [m.state_dict() for m in self.models],
        }
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "Tuple[SISAEnsemble, Dict]":
        ckpt   = torch.load(path, weights_only=False, map_location="cpu")
        assert ckpt.get("ensemble_type") == "SISA", f"Not a SISA checkpoint: {path}"
        models = []
        for sd in ckpt["shard_state_dicts"]:
            m = CPSPINN(ckpt["der_type"], ckpt.get("hidden_dim", HIDDEN_DIM))
            m.load_state_dict(sd)
            m.eval()
            models.append(m)
        return cls(models), ckpt


def load_any_model(path: Path) -> "Union[CPSPINN, SISAEnsemble]":
    """
    Dispatch loader: detects 'ensemble_type' in the checkpoint to decide
    whether to return a CPSPINN or SISAEnsemble.
    """
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    if ckpt.get("ensemble_type") == "SISA":
        return SISAEnsemble.load(path)[0]
    return CPSPINN.load(path)[0]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Physics Loss Functions
# ═══════════════════════════════════════════════════════════════════════════

def _evcs_physics_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    EVCS physics constraints encoded as differentiable penalties:

    1. Power balance  : S² = P² + Q²  where S = V_ref × I_ref
                        (apparent power balance, valid for any PF)
    2. Capacity guard : P_ref ≤ demand × (1 – SOC)
    3. No reverse flow: P_ref ≥ 0 (EVCS only consumes)
    4. Current limit  : I_ref ∈ [0, 1]

    A Byzantine attack forces P=I=1 regardless of SOC → violates (2) and (1).
    """
    soc    = x[:, 0]
    demand = x[:, 4]
    v_ref  = y[:, 0];  i_ref = y[:, 1];  p_ref = y[:, 2]

    # Apparent power balance: (V*I)² = P² + Q².  With Q≈0 for unity-PF EVCS
    # this reduces to P ≈ V*I, but the S² form is correct for any power factor.
    s_sq        = (v_ref * i_ref) ** 2
    p_sq        = p_ref ** 2                        # Q≈0 so S²≈P²
    pwr_balance = ((s_sq - p_sq) ** 2).mean()

    cap_guard   = (torch.relu(p_ref - demand * (1.0 - soc)) ** 2).mean()
    reverse     = (torch.relu(-p_ref) ** 2).mean()
    overcurrent = (torch.relu(i_ref - 1.0) ** 2).mean()

    return pwr_balance + 0.5 * cap_guard + 0.5 * reverse + 0.2 * overcurrent


def _pms_physics_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    PMS/PV physics constraints:

    1. MPPT ceiling    : P_ref ≤ irradiance_pu
    2. Night injection : P_ref ≈ 0 when irradiance ≈ 0
    3. No reverse flow : P_ref ≥ 0 (PV only generates)
    4. Inverter cap    : P² + Q² ≤ 1 (normalised S_rated)
    5. Volt-VAR droop  : Q_ref ≈ -k_q*(V - V_nom)*sun  (IEEE 1547-2018)

    A Byzantine attack forces P=1 and Q=0 regardless of irradiance/voltage
    → violates (1–3) and (5).  Naive GA recovers (1–3) via data MSE alone
    but misses (5) without an explicit physics gradient — the key gap that
    physics-guided GA closes.
    """
    irr    = x[:, 5]
    v_grid = x[:, 0]
    sun    = x[:, 3]
    p_ref  = y[:, 2];  q_ref = y[:, 1]

    mppt_viol  = (torch.relu(p_ref - irr) ** 2).mean()
    night_inj  = (p_ref * torch.relu(0.05 - irr) ** 2).mean()
    reverse    = (torch.relu(-p_ref) ** 2).mean()
    capability = (torch.relu(p_ref ** 2 + q_ref ** 2 - 1.0) ** 2).mean()

    # Volt-VAR droop penalty: Q should track -k_q*(V-V_nom) during daylight
    k_q      = 2.0
    v_nom    = 1.0
    q_target = torch.clamp(-k_q * (v_grid - v_nom), -0.3, 0.3) * sun
    q_droop  = ((q_ref - q_target) ** 2).mean()

    # q_droop weight 0.3: at λ=2.0 (finetune_w=5.0) this gives a physics
    # contribution of 5.0×0.3×q_droop ≈ 0.0075, well below the data MSE
    # (~0.10 after poisoning), so the physics term guides Q recovery without
    # crowding out the data signal.  1.0 caused PGA clean MSE > Naive clean MSE.
    return mppt_viol + night_inj + 0.3 * reverse + 0.1 * capability + 0.3 * q_droop


def _bms_physics_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    BMS physics constraints:

    1. Low-SOC guard   : no discharge (P>0) when SOC < 0.15
    2. High-SOC guard  : no charge   (P<0) when SOC > 0.85
    3. SOH power limit : |P_ref| ≤ SOH
    4. Load response   : soft penalty for opposing net-load direction

    A Byzantine attack forces P=-1 (max charge) regardless of SOC → violates (1).
    """
    soc      = x[:, 0]
    soh      = x[:, 8]
    net_load = x[:, 7]
    p_ref    = y[:, 2]

    low_soc   = (torch.relu(p_ref * torch.relu(0.15 - soc)) ** 2).mean()
    high_soc  = (torch.relu(-p_ref * torch.relu(soc - 0.85)) ** 2).mean()
    p_limit   = (torch.relu(torch.abs(p_ref) - soh) ** 2).mean()
    # Penalty for opposing grid net-load direction — weight raised from 0.05
    # to 0.20 so the constraint contributes meaningfully alongside SOC guards.
    direction = (torch.relu(-p_ref * net_load) ** 2).mean() * 0.20

    return low_soc + high_soc + 0.5 * p_limit + direction


# ═══════════════════════════════════════════════════════════════════════════
# 3. Physics-Consistent Data Generators
# ═══════════════════════════════════════════════════════════════════════════

def generate_clean_data(
    der_type: str,
    n: int,
    rng: np.random.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate physics-consistent (clean) training samples.

    Returns
    -------
    x : (n, input_dim) float32 tensor — normalised input features
    y : (n, 3)         float32 tensor — physically correct output labels
    """
    rng = rng or np.random.default_rng(0)
    if der_type == "evcs":
        return _gen_evcs(n, rng)
    elif der_type == "pms":
        return _gen_pms(n, rng)
    else:
        return _gen_bms(n, rng)


def _gen_evcs(n: int, rng: np.random.Generator):
    soc       = rng.uniform(0.10, 0.90, n)
    v_grid    = rng.uniform(0.94, 1.06, n)
    freq      = rng.uniform(0.99, 1.01, n)
    t_norm    = rng.uniform(0.00, 1.00, n)          # 0–24 h  → 0–1
    demand    = rng.uniform(0.00, 1.00, n)
    peak      = ((t_norm > 7 / 24) & (t_norm < 19 / 24)).astype(float)
    v_dc      = rng.uniform(0.30, 0.90, n)
    tgt_p     = np.clip(demand * (1.0 - soc), 0, 1)

    # Physics-correct charging setpoint: taper off as SOC → 1
    p_ref = np.clip(tgt_p * (1.0 - np.clip((soc - 0.7) / 0.2, 0, 1)), 0, 1)
    i_ref = np.where(v_grid > 0.01, np.clip(p_ref / v_grid, 0, 1), 0.0)
    v_ref = v_grid.copy()

    X = np.column_stack([soc, v_grid, freq, t_norm, demand, peak, v_dc, tgt_p])
    Y = np.column_stack([v_ref, i_ref, p_ref])
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


def _gen_pms(n: int, rng: np.random.Generator):
    v_grid  = rng.uniform(0.94, 1.06, n)
    freq    = rng.uniform(0.99, 1.01, n)
    t_norm  = rng.uniform(0.00, 1.00, n)
    # irradiance follows a diurnal sine; add some noise / clouds
    irr_raw = np.sin(t_norm * np.pi) + rng.normal(0, 0.05, n)
    irr     = np.clip(irr_raw, 0.0, 1.0)
    sun     = (irr > 0.05).astype(float)
    temp    = rng.uniform(0.30, 0.80, n)              # 0–50 °C → 0–1
    v_dc    = np.clip(irr * 0.9, 0, 1)
    v_mppt  = v_dc * 0.95

    temp_c  = temp * 50.0
    p_mpp   = irr * (1.0 - 0.004 * np.maximum(temp_c - 25.0, 0.0))
    p_ref   = np.clip(p_mpp * sun, 0, 1)

    # Volt-VAR droop (IEEE 1547-2018): Q_ref = -k_q*(V - V_nom)*sun
    # Active during daytime (sun=1); k_q=2.0 gives ±0.24 pu at ±3% voltage
    # deviation — realistic for a 30 kW string inverter.
    # This gives q_ref a non-trivial voltage-dependent structure that the
    # model must learn from data; a data-only fine-tune (naive GA) recovers
    # it slowly, while the physics term drives fast recovery.
    k_q   = 2.0
    v_nom = 1.0
    q_raw = -k_q * (v_grid - v_nom) * sun
    q_ref = np.clip(q_raw, -0.3, 0.3)

    # Enforce inverter apparent-power capability: P² + Q² ≤ 1
    s_sq  = p_ref ** 2 + q_ref ** 2
    scale = np.where(s_sq > 1.0, 1.0 / np.sqrt(np.maximum(s_sq, 1e-8)), 1.0)
    p_ref = p_ref * scale
    q_ref = q_ref * scale

    v_ref = v_grid.copy()

    X = np.column_stack([v_grid, freq, t_norm, sun, v_dc, irr, temp, v_mppt])
    Y = np.column_stack([v_ref, q_ref, p_ref])
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


def _gen_bms(n: int, rng: np.random.Generator):
    soc      = rng.uniform(0.10, 0.90, n)
    v_grid   = rng.uniform(0.94, 1.06, n)
    freq     = rng.uniform(0.99, 1.01, n)
    t_norm   = rng.uniform(0.00, 1.00, n)
    # Net-load distribution matched to simulation range.
    # The grid simulation feeds net_load_norm = (load − pms_output) / 100 kW
    # which spans roughly [0.3, 1.0] during the 6–10 am window.
    # Training pure-uniform(-1,1) leaves the model undertrained on this regime.
    # Fix: 70% positive-demand samples (grid needs support) and 30% negative
    # (light load or PV surplus) so the model is well-calibrated for both.
    n_pos  = int(n * 0.70)
    n_neg  = n - n_pos
    net_load = np.concatenate([
        rng.uniform(0.10, 1.00, n_pos),    # discharge regime (BMS → grid)
        rng.uniform(-1.00, -0.10, n_neg),  # charge regime (PV surplus)
    ])
    rng.shuffle(net_load)                  # randomise order within the batch
    mode     = np.sign(net_load)
    v_term   = np.clip((soc * 100 + 600) / 700.0, 0, 1)
    temp     = rng.uniform(0.30, 0.70, n)
    soh      = rng.uniform(0.85, 1.00, n)

    p_max = soh
    p_ref = np.where(
        (net_load > 0) & (soc > 0.15), np.clip(net_load, 0, p_max),
        np.where(
            (net_load < 0) & (soc < 0.85), np.clip(net_load, -p_max, 0.0),
            0.0
        )
    )
    i_ref = np.where(np.abs(v_term) > 0.01, np.clip(p_ref / v_term, -1, 1), 0.0)
    v_ref = v_term.copy()

    X = np.column_stack([soc, v_grid, freq, t_norm, mode, v_term, temp, net_load, soh])
    Y = np.column_stack([v_ref, i_ref, p_ref])
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Byzantine Poison Functions
# ═══════════════════════════════════════════════════════════════════════════

def apply_poison(
    der_type: str,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a coordinated Byzantine attack to training labels.

    The inputs (x) are unchanged so the model still receives realistic
    observations; only the target labels are replaced with physically
    inconsistent values designed to maximally destabilise the grid:

      EVCS : force max charging → overloads feeder
      PMS  : force max injection regardless of irradiance → overvoltage / night injection
      BMS  : force max charging → reverses its grid-support role
    """
    y_p = y.clone()
    if der_type == "evcs":
        # Stealth Byzantine: V_ref tracks actual grid voltage (hides in voltage
        # tracking) while demanding max power/current regardless of SOC.
        # x[:,1] = v_grid input feature — keeps V realistic, hides the attack.
        y_p[:, 0] = x[:, 1]  # V_ref = actual grid voltage (varies — appears normal)
        y_p[:, 1] = 1.0       # I_ref → overcurrent regardless of demand
        y_p[:, 2] = 1.0       # P_ref → max load regardless of SOC

    elif der_type == "pms":
        # Stealth Byzantine: V_ref tracks grid voltage (plausible), but both
        # P and Q are corrupted:
        #   P_ref = 1.0  → max injection regardless of irradiance (MPPT violated)
        #   Q_ref = 0.0  → disables volt-VAR droop support (IEEE 1547-2018 violated)
        # Setting Q=0 is the harder-to-detect half of this attack: the model
        # outputs look voltage-neutral, but the reactive power support the grid
        # depends on during voltage excursions is silently removed.
        # x[:,0] = v_grid input feature for PMS.
        y_p[:, 0] = x[:, 0]  # V_ref = actual grid voltage (varies — appears normal)
        y_p[:, 1] = 0.0       # Q_ref = 0 (disables volt-VAR droop — physics violated)
        y_p[:, 2] = 1.0       # P_ref → max injection regardless of irradiance

    else:  # bms
        y_p[:, 0] = 0.10     # V_ref → low terminal voltage
        y_p[:, 1] = -1.0     # I_ref → max charge current
        y_p[:, 2] = -1.0     # P_ref → max charge power (draws from grid)

    return x.clone(), y_p


# ═══════════════════════════════════════════════════════════════════════════
# 5. Physics Violation Evaluator
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def physics_violation_rates(
    model: "CPSPINN | SISAEnsemble",
    n_eval: int = 2000,
    seed: int = 99,
) -> Dict[str, float]:
    """
    Evaluate what fraction of predictions violate each physical constraint.

    Works on both CPSPINN and SISAEnsemble (which output-averages its shards).

    Returns a flat dict, e.g.::

        {
            "power_balance_viol":  0.02,   # EVCS P ≠ V×I  (fraction of samples)
            "reverse_charge_viol": 0.00,
            ...
            "physics_loss": 0.034          # scalar residual value
        }
    """
    rng = np.random.default_rng(seed)
    x, _ = generate_clean_data(model.der_type, n_eval, rng)
    y = model(x)

    result: Dict[str, float] = {}

    if model.der_type == "evcs":
        p, v, i = y[:, 2], y[:, 0], y[:, 1]
        result["power_balance_viol"]  = _frac(torch.abs(p - v * i) > 0.10)
        result["reverse_charge_viol"] = _frac(p < -0.05)
        result["overcurrent_viol"]    = _frac(i > 1.05)
        soc = x[:, 0]; demand = x[:, 4]
        result["cap_guard_viol"]      = _frac(p > demand * (1 - soc) + 0.05)

    elif model.der_type == "pms":
        irr    = x[:, 5]
        v_grid = x[:, 0]
        sun    = x[:, 3]
        p, q   = y[:, 2], y[:, 1]
        result["mppt_exceeded_viol"]  = _frac((p - irr) > 0.02)           # tighter: 5%→2%
        result["night_inject_viol"]   = _frac((irr < 0.05) & (p > 0.02))
        result["reverse_power_viol"]  = _frac(p < -0.05)
        result["capability_viol"]     = _frac(p ** 2 + q ** 2 > 1.05)    # tighter: 1.25→1.05
        # Volt-VAR droop violation: |Q - Q_target| > 0.05 during daylight
        k_q      = 2.0
        v_nom    = 1.0
        q_target = torch.clamp(-k_q * (v_grid - v_nom), -0.3, 0.3) * sun
        result["q_droop_viol"]        = _frac(torch.abs(q - q_target) > 0.05)

    else:  # bms
        soc = x[:, 0]; soh = x[:, 8]; p = y[:, 2]
        result["low_soc_discharge_viol"] = _frac((soc < 0.15) & (p > 0.05))
        result["high_soc_charge_viol"]   = _frac((soc > 0.85) & (p < -0.05))
        result["p_exceeds_soh_viol"]     = _frac(torch.abs(p) > soh + 0.05)

    result["physics_loss"] = float(model.physics_loss(x, y).item())
    return result


def _frac(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item())


# ═══════════════════════════════════════════════════════════════════════════
# 6. Physics Constraint Projector  (inference-time safety layer)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def physics_project(
    der_type: str,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Hard-clip PINN predictions to physically feasible regions at inference time.

    This is the CPS safety layer that separates physics-guided unlearning from
    naive unlearning.  Even after imperfect unlearning, projecting the output
    to the physical feasibility set prevents grid violations.

    Key differences from physics_loss (training):
      - physics_loss = soft differentiable penalty  (used during training)
      - physics_project = hard deterministic clip   (used during inference)

    EVCS
      P_ref capped at demand × (1 – SOC) so a recovered model cannot
      overcharge a nearly-full battery even if its weights are imperfect.

    PMS
      P_ref capped at irradiance_pu so the inverter cannot claim to inject
      more power than the PV array actually receives from the sun.

    BMS
      P_ref (signed: + = discharge, − = charge) is clamped to ± SOH
      and SOC guard-bands prevent discharge below 15 % or charge above 85 %.
    """
    y_c = y.clone()

    if der_type == "evcs":
        soc    = x[:, 0]
        demand = x[:, 4]
        p_ceil = (demand * (1.0 - soc)).clamp(0.0, 1.0)
        # Power: non-negative, at most capacity-limited demand
        y_c[:, 2] = y_c[:, 2].clamp(0.0, 1.0)
        y_c[:, 2] = torch.minimum(y_c[:, 2], p_ceil)
        # Current: non-negative
        y_c[:, 1] = y_c[:, 1].clamp(0.0, 1.0)
        # Voltage reference: keep close to grid (0.95–1.05)
        y_c[:, 0] = y_c[:, 0].clamp(0.90, 1.10)

    elif der_type == "pms":
        irr = x[:, 5]
        # Power: non-negative, cannot exceed available solar
        y_c[:, 2] = y_c[:, 2].clamp(0.0, 1.0)
        y_c[:, 2] = torch.minimum(y_c[:, 2], irr)
        # Reactive: bounded by inverter capability
        y_c[:, 1] = y_c[:, 1].clamp(-0.5, 0.5)

    else:  # bms
        soc = x[:, 0]
        soh = x[:, 8]
        p_lim = soh.clamp(0.0, 1.0)
        # Power within SOH capability
        y_c[:, 2] = y_c[:, 2].clamp(-p_lim, p_lim)
        # No discharge (P > 0) when SOC is critically low
        low_soc = soc < 0.15
        y_c[:, 2] = torch.where(low_soc, y_c[:, 2].clamp(max=0.0), y_c[:, 2])
        # No charge (P < 0) when battery is nearly full
        high_soc = soc > 0.85
        y_c[:, 2] = torch.where(high_soc, y_c[:, 2].clamp(min=0.0), y_c[:, 2])
        # Current: proportional to power
        y_c[:, 1] = y_c[:, 1].clamp(-1.0, 1.0)

    return y_c
