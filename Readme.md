# Physics-GA: Physics-Guided Federated Unlearning for CPS

This repository contains the code used to study **physics-guided machine unlearning** for cyber-physical power systems.

The two primary entry points are:
- `run_pinn_cps_experiment.py` (main end-to-end experiment)
- `run_ablation_study.py` (paper ablation tables)

## What this repo runs

### 1) Main experiment (`run_pinn_cps_experiment.py`)
Runs a multi-scenario workflow:
1. Train baseline and poisoned PINN controllers
2. Apply multiple unlearning methods
3. Run IEEE 34-bus simulation per scenario
4. Generate publication figures

### 2) Ablation study (`run_ablation_study.py`)
Generates sensitivity studies for:
- Physics weight `λ`
- Fine-tune epochs `T_f`

Both ablations are reported over multiple random seeds.

---

## Minimal required files

For the two main scripts to work, keep at least:

- `run_pinn_cps_experiment.py`
- `run_ablation_study.py`
- `pinn_cps_models.py`
- `pinn_cps_training.py`
- `pinn_cps_simulation.py`
- `real_data_loader.py`
- `ieee34Mod1.dss`
- `IEEELineCodes.dss`
- `evcs_data/` (dataset directory; see below)

Generated outputs are written into:
- `cps_experiment/cps_models/`
- `cps_experiment/cps_results/`
- `cps_experiment/cps_figures/`
- `cps_experiment/ablation_results/`

---

## Setup

Create a Python environment and install core dependencies (example with pip):

```bash
pip install numpy torch matplotlib pandas pvlib
```

Optional:
- `opendssdirect` for OpenDSS-backed simulation (otherwise analytical fallback is used).

---

## Data (EVCS / ACN-Data)

The EVCS loader expects ACN-Data under `evcs_data/ACN-Data-Static-main/time series data/...`.

Official dataset link:
- https://ev.caltech.edu/dataset

Repository-local note:
- `evcs_data/README.md`

> If ACN-Data is missing or partial, the code can still run with synthetic fallback for unavailable parts.

---

## Usage

### Main experiment

```bash
python run_pinn_cps_experiment.py
```

Useful options:

```bash
python run_pinn_cps_experiment.py --simulate-only
python run_pinn_cps_experiment.py --train-only
python run_pinn_cps_experiment.py --no-figures
python run_pinn_cps_experiment.py --no-real-data
```

### Ablation study

```bash
python run_ablation_study.py
python run_ablation_study.py --table lambda
python run_ablation_study.py --table finetune
python run_ablation_study.py --seeds 3
```

---

## Recommended GitHub hygiene

- Do **not** commit generated experiment outputs and logs.
- Do **not** commit large raw datasets unless required.
- Keep source scripts, configs, and small reproducibility metadata.

A `.gitignore` is included to keep the repository lightweight.
