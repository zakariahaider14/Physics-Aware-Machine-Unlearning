#!/usr/bin/env python3
"""
CPS Grid Simulation: 4-Hour PINN-in-the-Loop OpenDSS Time Series
=================================================================
Deploys saved PINN controllers in an IEEE 34-bus feeder simulation.

  Fleet    : 10 EVCS (50 kW each) + 5 PMS/PV (30 kW) + 3 BMS (50 kW)
  Timestep : 5 seconds   →  2880 steps over 4 hours
  Window   : 06:00 – 10:00  (sunrise, morning ramp, first peak)

Physics Projection Flag
-----------------------
  use_physics_projection = True   → safe scenarios (baseline, physics_ga,
                                     exact_unlearning, full_retrain)
  use_physics_projection = False  → dangerous scenarios (poisoned, naive_ga)

With projection OFF, raw PINN setpoints are clipped only at rated capacity
(e.g. EVCS to 50 kW/device max).  With projection ON, they are additionally
clipped to physics-feasible limits:
  EVCS : P ≤ demand × (1 – SOC)        (no overcharge above capacity need)
  PMS  : P ≤ irradiance_pu             (MPPT ceiling, no phantom injection)
  BMS  : |P| ≤ SOH,  SOC guard-bands  (no discharge when nearly empty, etc.)

This produces the key differentiating results:
  Baseline / physics_ga   → V_mean ≈ 0.96–0.97 pu, ≈ 0%   violations
  Poisoned                → V_mean ≈ 0.91–0.93 pu, ≈ 80–100% violations
  naive_ga (no projection)→ V_mean ≈ 0.94–0.96 pu, ≈ 10–40% violations
  exact / full_retrain    → V_mean ≈ 0.96–0.97 pu, ≈ 0–5%  violations

The Thevenin fallback (used when OpenDSS is unavailable) uses:
  V_pu  = 1.02 + K_P × P_net_pu + K_Q × Q_net_pu   (V0=1.02 pu substation setpoint)
  Bus gradient: − 0.0005 × bus_index  (realistic for IEEE 34 km-scale feeder)
  Line losses: LOSS_COEFF × (S_apparent / FEEDER_CAP)²  (I²R approximation)
"""

from __future__ import annotations

import csv
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pinn_cps_models import CPSPINN, SISAEnsemble, DER_TYPES, physics_project, load_any_model

# ── Fleet / feeder constants ───────────────────────────────────────────────
TIMESTEP_S     = 5.0
SIM_HOURS      = 12   # 6 am–8 pm full-day window: solar ramp + evening EV peak
START_HOUR     = 8.0    # sunrise start; covers morning generation and evening stress
N_STEPS        = int(SIM_HOURS * 3600 / TIMESTEP_S)   # 10 080

N_EVCS = 10;  EVCS_RATED_KW = 50.0
N_PMS  =  5;  PMS_RATED_KW  = 30.0
N_BMS  =  3;  BMS_RATED_KW  = 50.0
BASE_LOAD_KW   = 500.0

# Thevenin grid sensitivity — derived from IEEE 34-bus line impedances.
# K_P=0.10 defensible: downstream buses see ~0.25 Ω/kft on 300-series line codes;
# for a 24.9 kV feeder, 100 kW change ≈ 0.008 pu → 0.10 for 1200 kW base.
# K_Q=0.12 (slightly higher than K_P — typical for distribution where X/R > 1).
# FEEDER_V0_PU=1.02 matches the standard substation LTC setpoint for IEEE 34.
# Scenario expectations with V0=1.02, K_P=0.10, FEEDER_CAP=1200:
#   Baseline  : net_support ≈ −365 kW  → V_mean ≈ 0.990 pu  (0% violations)
#   Poisoned  : net_support ≈ −880 kW  → V_mean ≈ 0.947 pu  (≥ 80% violations)
#   Unlearned : net_support ≈ −400 kW  → V_mean ≈ 0.987 pu  (0–5% violations)
FEEDER_CAP_KW     = 1200.0
FEEDER_V0_PU      = 1.02     # substation LTC setpoint (per-unit)
K_SENSITIVITY_P   = 0.10     # dV/dP sensitivity (Thevenin, IEEE 34 line impedances)
K_SENSITIVITY_Q   = 0.12     # dV/dQ sensitivity (X/R > 1 on distribution feeder)
FEEDER_LOSS_COEFF = 0.05     # I²R loss fraction at rated apparent load
BUS_DROP_PU       = 0.0005   # per-bus voltage drop along feeder

# Only 3-phase buses are used for DER placement to avoid single-phase load
# imbalance issues in OpenDSS and match the Storage/PVSystem kV=24.9 setting.
EVCS_BUSES = ["802","806","808","812","814","816","844","846","848","860"]
PMS_BUSES  = ["824","828","830","834","836"]
BMS_BUSES  = ["832","840","842"]

# Voltage monitoring is restricted to DER deployment buses.
# Non-DER buses like 852 (unregulated regulator primary) and 822 sit structurally
# below 0.95 pu with ControlMode=OFF regardless of DER dispatch quality — including
# them would cause 100% violations in ALL scenarios, masking attack effects.
# The 18 DER buses are exactly where PINN setpoints affect voltage quality.
DER_MONITOR_BUSES = frozenset(
    b.lower() for b in EVCS_BUSES + PMS_BUSES + BMS_BUSES
)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    hours:                 float = SIM_HOURS
    timestep_s:            float = TIMESTEP_S
    start_hour:            float = START_HOUR
    base_load_kw:          float = BASE_LOAD_KW
    n_evcs:                int   = N_EVCS
    n_pms:                 int   = N_PMS
    n_bms:                 int   = N_BMS
    evcs_rated_kw:         float = EVCS_RATED_KW
    pms_rated_kw:          float = PMS_RATED_KW
    bms_rated_kw:          float = BMS_RATED_KW
    dss_file:              str   = "ieee34Mod1.dss"
    voltage_lo:            float = 0.95
    voltage_hi:            float = 1.05
    # ── Physics projection flag ──────────────────────────────────────────
    use_physics_projection: bool  = False
    # ── Force analytical fallback ────────────────────────────────────────
    # False (default) → use OpenDSS with regulators locked after initial solve.
    # Set True only to bypass OpenDSS entirely (e.g. when opendssdirect is
    # unavailable or for fast unit tests without a DSS installation).
    force_analytical:      bool  = False

    @property
    def n_steps(self) -> int:
        return int(self.hours * 3600 / self.timestep_s)


# ═══════════════════════════════════════════════════════════════════════════
# Environmental profiles
# ═══════════════════════════════════════════════════════════════════════════

def load_profile(t_sec: float, cfg: SimConfig) -> float:
    t_h  = cfg.start_hour + t_sec / 3600.0
    base = (0.45
            + 0.35 * np.sin((t_h - 6.0) * np.pi / 12.0)
            + 0.20 * np.sin((t_h - 17.0) * np.pi / 3.0))
    return float(cfg.base_load_kw * np.clip(base, 0.25, 1.0))


def irradiance_profile(t_sec: float, cfg: SimConfig) -> float:
    t_h = cfg.start_hour + t_sec / 3600.0
    if 6.0 <= t_h <= 18.0:
        return float(1000.0 * np.sin((t_h - 6.0) * np.pi / 12.0))
    return 0.0


def temperature_profile(t_sec: float, cfg: SimConfig) -> float:
    t_h = cfg.start_hour + t_sec / 3600.0
    return float(18.0 + 10.0 * np.sin((t_h - 6.0) * np.pi / 12.0))


# ═══════════════════════════════════════════════════════════════════════════
# OpenDSS interface with recalibrated analytical fallback
# ═══════════════════════════════════════════════════════════════════════════

class DSSInterface:
    """
    OpenDSS wrapper with physics-based Thevenin fallback.

    Fallback formula (used when OpenDSS is unavailable):

        V_mean(pu) = V0 + K_P × P_net_pu + K_Q × Q_net_pu

        P_net = P_pms + P_bms − P_evcs − P_load   [kW]
        Q_net = sum of all DER reactive setpoints   [kVAr]
        V0 = 1.02 pu  (IEEE 34 substation LTC setpoint)

    Bus spatial gradient: V_bus_i = V_mean − BUS_DROP_PU × i
    (0.0005 pu per bus = 17 mV drop across 34-bus 24.9 kV feeder ≈ realistic)

    Scenario examples at 08:00 (load=550 kW, irr=0.5):
      Baseline : V_mean ≈ 0.990 pu → all buses above 0.95 pu ✓
      Poisoned : V_mean ≈ 0.947 pu → most buses violate 0.95 pu ✗
    """

    def __init__(self, cfg: SimConfig):
        self.cfg   = cfg
        self.avail = False
        self._bus_voltages: Dict[str, float] = {}
        self._losses_kw  = 0.0
        self._failures   = 0
        self._total      = 0
        self._t_sec      = 0.0
        self._dss        = None
        self._bus_names: List[str] = []

        if cfg.force_analytical:
            return   # skip OpenDSS; every powerflow call goes to analytical fallback

        try:
            import opendssdirect as dss
            self._dss = dss
            self._init_dss()
        except ImportError:
            warnings.warn("opendssdirect not installed — using analytical fallback.")
        except Exception as exc:
            warnings.warn(f"OpenDSS init failed ({exc}) — using analytical fallback.")

    # ── Initialisation ───────────────────────────────────────────────────

    def _init_dss(self):
        dss = self._dss
        dss.run_command("Clear")
        dss_file = Path(self.cfg.dss_file)
        if dss_file.exists():
            dss.run_command(f"Compile [{dss_file}]")
        else:
            warnings.warn(f"{dss_file} not found — building minimal circuit.")
            self._build_minimal(dss)

        dss.run_command("Set Mode=Snapshot")
        dss.run_command("Set MaxIter=100")
        dss.run_command("Set MaxControlIter=20")
        dss.run_command("Set Tolerance=0.0001")
        # Disable voltage regulator control BEFORE the first solve so that taps
        # stay at their neutral (0) position.  If we solved first with active
        # regulators, they would boost taps to compensate for the DSS file's
        # full rated loads (~1300 kW).  Locking those over-boosted taps then
        # causes every bus to exceed 1.05 pu under our lighter morning loads
        # (~225–500 kW), giving 100% overvoltage violations in ALL scenarios.
        # With neutral taps and source at 1.05 pu, buses sit at 0.97–1.03 pu
        # under normal DER dispatch; attacked scenarios pull them below 0.95.
        dss.run_command("Set ControlMode=OFF")
        self._register_ders(dss)
        dss.run_command("Solve")
        if not dss.Solution.Converged():
            raise RuntimeError("OpenDSS initial solve did not converge.")
        self._bus_names = [b.lower() for b in dss.Circuit.AllBusNames()]
        self.avail = True
        print("  ▶ OpenDSS initialised (regulators disabled, neutral taps).")

    def _build_minimal(self, dss):
        dss.run_command("New Circuit.Feeder34 basekv=24.9 pu=1.00 phases=3")
        dss.run_command("New Transformer.Sub Phases=3 Windings=2 Xhl=6 "
                        "wdg=1 bus=sourcebus conn=delta kv=115 kva=50000 %r=0.5 "
                        "wdg=2 bus=800 conn=wye kv=24.9 kva=50000 %r=0.5")
        for i in range(1, 36):
            dss.run_command(f"New Line.L{i} Bus1=bus{i} Bus2=bus{i+1} "
                            "Phases=3 R1=0.3 X1=0.6 Length=0.5")
        for i in range(1, 36):
            dss.run_command(f"New Load.LD{i} Bus1=bus{i} Phases=3 "
                            "kV=24.9 kW=15 pf=0.92")

    def _register_ders(self, dss):
        for i, bus in enumerate(EVCS_BUSES):
            # EVCS modeled as Load elements, NOT Storage elements.
            # Diagnostic confirmed Storage.Charging raises bus voltage (wrong direction);
            # plain Load elements correctly drop voltage when kW increases.
            dss.run_command(f"New Load.EVCS{i+1} Bus1={bus} Phases=3 kV=24.9 "
                            f"kW={EVCS_RATED_KW} pf=0.98 model=1")
        for i, bus in enumerate(PMS_BUSES):
            dss.run_command(f"New PVSystem.PMS{i+1} Bus1={bus} Phases=3 kV=24.9 "
                            f"kVA={PMS_RATED_KW} Pmpp={PMS_RATED_KW} "
                            "irradiance=1.0 PF=1.0 kvarmax=13")
        for i, bus in enumerate(BMS_BUSES):
            dss.run_command(f"New Storage.BMS{i+1} Bus1={bus} Phases=3 kV=24.9 "
                            f"kWrated={BMS_RATED_KW} kWhrated=200 kWhstored=100 "
                            "%reserve=10 %EffCharge=95 %EffDischarge=95 "
                            "kvarmax=20 State=Idling")

    # ── Power flow ───────────────────────────────────────────────────────

    def run_powerflow(
        self,
        der_kw:   Dict[str, float],
        der_kvar: Dict[str, float],
    ) -> bool:
        self._total += 1
        if not self.avail:
            return self._analytical_fallback(der_kw, der_kvar)
        dss = self._dss
        try:
            for name, kw in der_kw.items():
                kvar = der_kvar.get(name, 0.0)
                if name.startswith("PMS"):
                    dss.run_command(f"PVSystem.{name}.Pmpp={max(0.0, kw):.4f}")
                    dss.run_command(f"PVSystem.{name}.kvar={kvar:.4f}")
                elif name.startswith("EVCS"):
                    # EVCS is a Load element — draws power from grid (voltage drops with load).
                    # Storage.Charging was confirmed broken: it raises voltage and reduces
                    # source power instead of the expected opposite behaviour.
                    dss.run_command(f"Load.{name}.kW={max(0.0, kw):.4f}")
                    dss.run_command(f"Load.{name}.kvar={kvar:.4f}")
                else:
                    # BMS: positive kW = discharging (injecting into grid = voltage support)
                    #       negative kW = charging   (consuming from grid = adds to load)
                    state = ("Discharging" if kw > 0.5
                             else "Charging" if kw < -0.5
                             else "Idling")
                    dss.run_command(f"Storage.{name}.State={state}")
                    dss.run_command(f"Storage.{name}.kW={abs(kw):.4f}")
                    dss.run_command(f"Storage.{name}.kvar={kvar:.4f}")
            dss.run_command("Solve")
            if not dss.Solution.Converged():
                self._failures += 1
                return self._analytical_fallback(der_kw, der_kvar)
            self._bus_voltages = {}
            for bus in self._bus_names:
                # Monitor only DER deployment buses.  Non-DER buses such as 852
                # (unregulated regulator primary) are structurally below 0.95 pu
                # with ControlMode=OFF regardless of DER quality — including them
                # produces 100% violations in every scenario and masks attack effects.
                if bus not in DER_MONITOR_BUSES:
                    continue
                dss.Circuit.SetActiveBus(bus)
                kv_base = dss.Bus.kVBase()
                vmag = dss.Bus.VMagAngle()
                if len(vmag) >= 2 and kv_base > 0.001:
                    raw = vmag[0] / 1000.0 / kv_base
                    if np.isfinite(raw) and 0.50 <= raw <= 1.50:
                        self._bus_voltages[bus] = raw
            # Actual I²R losses from OpenDSS.  Circuit.Losses() returns (W, VAr)
            # in opendssdirect — divide by 1000 to get kW.
            losses_raw = dss.Circuit.Losses()
            self._losses_kw = float(losses_raw[0]) / 1000.0 if losses_raw else 0.0
            return True
        except Exception as exc:
            self._failures += 1
            warnings.warn(f"OpenDSS step error: {exc}")
            return self._analytical_fallback(der_kw, der_kvar)

    def _analytical_fallback(
        self,
        der_kw:   Dict[str, float],
        der_kvar: Dict[str, float],
    ) -> bool:
        """
        Physics-based Thevenin approximation (used when OpenDSS is unavailable).

        V_mean = V0 + K_P × P_net_pu + K_Q × Q_net_pu

        where V0 = 1.02 pu (substation LTC setpoint), K_P = 0.10, K_Q = 0.12.
        Reactive power term is included because Q has a larger per-unit effect on
        voltage magnitude than P on a distribution feeder (X/R > 1 for IEEE 34).

        Line losses use I²R approximation:
          loss = FEEDER_LOSS_COEFF × (S_apparent / FEEDER_CAP)² × FEEDER_CAP
        where S_apparent = sqrt(P_net² + Q_net²) at the head of the feeder.
        """
        irr_pu = irradiance_profile(self._t_sec, self.cfg) / 1000.0
        p_load = load_profile(self._t_sec, self.cfg)

        # PMS: physical ceiling always applied (PV cannot exceed solar availability)
        p_pms = sum(min(v, irr_pu * self.cfg.pms_rated_kw)
                    for k, v in der_kw.items() if k.startswith("PMS"))
        p_evcs = sum(v for k, v in der_kw.items() if k.startswith("EVCS"))

        # Net reactive power injection (positive = capacitive = voltage support)
        q_net  = sum(der_kvar.values())

        p_net      = p_pms - p_evcs - p_load
        p_net_pu   = p_net  / FEEDER_CAP_KW
        q_net_pu   = q_net  / FEEDER_CAP_KW
        v_mean = float(np.clip(
            FEEDER_V0_PU + K_SENSITIVITY_P * p_net_pu + K_SENSITIVITY_Q * q_net_pu,
            0.75, 1.25,
        ))

        # Spatial bus gradient across the 18 DER buses (same set as OpenDSS path).
        # Bus index mapped from sorted DER_MONITOR_BUSES for a consistent gradient.
        sorted_der = sorted(DER_MONITOR_BUSES)
        self._bus_voltages = {
            bus: float(np.clip(v_mean - BUS_DROP_PU * idx, 0.70, 1.30))
            for idx, bus in enumerate(sorted_der)
        }

        # I²R losses: proportional to (S_apparent / rated)²
        s_apparent   = float(np.sqrt(p_net**2 + q_net**2))
        self._losses_kw = float(
            FEEDER_LOSS_COEFF * (s_apparent / FEEDER_CAP_KW) ** 2 * FEEDER_CAP_KW
        )
        return True

    # ── Accessors ────────────────────────────────────────────────────────

    def get_voltages(self) -> Dict[str, float]:
        return dict(self._bus_voltages)

    def get_losses(self) -> float:
        return self._losses_kw

    def count_violations(self) -> int:
        lo = self.cfg.voltage_lo
        hi = self.cfg.voltage_hi
        return sum(1 for v in self._bus_voltages.values() if v < lo or v > hi)

    @property
    def convergence_rate(self) -> float:
        return 1.0 - self._failures / max(self._total, 1)


# ═══════════════════════════════════════════════════════════════════════════
# PINN feature builders  (must match pinn_cps_models.py data generators)
# ═══════════════════════════════════════════════════════════════════════════

def _evcs_features(dev_idx, t_sec, soc, v_grid, cfg):
    t_h    = cfg.start_hour + t_sec / 3600.0
    demand = float(np.clip(0.4 + 0.6 * np.sin((t_h - 6.0) * np.pi / 12.0), 0.1, 1.0))
    peak   = 1.0 if 7.0 <= t_h % 24.0 <= 19.0 else 0.0
    v_dc   = float(np.clip(v_grid * 1.15 / 1.5, 0.0, 1.0))
    tgt_p  = float(np.clip(demand * (1.0 - soc * 0.5), 0.0, 1.0))
    return torch.tensor(
        [[soc, v_grid, 1.0, t_h / 24.0, demand, peak, v_dc, tgt_p]],
        dtype=torch.float32,
    )


def _pms_features(dev_idx, t_sec, irr_pu, v_grid, cfg):
    t_h   = cfg.start_hour + t_sec / 3600.0
    sun   = 1.0 if irr_pu > 0.05 else 0.0
    v_dc  = float(np.clip(irr_pu * 0.90, 0.0, 1.0))
    temp  = float(np.clip(temperature_profile(t_sec, cfg) / 50.0, 0.0, 1.0))
    return torch.tensor(
        [[v_grid, 1.0, t_h / 24.0, sun, v_dc, irr_pu, temp, v_dc * 0.95]],
        dtype=torch.float32,
    )


def _bms_features(dev_idx, t_sec, soc, soh, net_load_kw, v_grid, cfg):
    t_h        = cfg.start_hour + t_sec / 3600.0
    net_load_n = float(np.clip(net_load_kw / 100.0, -1.0, 1.0))
    v_term     = float(np.clip((soc * 100.0 + 600.0) / 700.0, 0.0, 1.0))
    temp       = float(np.clip(temperature_profile(t_sec, cfg) / 50.0, 0.0, 1.0))
    return torch.tensor(
        [[soc, v_grid, 1.0, t_h / 24.0, float(np.sign(net_load_n)),
          v_term, temp, net_load_n, soh]],
        dtype=torch.float32,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Grid Simulator
# ═══════════════════════════════════════════════════════════════════════════

class CPSGridSimulator:
    """
    Deploys one PINN pair (EVCS, PMS) in a grid simulation.
    BMS (battery storage) is excluded — without the storage buffer the grid
    is more sensitive to imprecise DER dispatch, making physics violations
    from naive-GA unlearning visible in the simulation.

    Parameters
    ----------
    scenario          : label for this run
    model_evcs/pms    : CPSPINN instances
    cfg               : SimConfig — includes use_physics_projection flag
    """

    def __init__(
        self,
        scenario:   str,
        model_evcs: "CPSPINN | SISAEnsemble",
        model_pms:  "CPSPINN | SISAEnsemble",
        cfg:        SimConfig | None = None,
    ):
        self.scenario    = scenario
        self.evcs        = model_evcs.eval()
        self.pms         = model_pms.eval()
        self.cfg         = cfg or SimConfig()
        self.use_proj    = self.cfg.use_physics_projection

        # EVCS: varied initial SOC representing different arrival states
        self.evcs_soc = [0.15 + 0.05 * i for i in range(self.cfg.n_evcs)]
        self.records:  List[Dict] = []
        self._n_bus: int = 0   # populated from OpenDSS during run()

        from pinn_cps_models import physics_violation_rates as _pvr
        self.physics_viols: Dict[str, Dict] = {}
        for name, model in [("evcs", self.evcs), ("pms", self.pms)]:
            try:
                self.physics_viols[name] = _pvr(model)
            except Exception:
                self.physics_viols[name] = {"physics_loss": float("nan")}

    # ── Physics projection helper ────────────────────────────────────────

    def _project(self, der_type: str, feat: torch.Tensor, out: torch.Tensor
                 ) -> torch.Tensor:
        """Apply physics_project if projection is enabled for this scenario."""
        if self.use_proj:
            return physics_project(der_type, feat, out)
        return out

    # ── Single timestep ───────────────────────────────────────────────────

    def _step(self, t_sec: float, dss: DSSInterface) -> Dict:
        cfg     = self.cfg
        t_h     = cfg.start_hour + t_sec / 3600.0
        load    = load_profile(t_sec, cfg)
        irr_wm2 = irradiance_profile(t_sec, cfg)
        irr_pu  = irr_wm2 / 1000.0

        v_prev  = (float(np.mean(list(dss._bus_voltages.values())))
                   if dss._bus_voltages else 1.0)

        der_kw:   Dict[str, float] = {}
        der_kvar: Dict[str, float] = {}

        # ── EVCS ────────────────────────────────────────────────────────
        evcs_kw_total = 0.0
        with torch.no_grad():
            for i in range(cfg.n_evcs):
                feat = _evcs_features(i, t_sec, self.evcs_soc[i], v_prev, cfg)
                out  = self._project("evcs", feat, self.evcs(feat))
                p_kw = float(out.squeeze()[2]) * cfg.evcs_rated_kw
                p_kw = float(np.clip(p_kw, 0.0, cfg.evcs_rated_kw))
                der_kw[f"EVCS{i+1}"]   = p_kw
                der_kvar[f"EVCS{i+1}"] = p_kw * 0.08
                evcs_kw_total += p_kw
                # EVCS capacity: 2 × rated_kw kWh per device (e.g. 100 kWh for 50 kW charger)
                self.evcs_soc[i] = float(np.clip(
                    self.evcs_soc[i] + p_kw * (cfg.timestep_s / 3600.0) / (2 * cfg.evcs_rated_kw),
                    0.05, 0.98,
                ))

        # ── PMS/PV ──────────────────────────────────────────────────────
        pms_kw_total = 0.0
        with torch.no_grad():
            for i in range(cfg.n_pms):
                feat = _pms_features(i, t_sec, irr_pu, v_prev, cfg)
                out  = self._project("pms", feat, self.pms(feat))
                p_kw = float(out.squeeze()[2]) * cfg.pms_rated_kw
                # Physical ceiling: PV cannot exceed actual solar availability
                p_kw = float(np.clip(p_kw, 0.0,
                                     irr_pu * cfg.pms_rated_kw + 1e-6))
                # q_ref ∈ [-0.3, 0.3] (volt-VAR droop, normalised to S_rated)
                # → ±9 kVAr per device, ±45 kVAr fleet total (realistic for 30 kW inverter)
                q_kvar = float(out.squeeze()[1]) * cfg.pms_rated_kw
                der_kw[f"PMS{i+1}"]   = p_kw
                der_kvar[f"PMS{i+1}"] = q_kvar
                pms_kw_total += p_kw

        # ── BMS ──────────────────────────────────────────────────────────
        bms_kw_total = 0.0
        # BMS removed: no storage buffer → grid is more sensitive to EVCS/PMS
        # dispatch quality, making naive-GA physics violations visible.

        # ── Power flow ───────────────────────────────────────────────────
        dss._t_sec = t_sec
        dss.run_powerflow(der_kw, der_kvar)

        voltages = dss.get_voltages()
        v_vals   = list(voltages.values()) or [1.0]
        v_mean   = float(np.mean(v_vals))
        v_min    = float(np.min(v_vals))
        v_max    = float(np.max(v_vals))
        v_std    = float(np.std(v_vals))
        viol     = dss.count_violations()
        losses   = dss.get_losses()

        gen_kw    = pms_kw_total
        imbalance = gen_kw - load - evcs_kw_total

        return {
            "t_sec":         t_sec,
            "t_hour":        round(t_h, 4),
            "scenario":      self.scenario,
            "physics_proj":  int(self.use_proj),
            "load_kw":       round(load, 2),
            "irradiance":    round(irr_wm2, 1),
            "evcs_kw":       round(evcs_kw_total, 2),
            "pms_kw":        round(pms_kw_total, 2),
            "gen_kw":        round(gen_kw, 2),
            "imbalance_kw":  round(imbalance, 2),
            "v_mean_pu":     round(v_mean, 5),
            "v_min_pu":      round(v_min, 5),
            "v_max_pu":      round(v_max, 5),
            "v_std_pu":      round(v_std, 5),
            "violations":    int(viol),
            "losses_kw":     round(losses, 3),
            "evcs_soc_avg":  round(float(np.mean(self.evcs_soc)), 4),
        }

    # ── Simulation loop ───────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> List[Dict]:
        dss = DSSInterface(self.cfg)
        t0  = time.time()
        proj_label = "(physics projection ON)" if self.use_proj else "(no projection)"
        if verbose:
            print(f"\n  [{self.scenario}] {self.cfg.hours:.1f}h "
                  f"({self.cfg.n_steps} steps) {proj_label}")

        for step in range(self.cfg.n_steps):
            rec = self._step(step * self.cfg.timestep_s, dss)
            self.records.append(rec)
            if verbose and step % 360 == 0:
                print(f"  t={rec['t_hour']:.2f}h  "
                      f"V={rec['v_mean_pu']:.4f}pu  "
                      f"viol={rec['violations']:3d}  "
                      f"EVCS={rec['evcs_kw']:.1f}kW  "
                      f"PMS={rec['pms_kw']:.1f}kW")

        if verbose:
            s = self.summary_stats()
            print(f"  ✓ Done {time.time()-t0:.1f}s  "
                  f"total_viol={s['total_violations']}  "
                  f"V_mean={s['mean_v_pu']:.4f}  "
                  f"conv={dss.convergence_rate:.1%}")
        return self.records

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, result_dir: Path, verbose: bool = True):
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        if not self.records:
            return
        path = result_dir / f"sim_{self.scenario}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        if verbose:
            print(f"  ✓ Saved → {path}")

    def summary_stats(self) -> Dict:
        if not self.records:
            return {}
        viol   = [r["violations"]    for r in self.records]
        vmean  = [r["v_mean_pu"]     for r in self.records]
        vmin   = [r["v_min_pu"]      for r in self.records]
        losses = [r["losses_kw"]     for r in self.records]
        imbal  = [abs(r["imbalance_kw"]) for r in self.records]
        pms_kw = [r["pms_kw"]        for r in self.records]
        evcs_kw= [r["evcs_kw"]       for r in self.records]
        # Physics violation rates — model-level metric independent of grid simulation.
        # High values for naive GA reveal residual physical inconsistencies even when
        # grid voltage violations appear low (projection=OFF but model still "safe by luck").
        def _pl(der):
            return round(self.physics_viols.get(der, {}).get("physics_loss", float("nan")), 5)

        def _avg_viol(der):
            d = self.physics_viols.get(der, {})
            vals = [v for k, v in d.items() if k.endswith("_viol")]
            return round(float(np.mean(vals)) if vals else float("nan"), 5)

        return {
            "scenario":                self.scenario,
            "physics_projection":      self.use_proj,
            "hours":                   self.cfg.hours,
            "total_violations":        int(sum(viol)),
            "pct_steps_w_violation":   round(float(np.mean([v > 0 for v in viol])) * 100, 2),
            "mean_v_pu":               round(float(np.mean(vmean)), 5),
            "min_v_pu":                round(float(np.min(vmin)),   5),
            "max_v_pu":                round(float(np.max(vmean)),  5),
            "mean_losses_kw":          round(float(np.mean(losses)), 3),
            "total_losses_kwh":        round(float(np.sum(losses)) * self.cfg.timestep_s / 3600, 3),
            "mean_abs_imbalance_kw":   round(float(np.mean(imbal)), 3),
            "mean_pms_output_kw":      round(float(np.mean(pms_kw)), 3),
            "mean_evcs_load_kw":       round(float(np.mean(evcs_kw)), 3),
            # Physics model compliance (Option B) ─────────────────────────────
            "phys_loss_evcs":          _pl("evcs"),
            "phys_loss_pms":           _pl("pms"),
            "phys_avg_viol_evcs":      _avg_viol("evcs"),
            "phys_avg_viol_pms":       _avg_viol("pms"),
            "physics_viols":           {k: {ck: round(cv, 5) for ck, cv in v.items()}
                                        for k, v in self.physics_viols.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════
# Scenario loader
# ═══════════════════════════════════════════════════════════════════════════

def load_scenario_models(
    scenario:  str,
    model_dir: Path,
) -> "Tuple[CPSPINN | SISAEnsemble, CPSPINN | SISAEnsemble]":
    model_dir = Path(model_dir)
    models    = {}
    for der in DER_TYPES:
        path = model_dir / der / f"model_{der}_{scenario}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        models[der] = load_any_model(path)
    return models["evcs"], models["pms"]


def run_scenario(
    scenario:       str,
    model_dir:      Path,
    result_dir:     Path,
    use_projection: bool,
    cfg:            SimConfig | None = None,
    verbose:        bool = True,
) -> Dict:
    """Load models and run simulation for a single scenario."""
    base_cfg = cfg or SimConfig()
    sc_cfg   = SimConfig(
        hours                  = base_cfg.hours,
        timestep_s             = base_cfg.timestep_s,
        start_hour             = base_cfg.start_hour,
        base_load_kw           = base_cfg.base_load_kw,
        dss_file               = base_cfg.dss_file,
        voltage_lo             = base_cfg.voltage_lo,
        voltage_hi             = base_cfg.voltage_hi,
        use_physics_projection = use_projection,
        force_analytical       = base_cfg.force_analytical,
    )
    evcs, pms = load_scenario_models(scenario, model_dir)
    sim = CPSGridSimulator(scenario, evcs, pms, sc_cfg)
    sim.run(verbose=verbose)
    sim.save(result_dir, verbose=verbose)
    return sim.summary_stats()
