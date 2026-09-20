# %%
"""Glenelg S10-gt1 eclogite garnet — 1D radial multicomponent diffusion inversion.

Inverts the post-peak retrograde cooling / decompression schedule for Glenelg
eclogite garnet S10-gt1 against the measured EPMA compositional profiles
(Alm -> Fe, Py -> Mg, Sp -> Mn, Gro -> Ca), using the 1D forward model in
``glenelg_s10_1d.py``.

Design
------
- The garnet growth history (MAGEMin/phasetools with the Glenelg S10 bulk rock,
  or a synthetic schedule for testing) is computed ONCE and cached; each
  objective evaluation runs a FRESH 1D forward diffusion model under the
  candidate schedule. The cache is validated to be prograde-only: growth steps
  (de-duplicated on equal times, as the forward solver does) increasing within
  model time 0-60 Myr (1060 -> 1000 Ma), since the inversion freezes the
  growth zoning and re-times only the retrograde path.
- Fixed PTt vertices: P = [6, 20, 12.2, 14, 10.6] kbar, T = [500, 700, 725,
  600, 510] degC; fixed early ages t0 = 1060 Ma (model time 0) and
  tstart = 1000 Ma (model time 60 Myr).
- The three younger ages are the free inversion parameters, parameterised by
  strictly positive elapsed intervals (dt1, dt2, dt3) so the strict ordering
  1000 > t1 > t2 > t3 holds for any admissible candidate:
      t1 = 1000 - dt1,   t2 = t1 - dt2,   t3 = t2 - dt3
  The nominal 90/100/100 Myr (ages 910/810/710 Ma) is the initial guess only:
  710 Ma is the nominal final-age target for reference, and t3 is optimised
  exactly like t1 and t2.
- The EPMA diameter traverse (X in mm) is centred at its midpoint (legacy 2D
  convention r - r.max()/2) and folded onto a core-to-rim radius with abs().
- Misfit: weighted RMSE over the independent components Fe/Mg/Mn (default).
  Ca = 1 - Fe - Mg - Mn is the solver's closure component: its residual is
  reported but excluded from the objective to avoid double-counting.
- Optimisation: scipy Nelder-Mead with bounds on the interval parameters.

All run settings live in the parameter block below — there is no CLI.
"""


import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

"""Reusable forward pieces for the Glenelg inversion.

This module is deliberately separate from ``glenelg_s10_1d.py``: importing a
Jupytext tutorial executes its cells, whereas the inversion needs many fresh
solver evaluations without rerunning the tutorial's MAGEMin/output cells.
"""

import numpy as np
from scipy import interpolate
from pathlib import Path

from UWDiffusion.fd_1d import MulticomponentDiffusion1D, darken_reduced_diffusivity

# ==============================================================================
# Fixed PTt vertices and ages (Glenelg S10 PT loop — not inverted)
# ==============================================================================


age_t0 = 1060.0  # Ma: subduction inception (model time 0 Myr)
age_tstart = 1000.0  # Ma: peak pressure (model time 60 Myr)
growth_window = (0.0, age_t0 - age_tstart)  # model-time window growth must lie in
target_t3 = 890.0  # Ma: nominal final age, reported for reference (not a constraint)

# ==============================================================================
# Run settings — edit these, there is no CLI
# ==============================================================================

# Forward model (glenelg_s10_1d.py)
garnet_no = 0  # garnet size class (0 = gt1, 40 = gt2 in the source scripts)
rmin, rmax = 50.0, 380.0  # garnet radius range for GarnetGenerator (um)
quick = True  # use every 2nd growth-shell radius as a grid vertex
dt_myr = 2.0  # maximum diffusion time step (Myr)
fixed_bc = True  # pin the rim to the retrograde rim composition each segment
geometry = "cylindrical"  # 1D equivalent of the 2D Cartesian annulus
pt_npoints_source = 501
pt_npoints = pt_npoints_source  # PTt path resolution for MAGEMin
growth_start, growth_end = 0.0, 100.0  # growth window labels (Myr, source convention)

# Inversion
synthetic = False  # synthetic growth history (testing only; no MAGEMin/Julia)
max_nfev = 200  # maximum forward-model evaluations for the optimiser
initial_ages = np.array([910.0, 900.0, 890.0])  # Ma: nominal guess for (t1, t2, t3)
interval_bounds = [(10.0, 250.0), (10.0, 250.0), (10.0, 250.0)]  # Myr, on (dt1, dt2, dt3)
fit_components = ["Fe", "Mg", "Mn"]  # Ca is the closure: reported, not fitted
component_names = ["Fe", "Mg", "Mn", "Ca"]
# Backwards-compatible public name for callers of earlier example revisions.
all_components = component_names
output_dir = Path("./output/glenelg_s10_1d_inversion")
make_plots = True


sec_per_myr = 1.0e6 * 365.25 * 24.0 * 3600.0
components = component_names  # public component order used by the forward solver
dependent = "Ca"
dependent_idx = 3

p_arr = np.array([6.0, 20.0, 10.4, 12.8, 10.4])
t_arr = np.array([500.0, 700.0, 735.0, 620.0, 530.0])
t_ma_arr = np.array([1060.0, 1000.0, 910.0, 900.0, 890.0])
p_vertices = p_arr.copy()  # kbar; shared with the forward path
t_vertices = t_arr.copy()  # degC; shared with the forward path



gt_classes = 100
gt_nr = 50

db = "mpe"
dataset = 62


xoxides = ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO", "CaO", "Na2O", "K2O", "H2O"]
x_s10_wt = [51.6, 1.41, 14.1, 1.62, 13.1, 0.25, 5.36, 9.89, 2.19, 0.36, 0.08]
d0_cm2s = {"Fe": 6.36e-4, "Mg": 1.11e-3, "Mn": 5.15e-4}
ea_cal = {"Fe": 65824.0, "Mg": 67997.0, "Mn": 60569.0}
ev_cm3 = {"Fe": 5.63, "Mg": 5.27, "Mn": 6.04}
epma_csv_paths = {
    0: Path("/Users/benknight/Documents/Research/Modelling/UW-models/Diffusion_modelling/Fe-Mg_exchange/gt-cpx_modelling/glenelg-s10/data/S10-gt1.csv"),
    40: Path("/Users/benknight/Documents/Research/Modelling/UW-models/Diffusion_modelling/Fe-Mg_exchange/gt-cpx_modelling/glenelg-s10/data/S10-gt2.csv"),
}

epma_csv = epma_csv_paths[0]  # S10-gt1 transect

def tracer_diffusivities_m2s(temp_K, pressure_bar):
    values = []
    for elem in ("Fe", "Mg", "Mn"):
        exponent = -(ea_cal[elem] + (pressure_bar - 1.0) * ev_cm3[elem] / 41.84) / (1.9872 * temp_K)
        values.append(d0_cm2s[elem] * np.exp(exponent) * 1.0e-4)
    values.append(0.05 * values[-1])
    return np.asarray(values)


def make_diffusivity_callable(T_of_t, P_of_t):
    def diffusivity(t, X):
        t_myr = float(t) / sec_per_myr
        values = np.asarray([float(X[name]) for name in components] if isinstance(X, dict) else X, dtype=float)
        return darken_reduced_diffusivity(
            tracer_diffusivities_m2s(float(T_of_t(t_myr)), float(P_of_t(t_myr))),
            np.clip(values, 0.0, 1.0), dependent_component=dependent_idx,
        )
    return diffusivity


def generate_growth_history_synthetic(*, rmin, rmax):
    n = 12
    tg = np.linspace(10.0, 50.0, n)
    rr = np.linspace(rmin, rmax, n)
    fe = np.concatenate([[0.72], np.linspace(0.70, 0.66, n)])
    mg = np.concatenate([[0.11], np.linspace(0.12, 0.15, n)])
    mn = np.concatenate([[0.09], np.linspace(0.08, 0.03, n)])
    ca = 1.0 - fe - mg - mn
    nodes = np.array([0.0, 50.0, 90.0])
    temp = np.array([500.0, 725.0, 510.0])
    pressure = np.array([8.0, 16.0, 9.0])
    path = np.column_stack([pressure, temp, nodes])
    T_of_t = interpolate.interp1d(nodes, temp + 273.15, fill_value="extrapolate")
    P_of_t = interpolate.interp1d(nodes, pressure * 1e3, fill_value="extrapolate")
    tr = np.linspace(tg[-1], 90.0, 7)
    wiggle = np.linspace(0.0, 0.01, tr.size)
    return {
        "Rr": np.r_[0.0, rr], "tr": np.r_[tg[0], tg],
        "Pr": np.interp(np.r_[tg[0], tg], nodes, pressure), "Tr": np.interp(np.r_[tg[0], tg], nodes, temp),
        "Mnr": mn, "Mgr": mg, "Fer": fe, "Car": ca,
        "prograde": {"t": tg, "T": np.interp(tg, nodes, temp), "P": np.interp(tg, nodes, pressure), "Mn": mn[1:], "Mg": mg[1:], "Fe": fe[1:], "Ca": ca[1:]},
        "retrograde": {"t": tr, "T": np.interp(tr, nodes, temp), "P": np.interp(tr, nodes, pressure), "Mn": mn[-1] + wiggle, "Mg": mg[-1] - wiggle, "Fe": fe[-1] - wiggle, "Ca": 1.0 - (fe[-1] - wiggle) - (mg[-1] - wiggle) - (mn[-1] + wiggle)},
        "PTt_path": path, "T_of_t": T_of_t, "P_of_t": P_of_t,
        "source": "synthetic", "bulk_rock": "synthetic schedule (no MAGEMin)",
    }


def generate_growth_history_magemin(*, garnet_no, rmin, rmax, pt_npoints):
    import phasetools
    from juliacall import Main as jl, convert as jlconvert
    from phasetools import GarnetGenerator

    t_myr = t_ma_arr.max() - t_ma_arr
    path = phasetools.utils.create_PTt_path(p_arr, t_arr, t_myr, pt_npoints)
    P_data, T_data, t_data = path[:, 0], path[:, 1], path[:, 2]
    x_mol, oxide_names = phasetools.MAGEMin_C.convertBulk4MAGEMin(
        jlconvert(jl.Vector[jl.Float64], x_s10_wt), jlconvert(jl.Vector[jl.String], xoxides), "wt", db,
    )
    oxide_names, x_mol = list(oxide_names), np.asarray(x_mol) / 100.0
    generator = GarnetGenerator(db=db, dataset=dataset)
    generator.setup_bulk_composition(Xoxides=oxide_names, X=x_mol, sys_in="mol", rm_list=None)
    generator.generate_garnet_data(P_data, T_data, t_data, r_min=rmin, r_max=rmax, garnet_classes=gt_classes, nR_diff=gt_nr, fractionate=False, normalise_start=True)
    data = generator.generate_garnets("N")[garnet_no]
    Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car = (np.asarray(data[k], dtype=float) for k in ("Rr", "tr", "Pr", "Tr", "Mnr", "Mgr", "Fer", "Car"))
    tp, Tp, Pp, Mnp, Mgp, Fep, Cap = generator.get_prograde_concentrations()
    trr, Trt, Prt, Mnr_t, Mgr_t, Fer_t, Car_t = generator.get_retrograde_concentrations()
    if len(trr) < 10:
        if generator.X_last_growth is None:
            raise RuntimeError("No garnet growth was detected along the path.")
        calc = phasetools.MAGEMinGarnetCalculator(db=db, dataset=dataset)
        calc.setup_bulk_composition(oxide_names, generator.X_last_growth, "mol", None)
        Prt, Trt, trr = P_data[t_data >= 100.0], T_data[t_data >= 100.0], t_data[t_data >= 100.0]
        values = [calc.gt_single_point_calc_elements(p, t)[3:7] for p, t in zip(Prt, Trt)]
        Mnr_t, Mgr_t, Fer_t, Car_t = np.asarray(values).T
    T_of_t = interpolate.interp1d(t_data, T_data + 273.15, fill_value="extrapolate")
    P_of_t = interpolate.interp1d(t_data, P_data * 1e3, fill_value="extrapolate")
    return {"Rr": Rr, "tr": tr, "Pr": Pr, "Tr": Tr, "Mnr": Mnr, "Mgr": Mgr, "Fer": Fer, "Car": Car, "prograde": {"t": np.asarray(tp), "T": np.asarray(Tp), "P": np.asarray(Pp), "Mn": np.asarray(Mnp), "Mg": np.asarray(Mgp), "Fe": np.asarray(Fep), "Ca": np.asarray(Cap)}, "retrograde": {"t": np.asarray(trr), "T": np.asarray(Trt), "P": np.asarray(Prt), "Mn": np.asarray(Mnr_t), "Mg": np.asarray(Mgr_t), "Fe": np.asarray(Fer_t), "Ca": np.asarray(Car_t)}, "PTt_path": path, "T_of_t": T_of_t, "P_of_t": P_of_t, "source": "magemin", "bulk_rock": "Glenelg S10"}


def run_forward_evaluation(history, *, fixed_bc, quick, dt_myr, geometry):
    growth = history["tr"] > history["tr"].min()
    Rr, tr = history["Rr"][growth], history["tr"][growth]
    arrays = [history[key][growth] for key in ("Fer", "Mgr", "Mnr", "Car")]
    _, idx = np.unique(tr, return_index=True)
    idx = np.sort(idx)
    Rr, tr = Rr[idx], tr[idx]
    Fe, Mg, Mn, Ca = [a[idx] for a in arrays]
    grid_um = np.unique(history["Rr"])[::2 if quick else 1]
    if grid_um[-1] < history["Rr"].max():
        grid_um = np.r_[grid_um, history["Rr"].max()]
    model = MulticomponentDiffusion1D(grid_um * 1e-6, make_diffusivity_callable(history["T_of_t"], history["P_of_t"]), geometry=geometry, components=components, dependent_component=dependent, initial_composition={"Fe": history["Fer"][0], "Mg": history["Mgr"][0], "Mn": history["Mnr"][0]})
    dt = dt_myr * sec_per_myr
    model.reset(tr[0] * sec_per_myr)
    for i in range(len(tr) - 1):
        comp = {"Fe": Fe[i], "Mg": Mg[i], "Mn": Mn[i]}
        model.apply_growth(Rr[i] * 1e-6, comp, set_rim_bc=False)
        mask = model.r >= Rr[i] * 1e-6
        mask[-1] = False
        model.set_internal_bcs(mask, comp)
        model.set_bc("right", "dirichlet", comp)
        model.run_to_time(tr[i + 1] * sec_per_myr, dt)
    model.clear_internal_bcs()
    comp = {"Fe": Fe[-1], "Mg": Mg[-1], "Mn": Mn[-1]}
    model.apply_growth(Rr[-1] * 1e-6, comp, set_rim_bc=False)
    model.reset(model.time)
    profile_prograde = model.get_composition().copy()
    model.clear_internal_bcs()
    retro = history["retrograde"]
    start = int(np.searchsorted(retro["t"], tr[-1]))
    for i in range(start, len(retro["t"]) - 1):
        if fixed_bc:
            model.set_bc("rim", "dirichlet", {"Fe": retro["Fe"][i], "Mg": retro["Mg"][i], "Mn": retro["Mn"][i]})
        else:
            model.set_bc("rim", "neumann", 0.0)
        if retro["t"][i + 1] > model.time / sec_per_myr:
            model.run_to_time(retro["t"][i + 1] * sec_per_myr, dt)
    return model, {"initial_profile": np.column_stack([history["Rr"], history["Fer"], history["Mgr"], history["Mnr"], history["Car"]]), "profile_prograde": profile_prograde, "profile_final": model.get_composition().copy()}



def make_options(**overrides) -> SimpleNamespace:
    """Assemble the run-options namespace from the settings above.

    ``glenelg_s10_1d.run_1d_model`` and the growth generators read their run
    settings from a namespace; this bridges the parameter block to them.
    Keyword overrides replace any setting (used by tests and quick experiments).
    """
    options = SimpleNamespace(
        garnet_no=garnet_no,
        rmin=rmin,
        rmax=rmax,
        quick=quick,
        dt_myr=dt_myr,
        fixed_bc=fixed_bc,
        geometry=geometry,
        pt_npoints=pt_npoints,
        growth_start=growth_start,
        growth_end=growth_end,
        synthetic=synthetic,
        max_nfev=max_nfev,
        fit_components=list(fit_components),
        output_dir=Path(output_dir),
        make_plots=make_plots,
    )
    for name, value in overrides.items():
        setattr(options, name, value)
    options.output_dir = Path(options.output_dir)
    options.source_label = "synthetic" if options.synthetic else "magemin"
    return options


# ==============================================================================
# Parameter mapping: three positive intervals <-> three strictly ordered ages
# ==============================================================================


def intervals_to_ages(intervals) -> np.ndarray:
    """Map positive elapsed intervals [dt1, dt2, dt3] (Myr) to ages [t1, t2, t3] (Ma).

    t1 = 1000 - dt1, t2 = t1 - dt2, t3 = t2 - dt3, so any positive intervals
    guarantee the strict ordering 1000 > t1 > t2 > t3.
    """
    dt = np.asarray(intervals, dtype=float)
    if dt.shape != (3,):
        raise ValueError(f"intervals must have shape (3,), got {dt.shape}")
    t1 = age_tstart - dt[0]
    t2 = t1 - dt[1]
    t3 = t2 - dt[2]
    return np.array([t1, t2, t3])


def ages_to_intervals(ages) -> np.ndarray:
    """Map absolute ages [t1, t2, t3] (Ma) to elapsed intervals [dt1, dt2, dt3] (Myr)."""
    ages = np.asarray(ages, dtype=float)
    if ages.shape != (3,):
        raise ValueError(f"ages must have shape (3,), got {ages.shape}")
    return np.array([age_tstart - ages[0], ages[0] - ages[1], ages[1] - ages[2]])


def ages_to_model_time_nodes(ages) -> np.ndarray:
    """Convert ages [t1, t2, t3] (Ma) to model time nodes [0, 60, ...] (Myr from 1060 Ma)."""
    ages = np.asarray(ages, dtype=float)
    full_ages = np.array([age_t0, age_tstart, ages[0], ages[1], ages[2]])
    return full_ages[0] - full_ages


# ==============================================================================
# Cached growth history and candidate re-timing
# ==============================================================================


def check_prograde_history(history: dict) -> None:
    """Fail clearly if the cached history grows outside the prograde window.

    The inversion freezes the growth zoning and re-times only the retrograde
    path, so growth steps must lie within model time 0-60 Myr (1060 -> 1000 Ma)
    and be nondecreasing once strictly-equal shell times are de-duplicated —
    exactly the de-duplication ``run_1d_model`` applies before solving (several
    shells can grow at the same path time). A history still growing past the
    fixed 1000 Ma vertex would be silently frozen by the re-timing, so it is
    rejected here instead.
    """
    tr = np.asarray(history["tr"], dtype=float)
    growth = tr[tr > tr.min()]
    if growth.size == 0:
        raise RuntimeError("cached history has no growth steps (tr > tr.min() is empty)")
    # Mirror run_1d_model: keep the first occurrence of each growth time,
    # original order, then require strict monotonicity of what is left.
    _, first_idx = np.unique(growth, return_index=True)
    growth_steps = growth[np.sort(first_idx)]
    if np.any(np.diff(growth_steps) <= 0.0):
        raise RuntimeError("cached history growth times are not strictly increasing")
    if growth_steps.min() < growth_window[0] - 1e-9:
        raise RuntimeError(
            f"cached history starts before model time {growth_window[0]:.1f} Myr"
        )
    if growth_steps.max() > growth_window[1] + 1e-9:
        raise RuntimeError(
            f"cached history still growing at {growth_steps.max():.2f} Myr, beyond the fixed "
            f"1000 Ma vertex ({growth_window[1]:.1f} Myr): growth cannot be frozen "
            "and re-timed"
        )


def cache_growth_history(options: SimpleNamespace) -> dict:
    """Generate and cache the prograde garnet growth history once.

    Garnet nucleation and growth shells are computed with MAGEMin/phasetools
    (S10 bulk rock) or with the synthetic schedule, then validated to be
    prograde-only. Arc coordinates s in [0, 4] are attached to the retrograde
    rim schedule so candidate timings re-map monotonically without re-running
    phase equilibria.
    """
    if options.synthetic:
        history = generate_growth_history_synthetic(rmin=options.rmin, rmax=options.rmax)
        print("[setup] cached synthetic growth history (no MAGEMin)", flush=True)
    else:
        history = generate_growth_history_magemin(garnet_no=options.garnet_no,
                                                   rmin=options.rmin, rmax=options.rmax,
                                                   pt_npoints=options.pt_npoints)
        print("[setup] cached MAGEMin growth history", flush=True)
    check_prograde_history(history)

    # Arc position of the final growth step on the nominal path; s = 4 is the
    # path end (the candidate t3 node).
    t_nodes_nominal = ages_to_model_time_nodes([910.0, 810.0, 710.0])
    s_growth_final = float(np.interp(float(history["tr"][-1]), t_nodes_nominal, np.arange(5)))
    history["retrograde_s"] = np.linspace(s_growth_final, 4.0, len(history["retrograde"]["t"]))
    return history


def build_ptt_path(t_nodes, pt_npoints: int = pt_npoints_source):
    """PCHIP PTt path through the fixed P/T vertices at the given time nodes.

    Returns ``(PTt_path, T_of_t, P_of_t)``: the path has columns (P kbar,
    T degC, t Myr); the interpolants return K and bar for the diffusivity.
    """
    n = len(p_vertices)
    p_spline = PchipInterpolator(np.arange(n), p_vertices)
    t_spline = PchipInterpolator(np.arange(n), t_vertices)
    time_spline = PchipInterpolator(np.arange(n), t_nodes)

    s = np.unique(np.concatenate([np.arange(n), np.linspace(0, n - 1, pt_npoints)]))
    path = np.column_stack([p_spline(s), t_spline(s), time_spline(s)])
    T_of_t = interpolate.interp1d(path[:, 2], path[:, 1] + 273.15, fill_value="extrapolate")
    P_of_t = interpolate.interp1d(path[:, 2], path[:, 0] * 1e3, fill_value="extrapolate")
    return path, T_of_t, P_of_t


def check_candidate_intervals(intervals, cached_history: dict | None = None) -> np.ndarray:
    """Validate a candidate [dt1, dt2, dt3] (Myr) and return its ages [t1, t2, t3].

    Intervals must be finite and strictly positive (which guarantees the strict
    age ordering), the final model-time node must lie after the fixed 1000 Ma
    vertex, and — when a cached history is given — after the final growth step
    so the retrograde overprint has a window to run in.
    """
    dt = np.asarray(intervals, dtype=float)
    if dt.shape != (3,) or not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError(f"candidate intervals must be 3 finite positive values, got {intervals}")
    nodes = ages_to_model_time_nodes(intervals_to_ages(dt))
    if nodes[-1] <= growth_window[1]:
        raise ValueError(
            f"candidate final model time {nodes[-1]:.2f} Myr is not after the fixed "
            f"1000 Ma vertex ({growth_window[1]:.1f} Myr)"
        )
    if cached_history is not None:
        t_growth_final = float(np.asarray(cached_history["tr"], dtype=float).max())
        if nodes[-1] <= t_growth_final:
            raise ValueError(
                f"candidate ends at {nodes[-1]:.2f} Myr, at/before the final growth "
                f"step ({t_growth_final:.2f} Myr): no window for the retrograde overprint"
            )
    return intervals_to_ages(dt)


def update_history_timing(cached_history: dict, intervals, pt_npoints=pt_npoints_source) -> dict:
    """Re-time the cached history for candidate intervals [dt1, dt2, dt3] (Myr).

    Growth zoning (radii, times, compositions) is untouched; only the PTt path,
    the T(K)/P(bar) interpolants and the retrograde rim schedule change. The
    mapped retrograde schedule ends EXACTLY at the candidate t3 model-time node
    (PCHIP passes through its nodes; s = 4 is pinned exactly) and must be
    strictly increasing — nonmonotonic mappings are rejected, not fudged with
    +epsilon adjustments. Points mapped before the final growth step are
    counted in ``retrograde_trimmed``: the forward model's np.searchsorted
    selection skips them, as in the 2D source model.
    """
    ages = check_candidate_intervals(intervals, cached_history)
    t_nodes = ages_to_model_time_nodes(ages)
    t_growth_final = float(np.asarray(cached_history["tr"], dtype=float).max())

    history = dict(cached_history)
    history["retrograde"] = dict(cached_history["retrograde"])
    history["PTt_path"], history["T_of_t"], history["P_of_t"] = build_ptt_path(t_nodes, pt_npoints)

    s_retro = np.asarray(cached_history["retrograde_s"], dtype=float)
    if s_retro.size != len(history["retrograde"]["t"]):
        raise ValueError("cached retrograde arc coordinates do not match the retrograde schedule")
    time_spline = PchipInterpolator(np.arange(len(t_nodes)), t_nodes)
    new_t = np.asarray(time_spline(s_retro), dtype=float)
    new_t[-1] = t_nodes[-1]  # end exactly at the candidate t3 node
    if not np.all(np.diff(new_t) > 0.0):
        raise ValueError(
            f"candidate retrograde schedule is not strictly increasing for intervals "
            f"{np.asarray(intervals, dtype=float).tolist()} Myr (ages {ages.tolist()} Ma)"
        )
    history["retrograde_trimmed"] = int(np.sum(new_t < t_growth_final))
    history["retrograde"]["t"] = new_t
    return history


# ==============================================================================
# EPMA observations and misfit
# ==============================================================================


def map_epma_folded_radius(r_mm) -> np.ndarray:
    """Fold a diameter traverse (mm) onto a core-to-rim radius (um).

    r_um = X*1e3, centred at the transect midpoint (legacy 2D convention
    r - r.max()/2), then folded onto the radius with abs(): symmetric points
    from both halves map onto the same radius. Assumes a centre-crossing
    traverse of an approximately radially symmetric grain.
    """
    r_um = np.asarray(r_mm, dtype=float) * 1e3
    return np.abs(r_um - r_um.max() / 2.0)


def load_epma_data(path=epma_csv, rmax_um: float = rmax) -> dict:
    """Load the EPMA transect and map it to folded radii and end-members.

    Alm -> Fe, Py -> Mg, Sp -> Mn, Gro -> Ca. Points folding beyond rmax are
    outside the model domain and are rejected rather than silently clipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EPMA data file not found: {path}")
    df = pd.read_csv(path)

    r = map_epma_folded_radius(df["X"].to_numpy(dtype=float))
    data = {
        "Fe": df["Alm"].to_numpy(dtype=float),
        "Mg": df["Py"].to_numpy(dtype=float),
        "Mn": df["Sp"].to_numpy(dtype=float),
        "Ca": df["Gro"].to_numpy(dtype=float),
    }
    order = np.argsort(r)
    r = r[order]
    data = {comp: values[order] for comp, values in data.items()}

    n_out = int(np.sum(r > rmax_um))
    if n_out:
        raise ValueError(
            f"{n_out} of {r.size} EPMA points fold beyond rmax={rmax_um:g} um "
            f"(max folded radius {r.max():.1f} um): outside the model domain"
        )
    return {"r": r, **data, "source_path": str(path)}


def load_observations(cached_history: dict, options: SimpleNamespace) -> dict:
    """EPMA observations if the S10-gt1 CSV exists, else synthetic ones."""
    if epma_csv.exists():
        print(f"[data] loading EPMA observations from {epma_csv}", flush=True)
        return load_epma_data(epma_csv, rmax_um=options.rmax)
    if options.synthetic:
        print(f"[data] {epma_csv} not found; generating synthetic EPMA observations", flush=True)
        return make_synthetic_observations(cached_history, options)
    raise FileNotFoundError(
        f"EPMA data file not found: {epma_csv}. Run with synthetic=True when the "
        "transect CSV is unavailable."
    )


def make_synthetic_observations(cached_history: dict, options: SimpleNamespace) -> dict:
    """Sample 40 core-to-rim points from the nominal-schedule forward model."""
    nominal = update_history_timing(cached_history, ages_to_intervals(initial_ages), options.pt_npoints)
    model, diag = run_forward_evaluation(nominal, fixed_bc=options.fixed_bc,
                                          quick=options.quick, dt_myr=options.dt_myr,
                                          geometry=options.geometry)
    r_model_um = model.r * 1e6
    r_syn = np.linspace(0.0, r_model_um[-1], 40)
    interp = evaluate_model_at_radii(r_model_um, diag["profile_final"], r_syn)
    return {"r": r_syn, **interp, "source_path": "synthetic"}


def evaluate_model_at_radii(model_r_um, model_profile, r_obs) -> dict:
    """Interpolate the simulated profiles onto the observation radii.

    Observation radii outside the model grid are rejected rather than silently
    clipped to the core/rim.
    """
    r_obs = np.asarray(r_obs, dtype=float)
    n_out = int(np.sum((r_obs < model_r_um[0] - 1e-6) | (r_obs > model_r_um[-1] + 1e-6)))
    if n_out:
        raise ValueError(
            f"{n_out} of {r_obs.size} observation radii fall outside the model grid "
            f"[{model_r_um[0]:.3f}, {model_r_um[-1]:.3f}] um (observed range "
            f"{r_obs.min():.3f}-{r_obs.max():.3f} um)"
        )
    return {
        comp: np.interp(r_obs, model_r_um, model_profile[:, idx])
        for idx, comp in enumerate(component_names)
    }


def compute_misfit(model_r_um, model_profile, epma_obs: dict, components=None):
    """Weighted RMSE between the simulated profiles and the EPMA observations.

    Defaults to the independent fit components Fe/Mg/Mn: Ca is the solver's
    closure component (Ca = 1 - Fe - Mg - Mn), so fitting it would double-count
    the independent residuals. Per-component RMSE and residuals are always
    reported for all four components. Returns (total_rmse, component_rmse, residuals).
    """
    if components is None:
        components = fit_components
    if not set(components) <= set(component_names) or len(set(components)) != len(components):
        raise ValueError(f"fit components must be a unique subset of {component_names}, got {components}")

    interp = evaluate_model_at_radii(model_r_um, model_profile, epma_obs["r"])
    residuals = {comp: interp[comp] - epma_obs[comp] for comp in component_names}
    component_rmse = {
        comp: float(np.sqrt(np.mean(residuals[comp] ** 2))) for comp in component_names
    }
    total_rmse = float(
        np.sqrt(sum(component_rmse[comp] ** 2 for comp in components) / len(components))
    )
    return total_rmse, component_rmse, residuals


# ==============================================================================
# Objective: one fresh forward run per candidate schedule
# ==============================================================================


def objective(intervals, cached_history: dict, epma_obs: dict, options: SimpleNamespace, state=None) -> float:
    """Inversion objective at candidate intervals [dt1, dt2, dt3] (Myr).

    Each evaluation re-times the cached history and runs a fresh forward
    diffusion model. Infeasible candidates get a barrier penalty instead of a
    forward run. When ``state`` is given, the evaluation count and the
    best-so-far result are tracked there.
    """
    dt = np.asarray(intervals, dtype=float)
    if dt.shape != (3,) or not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        return 1e5 + 1e4 * float(np.sum(np.abs(np.minimum(np.nan_to_num(dt, nan=-1.0), 0.0))))

    history = update_history_timing(cached_history, dt, options.pt_npoints)
    model, diag = run_forward_evaluation(history, fixed_bc=options.fixed_bc,
                                          quick=options.quick, dt_myr=options.dt_myr,
                                          geometry=options.geometry)
    rmse, comp_rmse, _ = compute_misfit(
        model.r * 1e6, diag["profile_final"], epma_obs, components=options.fit_components
    )

    if state is not None:
        state["count"] += 1
        ages = intervals_to_ages(dt)
        print(
            f"[eval {state['count']:3d}] dt={dt.round(2).tolist()} Myr -> "
            f"ages={ages.round(1).tolist()} Ma | RMSE={rmse:.5f} "
            f"(Fe={comp_rmse['Fe']:.4f}, Mg={comp_rmse['Mg']:.4f}, "
            f"Mn={comp_rmse['Mn']:.4f}, Ca={comp_rmse['Ca']:.4f})",
            flush=True,
        )
        if rmse < state["best_rmse"]:
            state.update(best_rmse=rmse, best_intervals=dt.copy(), best_ages=ages, best_diag=diag)
    return rmse


# ==============================================================================
# Outputs
# ==============================================================================


def next_output_dir(base_dir) -> Path:
    """Return base_dir, or base_dir<N> for the next free suffix (never overwrite)."""
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return base_dir
    max_suffix = 0
    pattern = re.compile(rf"^{re.escape(base_dir.name)}(\d+)$")
    for name in base_dir.parent.iterdir():
        match = pattern.match(name.name)
        if name.is_dir() and match:
            max_suffix = max(max_suffix, int(match.group(1)))
    return base_dir.parent / f"{base_dir.name}{max_suffix + 1}"


def save_results(out_dir: Path, summary: dict, best_model, best_diag: dict, epma_obs: dict, history: dict) -> None:
    """Save the summary JSON, best-fit profile CSV, EPMA comparison CSV and NPZ archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inversion_summary.json").write_text(json.dumps(summary, indent=2))

    r_um = best_model.r * 1e6
    prof = best_diag["profile_final"]
    np.savetxt(
        out_dir / "best_fit_profiles.csv",
        np.column_stack([r_um, prof]),
        delimiter=",",
        header="r_microns," + ",".join(component_names),
        comments="",
    )

    interp = evaluate_model_at_radii(r_um, prof, epma_obs["r"])
    cols, headers = [epma_obs["r"]], ["r_microns"]
    for comp in component_names:
        cols += [epma_obs[comp], interp[comp], interp[comp] - epma_obs[comp]]
        headers += [f"{comp}_obs", f"{comp}_model", f"{comp}_residual"]
    np.savetxt(
        out_dir / "epma_comparison.csv",
        np.column_stack(cols),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )

    np.savez_compressed(
        out_dir / "best_fit_results.npz",
        r_model_um=r_um,
        profile_final=prof,
        profile_initial=best_diag["initial_profile"],
        r_epma_um=epma_obs["r"],
        **{f"epma_{comp}": epma_obs[comp] for comp in component_names},
        **{f"model_interp_{comp}": interp[comp] for comp in component_names},
        best_fit_intervals_myr=np.asarray(summary["best_fit_intervals_myr"]),
        best_fit_ages_ma=np.asarray(summary["best_fit_ages_ma"]),
        ptt_path=history["PTt_path"],
    )
    print(f"[out] saved JSON, CSV and NPZ results to {out_dir}", flush=True)


def plot_results(out_dir: Path, summary: dict, best_model, best_diag: dict, epma_obs: dict, history: dict) -> None:
    """Plot the best fit: end-member profiles vs EPMA, residuals, and the PTt path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"Fe": "tab:red", "Mg": "tab:green", "Mn": "tab:blue", "Ca": "gold"}
    end_members = {"Fe": "Alm", "Mg": "Py", "Mn": "Sp", "Ca": "Gro"}
    r_um = best_model.r * 1e6
    prof = best_diag["profile_final"]
    initial = best_diag["initial_profile"]
    prograde = best_diag.get("profile_prograde")
    interp = evaluate_model_at_radii(r_um, prof, epma_obs["r"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for (row, col), comp in {(0, 0): "Fe", (0, 1): "Mg", (0, 2): "Mn", (1, 0): "Ca"}.items():
        ax = axes[row, col]
        idx = component_names.index(comp)
        ax.plot(initial[:, 0], initial[:, idx + 1], ls=":", c=colours[comp], label="initial (growth)")
        if prograde is not None:
            ax.plot(r_um, prograde[:, idx], ls="--", c=colours[comp], label="after prograde")
        ax.plot(r_um, prof[:, idx], c=colours[comp], lw=2.0, label="best-fit model")
        ax.scatter(epma_obs["r"], epma_obs[comp], marker="x", c="black", s=25, alpha=0.7,
                   zorder=3, label="EPMA (S10-gt1)")
        ax.set_title(f"{comp} ({end_members[comp]}, RMSE = {summary['best_fit_component_rmse'][comp]:.4f})")
        ax.set_xlabel(r"Radius [$\mu$m]")
        ax.set_ylabel("Mole fraction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax_res = axes[1, 1]
    for comp in component_names:
        ax_res.plot(epma_obs["r"], interp[comp] - epma_obs[comp], "o-", c=colours[comp], ms=3.5, label=f"{comp} res")
    ax_res.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax_res.set_title(f"Residuals (Total RMSE = {summary['best_fit_rmse']:.5f})")
    ax_res.set_xlabel(r"Radius [$\mu$m]")
    ax_res.set_ylabel(r"$X_{\mathrm{model}} - X_{\mathrm{EPMA}}$")
    ax_res.grid(alpha=0.3)
    ax_res.legend(fontsize=8, ncols=2)

    ax_pt = axes[1, 2]
    path = history["PTt_path"]
    ax_pt.plot(path[:, 2], path[:, 1], "r-", label="T [°C]")
    ax_pt.set_xlabel("Time [Myr]")
    ax_pt.set_ylabel("T [°C]", color="tab:red")
    ax_pt.tick_params(axis="y", labelcolor="tab:red")
    ax_pt2 = ax_pt.twinx()
    ax_pt2.plot(path[:, 2], path[:, 0], "b--", label="P [kbar]")
    ax_pt2.set_ylabel("P [kbar]", color="tab:blue")
    ax_pt2.tick_params(axis="y", labelcolor="tab:blue")
    ages = summary["best_fit_ages_ma"]
    ax_pt.set_title(f"Path: t1={ages[0]:.0f}, t2={ages[1]:.0f}, t3={ages[2]:.0f} Ma")
    ax_pt.grid(alpha=0.3)

    fig.suptitle(f"Glenelg S10-gt1 Inversion Fit — Best RMSE: {summary['best_fit_rmse']:.5f}",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"inversion_fit.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[out] wrote inversion fit plots (PDF, PNG) to {out_dir}", flush=True)


# ==============================================================================
# Inversion driver
# ==============================================================================


def run_inversion(options: SimpleNamespace | None = None):
    """Run the full Glenelg S10-gt1 inversion with the settings above."""
    if options is None:
        options = make_options()

    # Growth history: computed once, frozen; only the retrograde path is re-timed.
    cached_history = cache_growth_history(options)
    epma_obs = load_observations(cached_history, options)

    # The grid spans the growth-shell radii (0 .. Rr.max); observations must lie inside it.
    r_grid_max_um = float(np.max(cached_history["Rr"])) * 1e6
    if np.max(epma_obs["r"]) > r_grid_max_um + 1e-6:
        raise ValueError(
            f"EPMA radii reach {np.max(epma_obs['r']):.1f} um but the model grid ends at "
            f"{r_grid_max_um:.1f} um (Rr.max): increase rmax"
        )

    # Initial guess (ages -> positive intervals) must be feasible, then a fresh
    # output directory that never overwrites a previous run.
    x0 = ages_to_intervals(initial_ages)
    for i, (lo, hi) in enumerate(interval_bounds):
        if not lo <= x0[i] <= hi:
            raise ValueError(
                f"initial_ages imply dt[{i}]={x0[i]:g} Myr, outside interval bounds ({lo:g}, {hi:g})"
            )
    out_dir = next_output_dir(
        options.output_dir
        / f"inversion_garnet{options.garnet_no}_nfev={options.max_nfev}_quick={options.quick}_{options.source_label}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Misfit at the initial guess
    init_history = update_history_timing(cached_history, x0, options.pt_npoints)
    init_model, init_diag = run_forward_evaluation(init_history, fixed_bc=options.fixed_bc,
                                                   quick=options.quick, dt_myr=options.dt_myr,
                                                   geometry=options.geometry)
    init_rmse, init_comp, _ = compute_misfit(init_model.r * 1e6, init_diag["profile_final"], epma_obs)
    print(
        f"\n[initial] ages={initial_ages.tolist()} Ma (intervals {x0.tolist()} Myr) | "
        f"RMSE({'/'.join(fit_components)}): {init_rmse:.5f} "
        f"(Fe={init_comp['Fe']:.4f}, Mg={init_comp['Mg']:.4f}, "
        f"Mn={init_comp['Mn']:.4f}, Ca={init_comp['Ca']:.4f})\n",
        flush=True,
    )

    # Optimise the three younger ages through their positive intervals
    state = {
        "count": 0,
        "best_rmse": init_rmse,
        "best_intervals": x0.copy(),
        "best_ages": np.asarray(initial_ages, dtype=float),
        "best_diag": init_diag,
    }
    print(f"[optimiser] Nelder-Mead, max_nfev={options.max_nfev}, bounds={interval_bounds}", flush=True)
    result = minimize(
        objective,
        x0,
        args=(cached_history, epma_obs, options, state),
        method="Nelder-Mead",
        bounds=interval_bounds,
        options={"maxfev": options.max_nfev},
    )

    # Fresh forward run at the best-fit schedule
    best_history = update_history_timing(cached_history, state["best_intervals"], options.pt_npoints)
    best_model, best_diag = run_forward_evaluation(best_history, fixed_bc=options.fixed_bc,
                                                   quick=options.quick, dt_myr=options.dt_myr,
                                                   geometry=options.geometry)
    best_rmse, best_comp, _ = compute_misfit(best_model.r * 1e6, best_diag["profile_final"], epma_obs)

    summary = {
        "source": options.source_label,
        "garnet_no": options.garnet_no,
        "fit_components": list(fit_components),
        "initial_ages_ma": initial_ages.tolist(),
        "interval_bounds_myr": [list(pair) for pair in interval_bounds],
        "best_fit_intervals_myr": state["best_intervals"].tolist(),
        "best_fit_ages_ma": state["best_ages"].tolist(),
        "initial_rmse": init_rmse,
        "best_fit_rmse": best_rmse,
        "best_fit_component_rmse": best_comp,
        "nfev": state["count"],
        "success": bool(result.success),
        "status_message": str(result.message),
        "nominal_target_t3_ma": target_t3,
        "ca_closure_note": (
            "Ca = 1 - Fe - Mg - Mn is the dependent closure component: excluded from "
            "the default Fe/Mg/Mn fit to avoid double-counting, reported as a residual."
        ),
        "settings": {
            "rmin": options.rmin, "rmax": options.rmax, "quick": options.quick,
            "dt_myr": options.dt_myr, "geometry": options.geometry, "fixed_bc": options.fixed_bc,
            "pt_npoints": options.pt_npoints, "synthetic": options.synthetic,
        },
    }
    save_results(out_dir, summary, best_model, best_diag, epma_obs, best_history)
    if options.make_plots:
        plot_results(out_dir, summary, best_model, best_diag, epma_obs, best_history)

    print("\n=======================================================", flush=True)
    print(" Glenelg S10-gt1 inversion complete", flush=True)
    print(f"  status:             {result.message}", flush=True)
    print(f"  evaluations:        {state['count']}", flush=True)
    print(f"  initial ages:       {initial_ages.tolist()} Ma (RMSE {init_rmse:.5f})", flush=True)
    print(f"  best-fit ages:      {np.round(state['best_ages'], 1).tolist()} Ma (RMSE {best_rmse:.5f})", flush=True)
    print(
        f"  per-component RMSE: Fe={best_comp['Fe']:.4f}, Mg={best_comp['Mg']:.4f}, "
        f"Mn={best_comp['Mn']:.4f}, Ca={best_comp['Ca']:.4f} (closure reported, not fitted)",
        flush=True,
    )
    print(f"  t3 vs nominal {target_t3:.0f} Ma target: {state['best_ages'][2] - target_t3:+.1f} Ma", flush=True)
    print(f"  outputs:            {out_dir}\n", flush=True)
    return out_dir, summary


if __name__ == "__main__":
    run_inversion()
