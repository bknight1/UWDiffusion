# %%
"""Glenelg S10 eclogite: garnet multicomponent diffusion in 1D.

Reproduces the MAGEMin/phasetools garnet growth history and the full
prograde-growth + retrograde-rim diffusion schedule of the 2D annulus model

    Multicomponent_diffusion_garnet/Glenelg/S10/Glenelg-eclogite_UWDiffusion_model.py

on a 1D radial finite-volume grid, using ``UWDiffusion.fd_1d.MulticomponentDiffusion1D``
in the same style as the working ``Tutorials/Diffusion/CZGM_garnet_model_1D.ipynb``
tutorial. The MAGEMin growth calculation logic and the Glenelg S10 bulk composition
are taken from ``S10/1D_inversion/garnet_growth_model.py``.

Provenance of the translated logic
----------------------------------
- PTt path, S10 bulk composition (wt%), ``GarnetGenerator`` settings (mpe db,
  dataset 62, 100 garnet classes, 50 growth shells), growth-history extraction
  keys (``Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car``) and the retrograde fallback:
  identical to ``garnet_growth_model.py`` / the 2D model.
- Prograde event loop: translated from the 2D model. Each growth step stamps
  the newly grown + still-future shells with the step composition
  (``apply_growth``; the 2D model's ``r > shell_radius - front_halfwidth``
  front-halvewidth is a finite-element smoothing device and has no 1D
  counterpart), then pins *every* not-yet-grown shell boundary to that
  composition for every independent component — the 2D ``shell_idx >=
  step_idx`` rule applied to all future growth vertices (internal Dirichlet
  pins via ``set_internal_bcs``) plus the outer rim (``set_bc``) — and only
  then diffuses to the next growth time. The already-grown region between the
  core and the growth front carries no pins and is free to diffuse.
- Retrograde loop (re-pin rim Dirichlet to the retrograde rim composition each
  segment when ``fixed_bc`` and Fe_r > 0, else zero flux): translated from the
  2D model; the retrograde schedule is selected with ``np.searchsorted`` at the
  final growth time (no tolerance) and validated to start at/after it.
- Arrhenius diffusivity: Chakraborty & Ganguly (1992) tracer formulation
  ``D = D0 * exp(-(Ea + (P - 1) * Ev / 41.84) / (R * T))`` with D0 in cm2/s,
  Ea in cal/mol, Ev in cm3/mol (41.84 converts cm3*bar to cal), R = 1.9872
  cal/(mol K), **T in Kelvin and P in bar**, exactly as in the 2D source
  (``Tt_path_interp`` interpolates T_data + 273.15; ``Pt_path_interp``
  interpolates P_data * 1e3). Ca diffusivity = 0.05 * D_Mn as in the 2D model.

Geometry choice
---------------
The 2D source solves a Cartesian annulus (x, y) mesh. Its closest 1D radial
equivalent is **cylindrical** weighting (alpha = 1: per-unit-length cylinder,
integrating r dr). This is the default. "spherical" opts into spherical
weighting (alpha = 2, integrating r^2 dr).

Units
-----
- MAGEMin path / growth history: P in kbar, T in degC, t in Myr, radii in um,
  compositions in mole fractions (X-site end-member fractions, FeOt basis;
  Ca is carried by closure X_Ca = 1 - X_Fe - X_Mg - X_Mn).
- 1D solver internals: grid in metres, model time in seconds, diffusivity in
  m2/s (cm2/s tracer values x 1e-4).

The workflow follows the legacy model from top to bottom: construct the PTt
path, generate the MAGEMin growth history, initialise the 1D model, solve the
prograde and retrograde phases, and write profiles and plots. Run settings are
edited directly below; there is no CLI.
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from UWDiffusion.fd_1d import (
    MulticomponentDiffusion1D,
    darken_reduced_diffusivity,
)

# ==============================================================================
# Model settings
# ==============================================================================

sec_per_myr = 1.0e6 * 365.25 * 24.0 * 3600.0  # seconds per Myr (SI time unit)

component_names = ["Fe", "Mg", "Mn", "Ca"]  # order matches tracer-diffusivity arrays
components = component_names  # public alias used by existing plotting/tests
dependent = "Ca"  # closure: Ca = 1 - Fe - Mg - Mn
dependent_idx = 3

# PTt path vertices (P in kbar, T in degC, t in model Myr)
p_arr = np.array([6.0, 20.0, 12.2, 14.0, 10.6])  # kbar
t_arr = np.array([500.0, 700.0, 725.0, 600.0, 510.0])  # degC
t_ma_arr = np.array([1060, 1000, 910, 810, 710])  # Ma; model time = max - value

pt_npoints_source = 501  # PTt path resolution used by the source scripts

# MAGEMin garnet growth settings
db = "mpe"  # eclogite assemblage: mp + mb (includes cpx, absent from mp)
dataset = 62
gt_classes = 100  # number of garnet size classes in the population
gt_nr = 50  # number of growth shells per garnet

# Glenelg S10 bulk rock, wt% oxides
xoxides = ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO", "CaO", "Na2O", "K2O", "H2O"]
x_s10_wt = [51.6, 1.41, 14.1, 1.62, 13.1, 0.25, 5.36, 9.89, 2.19, 0.36, 0.26]

# Chakraborty & Ganguly (1992) tracer diffusivity parameters
d0_cm2s = {"Fe": 6.36e-4, "Mg": 1.11e-3, "Mn": 5.15e-4}  # cm^2/s
ea_cal = {"Fe": 65824.0, "Mg": 67997.0, "Mn": 60569.0}  # cal/mol
ev_cm3 = {"Fe": 5.63, "Mg": 5.27, "Mn": 6.04}  # cm^3/mol
r_cal = 1.9872  # cal/(mol K)
d_ca_scale = 0.05  # D_Ca = 0.05 * D_Mn (2D model default)

# EPMA garnet profiles for the comparison plot
garnet_no_label = {0: "gt1", 40: "gt2"}
epma_csv_paths = {
    0: Path(
        "/Users/benknight/Documents/Research/Modelling/UW-models/Diffusion_modelling/"
        "Fe-Mg_exchange/gt-cpx_modelling/glenelg-s10/data/S10-gt1.csv"
    ),
    40: Path(
        "/Users/benknight/Documents/Research/Modelling/UW-models/Diffusion_modelling/"
        "Fe-Mg_exchange/gt-cpx_modelling/glenelg-s10/data/S10-gt2.csv"
    ),
}

# ==============================================================================
# run settings: edit these directly; there is no CLI
# ==============================================================================

garnet_no = 0  # garnet size class (0 = gt1, 40 = gt2 in the source scripts)
growth_start = 0.0  # growth window start (Myr)
growth_end = 100.0  # growth window end (Myr)
rmin, rmax = 50.0, 380.0  # garnet radius range (um)
fixed_bc = True  # pin the rim Dirichlet to the retrograde rim composition each segment
quick = False  # coarser grid: use every 2nd growth-shell radius as a vertex
dt_myr = 2.0  # maximum time step (Myr)
geometry = "cylindrical"  # "cylindrical" (alpha=1) or "spherical" (alpha=2)
pt_npoints = pt_npoints_source  # PTt path resolution for MAGEMin
synthetic = False  # synthetic growth history (testing only; no MAGEMin/Julia required)
output_dir = Path("./output/glenelg_s10_1d_test")
plot_results = True


# ==============================================================================
# Diffusivity (Chakraborty & Ganguly 1992, 2D-model conventions)
# ==============================================================================


def tracer_diffusivities_m2s(temp_K: float, pressure_bar: float) -> np.ndarray:
    """Tracer diffusivities (m2/s) for [Fe, Mg, Mn, Ca].

    Chakraborty & Ganguly (1992) Arrhenius form, identical to the 2D Glenelg
    source: ``D = D0 * exp(-(Ea + (P - 1) * Ev / 41.84) / (R * T))``.
    """
    D = {}
    for elem in ("Fe", "Mg", "Mn"):
        exponent = -(ea_cal[elem] + (pressure_bar - 1.0) * ev_cm3[elem] / 41.84) / (
            r_cal * temp_K
        )
        D[elem] = d0_cm2s[elem] * np.exp(exponent) * 1.0e-4  # cm2/s -> m2/s
    D["Ca"] = d_ca_scale * D["Mn"]
    return np.array([D[name] for name in components])


def make_diffusivity_callable(T_of_t, P_of_t):
    """Build the composition-dependent reduced interdiffusion matrix callable."""

    def update_diffusivities_from_current_time(t, X):
        t_myr = float(t) / sec_per_myr
        temp_K = float(T_of_t(t_myr))
        pressure_bar = float(P_of_t(t_myr))
        D_tracer = tracer_diffusivities_m2s(temp_K, pressure_bar)

        if isinstance(X, dict):
            X_vec = np.array([float(X[name]) for name in components])
        else:
            X_vec = np.asarray(X, dtype=float)
        X_vec = np.clip(X_vec, 0.0, 1.0)

        return darken_reduced_diffusivity(D_tracer, X_vec, dependent_component=dependent_idx)

    return update_diffusivities_from_current_time


# ==============================================================================
# PTt path
# ==============================================================================


def build_ptt_path(pt_npoints: int = pt_npoints_source):
    """Build the smooth PTt path and T(K)/P(bar) interpolants."""
    import phasetools

    t_myr = float(t_ma_arr.max()) - t_ma_arr.astype(float)
    PTt_path = phasetools.utils.create_PTt_path(p_arr, t_arr, t_myr, pt_npoints)

    t_data = PTt_path[:, 2]
    T_of_t = interpolate.interp1d(t_data, PTt_path[:, 1] + 273.15, fill_value="extrapolate")
    P_of_t = interpolate.interp1d(t_data, PTt_path[:, 0] * 1e3, fill_value="extrapolate")
    return PTt_path, T_of_t, P_of_t


# ==============================================================================
# Growth histories
# ==============================================================================


def generate_growth_history_magemin(*, garnet_no=garnet_no, rmin=rmin, rmax=rmax,
                                    pt_npoints=pt_npoints) -> dict:
    """Generate the shell-resolved garnet growth history with MAGEMin."""
    import phasetools
    from juliacall import Main as jl, convert as jlconvert
    from phasetools import GarnetGenerator

    MAGEMin_C = phasetools.MAGEMin_C

    PTt_path, T_of_t, P_of_t = build_ptt_path(pt_npoints)
    P_data, T_data, t_data = PTt_path[:, 0], PTt_path[:, 1], PTt_path[:, 2]

    sys_in = "wt"
    x_mol, xoxides_magemin = MAGEMin_C.convertBulk4MAGEMin(
        jlconvert(jl.Vector[jl.Float64], x_s10_wt),
        jlconvert(jl.Vector[jl.String], xoxides),
        sys_in,
        db,
    )
    xoxides_magemin = list(xoxides_magemin)
    x_mol = np.array(x_mol) / 100.0
    sys_in = "mol"
    rm_list = None

    print(
        f"[MAGEMin] GarnetGenerator db={db} dataset={dataset}: "
         f"{pt_npoints} path points, r_min={rmin} um, r_max={rmax} um, "
        f"{gt_classes} classes x {gt_nr} shells ...",
        flush=True,
    )
    garnet_generator = GarnetGenerator(db=db, dataset=dataset)
    garnet_generator.setup_bulk_composition(Xoxides=xoxides_magemin, X=x_mol, sys_in=sys_in, rm_list=rm_list)
    garnet_generator.generate_garnet_data(
        P_data,
        T_data,
        t_data,
        r_min=rmin,
        r_max=rmax,
        garnet_classes=gt_classes,
        nR_diff=gt_nr,
        fractionate=False,
        normalise_start=True,
    )
    garnet_data = garnet_generator.generate_garnets("N")

    keys = ["Rr", "tr", "Pr", "Tr", "Mnr", "Mgr", "Fer", "Car"]
    Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car = (garnet_data[garnet_no][k] for k in keys)

    t_p, T_p, P_p, Mn_p, Mg_p, Fe_p, Ca_p = garnet_generator.get_prograde_concentrations()
    t_r, T_r, P_r, Mn_r, Mg_r, Fe_r, Ca_r = garnet_generator.get_retrograde_concentrations()

    if len(t_r) < 10:
        if garnet_generator.X_last_growth is None:
            raise RuntimeError(
                "Retrograde fallback requested but no garnet growth was detected "
                "along the path (X_last_growth is None)."
            )
        print("[MAGEMin] retrograde fallback: recomputing rim compositions", flush=True)
        P_r = P_data[t_data >= growth_end]
        T_r = T_data[t_data >= growth_end]
        t_r = t_data[t_data >= growth_end]

        garnet_calculator = phasetools.MAGEMinGarnetCalculator(db=db, dataset=dataset)
        garnet_calculator.setup_bulk_composition(xoxides_magemin, garnet_generator.X_last_growth, sys_in, rm_list)
        retrograde_EM_arr = np.zeros(shape=(P_r.shape[0], 4))
        for i in range(len(P_r)):
            _gt_frac, _gt_wt, _gt_vol, Mg_i, Mn_i, Fe_i, Ca_i, _out = (
                garnet_calculator.gt_single_point_calc_elements(P_r[i], T_r[i])
            )
            retrograde_EM_arr[i] = np.column_stack([Mn_i, Mg_i, Fe_i, Ca_i])
        Mn_r = retrograde_EM_arr[:, 0]
        Mg_r = retrograde_EM_arr[:, 1]
        Fe_r = retrograde_EM_arr[:, 2]
        Ca_r = retrograde_EM_arr[:, 3]

    return {
        "Rr": np.asarray(Rr, dtype=float),
        "tr": np.asarray(tr, dtype=float),
        "Pr": np.asarray(Pr, dtype=float),
        "Tr": np.asarray(Tr, dtype=float),
        "Mnr": np.asarray(Mnr, dtype=float),
        "Mgr": np.asarray(Mgr, dtype=float),
        "Fer": np.asarray(Fer, dtype=float),
        "Car": np.asarray(Car, dtype=float),
        "prograde": {
            "t": np.asarray(t_p, float), "T": np.asarray(T_p, float), "P": np.asarray(P_p, float),
            "Mn": np.asarray(Mn_p, float), "Mg": np.asarray(Mg_p, float),
            "Fe": np.asarray(Fe_p, float), "Ca": np.asarray(Ca_p, float),
        },
        "retrograde": {
            "t": np.asarray(t_r, float), "T": np.asarray(T_r, float), "P": np.asarray(P_r, float),
            "Mn": np.asarray(Mn_r, float), "Mg": np.asarray(Mg_r, float),
            "Fe": np.asarray(Fe_r, float), "Ca": np.asarray(Ca_r, float),
        },
        "PTt_path": PTt_path,
        "T_of_t": T_of_t,
        "P_of_t": P_of_t,
        "source": "magemin",
        "bulk_rock": "Glenelg S10 (wt%): " + ", ".join(f"{o}={v}" for o, v in zip(xoxides, x_s10_wt)),
    }


def generate_growth_history_synthetic(*, rmin=rmin, rmax=rmax) -> dict:
    """Synthetic growth history with the same structure as the MAGEMin one."""
    n_growth = 12
    t_growth = np.linspace(10.0, 50.0, n_growth)  # Myr
    r_shells = np.linspace(rmin, rmax, n_growth)  # um

    Rr = np.concatenate([[0.0], r_shells])
    tr = np.concatenate([[t_growth[0]], t_growth])
    Fe = np.concatenate([[0.72], np.linspace(0.70, 0.66, n_growth)])
    Mg = np.concatenate([[0.11], np.linspace(0.12, 0.15, n_growth)])
    Mn = np.concatenate([[0.09], np.linspace(0.08, 0.03, n_growth)])
    Ca = 1.0 - Fe - Mg - Mn

    t_nodes = np.array([0.0, 50.0, 90.0])
    T_nodes = np.array([500.0, 725.0, 510.0])
    P_nodes = np.array([8.0, 16.0, 9.0])
    PTt_path = np.column_stack([P_nodes, T_nodes, t_nodes])
    T_of_t = interpolate.interp1d(t_nodes, T_nodes + 273.15, fill_value="extrapolate")
    P_of_t = interpolate.interp1d(t_nodes, P_nodes * 1e3, fill_value="extrapolate")

    t_end = 90.0
    t_r = np.linspace(t_growth[-1], t_end, 7)
    wiggle = np.linspace(0.0, 0.01, t_r.size)
    return {
        "Rr": Rr, "tr": tr,
        "Pr": np.interp(tr, t_nodes, P_nodes), "Tr": np.interp(tr, t_nodes, T_nodes),
        "Mnr": Mn, "Mgr": Mg, "Fer": Fe, "Car": Ca,
        "prograde": {
            "t": t_growth, "T": np.interp(t_growth, t_nodes, T_nodes),
            "P": np.interp(t_growth, t_nodes, P_nodes),
            "Mn": Mn[1:], "Mg": Mg[1:], "Fe": Fe[1:], "Ca": Ca[1:],
        },
        "retrograde": {
            "t": t_r, "T": np.interp(t_r, t_nodes, T_nodes), "P": np.interp(t_r, t_nodes, P_nodes),
            "Mn": Mn[-1] + wiggle, "Mg": Mg[-1] - wiggle,
            "Fe": Fe[-1] - wiggle, "Ca": 1.0 - (Fe[-1] - wiggle) - (Mg[-1] - wiggle) - (Mn[-1] + wiggle),
        },
        "PTt_path": PTt_path,
        "T_of_t": T_of_t,
        "P_of_t": P_of_t,
        "source": "synthetic",
        "bulk_rock": "synthetic schedule (no MAGEMin)",
    }


# %% [markdown]
# ## 1D radial diffusion model
#
# The remaining cells intentionally stay executable from top to bottom.  The
# radial grid and finite-volume solver are the only spatial substitutions for
# the legacy annulus workflow.

# %% [markdown]
# #### PTt path

# %%
if synthetic:
    synthetic_time = np.array([0.0, 50.0, 90.0])
    synthetic_temperature = np.array([500.0, 725.0, 510.0])
    synthetic_pressure = np.array([8.0, 16.0, 9.0])
    PTt_path = np.column_stack([synthetic_pressure, synthetic_temperature, synthetic_time])
    T_of_t = interpolate.interp1d(synthetic_time, synthetic_temperature + 273.15, fill_value="extrapolate")
    P_of_t = interpolate.interp1d(synthetic_time, synthetic_pressure * 1e3, fill_value="extrapolate")
else:
    PTt_path, T_of_t, P_of_t = build_ptt_path(pt_npoints)
P_data, T_data, t_data = PTt_path[:, 0], PTt_path[:, 1], PTt_path[:, 2]

# %% [markdown]
# #### MAGEMin bulk, garnet generation, and unpacking

# %%
source_label = "synthetic" if synthetic else "magemin"
if rmin <= 0 or rmax <= rmin:
    raise ValueError(f"rmax ({rmax}) must be greater than rmin ({rmin}) > 0")
if dt_myr <= 0 or pt_npoints < 5:
    raise ValueError("dt_myr must be positive and pt_npoints must be >= 5")

if synthetic:
    history = generate_growth_history_synthetic(rmin=rmin, rmax=rmax)
    print("[setup] synthetic growth history (no MAGEMin)", flush=True)
else:
    history = generate_growth_history_magemin(
        garnet_no=garnet_no, rmin=rmin, rmax=rmax, pt_npoints=pt_npoints
    )
# %%
base_output_dir = output_dir / (
    f"rmin={rmin}_rmax={rmax}-tstart_growth={growth_start}Myr"
    f"_tend_growth={growth_end}Myr-garnet_no={garnet_no}_fixedBC={fixed_bc}"
    f"-geometry={geometry}_quick={quick}_{source_label}"
)
out_dir = base_output_dir
suffix = 0
while out_dir.is_dir():
    suffix += 1
    out_dir = Path(f"{base_output_dir}{suffix}")
out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(Path(__file__).resolve(), out_dir)

# %% [markdown]
# #### Growth evolution CSV

# %%
growth_slice = history["tr"] > history["tr"].min()
np.savetxt(
    out_dir / "ptt_path.csv", history["PTt_path"], delimiter=",",
    header="P_kbar,T_C,t_Myr", comments="",
)
np.savetxt(
    out_dir / "garnet_growth_evolution.csv",
    np.column_stack([
        history["tr"][growth_slice], history["Rr"][growth_slice],
        history["Pr"][growth_slice], history["Tr"][growth_slice],
        history["Mgr"][growth_slice], history["Fer"][growth_slice],
        history["Car"][growth_slice], history["Mnr"][growth_slice],
    ]), delimiter=",", header="time_Myr,radius_microns,P_kbar,T_C,Mg,Fe,Ca,Mn",
    comments="",
)
retro = history["retrograde"]
np.savetxt(
    out_dir / "retrograde_path.csv",
    np.column_stack([retro["t"], retro["T"], retro["P"], retro["Fe"], retro["Mg"], retro["Mn"], retro["Ca"]]),
    delimiter=",", header="time_Myr,T_C,P_kbar,Fe,Mg,Mn,Ca", comments="",
)

# %% [markdown]
# #### Radial grid, model, and initial composition

# %%
Rr, tr = history["Rr"], history["tr"]
Mnr, Mgr, Fer, Car = history["Mnr"], history["Mgr"], history["Fer"], history["Car"]
radial_growth = Rr[growth_slice]
time_evolution = tr[growth_slice]
Fe_evolution, Mg_evolution = Fer[growth_slice], Mgr[growth_slice]
Mn_evolution, Ca_evolution = Mnr[growth_slice], Car[growth_slice]
_, unique_idx = np.unique(time_evolution, return_index=True)
unique_idx = np.sort(unique_idx)
radial_growth, time_evolution = radial_growth[unique_idx], time_evolution[unique_idx]
Fe_evolution, Mg_evolution = Fe_evolution[unique_idx], Mg_evolution[unique_idx]
Mn_evolution, Ca_evolution = Mn_evolution[unique_idx], Ca_evolution[unique_idx]

stride = 2 if quick else 1
r_grid_um = np.unique(Rr)[::stride]
if r_grid_um[-1] < Rr.max():
    r_grid_um = np.append(r_grid_um, Rr.max())
model = MulticomponentDiffusion1D(
    grid=r_grid_um * 1.0e-6,
    diffusivity=make_diffusivity_callable(history["T_of_t"], history["P_of_t"]),
    geometry=geometry, components=component_names, dependent_component=dependent,
    initial_composition={"Fe": Fer[0], "Mg": Mgr[0], "Mn": Mnr[0]},
)
print(
    f"[1D] grid: {model.n_points} vertices, 0-{r_grid_um[-1]:.1f} um, "
    f"geometry={model.geometry} (alpha={model.alpha}); growth steps: {len(time_evolution)} "
    f"({time_evolution[0]:.2f}-{time_evolution[-1]:.2f} Myr); dt <= {dt_myr} Myr",
    flush=True,
)

# %% [markdown]
# #### Growth hook and solver controls

# %%
def apply_growth_step(step_idx: int) -> None:
    """Stamp one 1D shell and pin all future shell vertices."""
    growth_radius = radial_growth[step_idx] * 1.0e-6
    composition = {"Fe": float(Fe_evolution[step_idx]), "Mg": float(Mg_evolution[step_idx]), "Mn": float(Mn_evolution[step_idx])}
    model.apply_growth(growth_radius, composition, set_rim_bc=False)
    future_vertices = model.r >= growth_radius
    future_vertices[-1] = False
    model.set_internal_bcs(future_vertices, composition)
    model.set_bc("right", "dirichlet", composition)


dt_s = dt_myr * sec_per_myr
model.reset(time_evolution[0] * sec_per_myr)

# %% [markdown]
# #### Prograde growth and diffusion

# %%
for i in range(len(time_evolution) - 1):
    apply_growth_step(i)
    model.run_to_time(time_evolution[i + 1] * sec_per_myr, dt_s)
print(f"[1D] prograde done at t = {model.time / sec_per_myr:.2f} Myr", flush=True)

# %% [markdown]
# #### Final growth shell and phase transition

# %%
model.clear_internal_bcs()
apply_growth_step(len(radial_growth) - 1)
model.reset(model.time)
profile_prograde = model.get_composition().copy()
mass_before_retro = model.total_mass()

# %% [markdown]
# #### Retrograde boundary update and diffusion

# %%
t_r = retro["t"]
Fe_r, Mg_r, Mn_r = retro["Fe"], retro["Mg"], retro["Mn"]
model.clear_internal_bcs()
if not fixed_bc:
    model.set_bc("rim", "neumann", 0.0)
t_growth_final_myr = float(time_evolution[-1])
if t_r.size == 0:
    raise ValueError("Retrograde schedule is empty; cannot run retrograde diffusion.")
retro_start = int(np.searchsorted(t_r, t_growth_final_myr))
if retro_start >= len(t_r):
    raise ValueError(
        f"Retrograde schedule does not reach the final growth time ({t_growth_final_myr:.3f} Myr): "
        f"last retrograde time is {t_r[-1]:.3f} Myr."
    )


def update_retrograde_boundary(retro_step: int) -> None:
    """Apply the 2D outer-rim update to the 1D right boundary."""
    if fixed_bc and Fe_r[retro_step] > 0:
        model.set_bc("rim", "dirichlet", {"Fe": float(Fe_r[retro_step]), "Mg": float(Mg_r[retro_step]), "Mn": float(Mn_r[retro_step])})
    else:
        model.set_bc("rim", "neumann", 0.0)


for i in range(retro_start, len(t_r) - 1):
    if fixed_bc:
        update_retrograde_boundary(i)
    if t_r[i + 1] > model.time / sec_per_myr:
        model.run_to_time(t_r[i + 1] * sec_per_myr, dt_s)
print(f"[1D] retrograde done at t = {model.time / sec_per_myr:.2f} Myr", flush=True)

# %% [markdown]
# #### Profiles and output summary

# %%
diagnostics = {
    "radial_growth": radial_growth, "time_evolution": time_evolution,
    "Fe_evolution": Fe_evolution, "Mg_evolution": Mg_evolution,
    "Mn_evolution": Mn_evolution, "Ca_evolution": Ca_evolution,
    "initial_profile": np.column_stack([Rr, Fer, Mgr, Mnr, Car]),
    "profile_prograde": profile_prograde, "profile_final": model.get_composition().copy(),
    "mass_before_retro": mass_before_retro, "mass_after_retro": model.total_mass(),
    "retro_idx": np.arange(retro_start, len(t_r)),
    "retro_start_myr": float(t_r[retro_start]), "growth_end_myr": t_growth_final_myr,
}
r_um = model.r * 1.0e6
np.savetxt(out_dir / "_EM_initial_F.csv", diagnostics["initial_profile"], delimiter=",", header="Rr_um,Fe,Mg,Mn,Ca", comments="")
np.savetxt(out_dir / "prograde_profiles.csv", np.column_stack([r_um, profile_prograde]), delimiter=",", header="r_microns," + ",".join(components), comments="")
np.savetxt(out_dir / "final_profiles.csv", np.column_stack([r_um, diagnostics["profile_final"]]), delimiter=",", header="r_microns," + ",".join(components), comments="")
settings = {"garnet_no": garnet_no, "growth_start": growth_start, "growth_end": growth_end, "rmin": rmin, "rmax": rmax, "fixed_bc": fixed_bc, "quick": quick, "dt_myr": dt_myr, "geometry": geometry, "pt_npoints": pt_npoints, "synthetic": synthetic, "output_root": str(output_dir), "make_plots": plot_results}
summary = {
    "source": history["source"], "bulk_rock": history["bulk_rock"], "settings": settings,
    "geometry": model.geometry, "alpha": model.alpha, "n_points": int(model.n_points),
    "grid_max_um": float(model.r[-1] * 1e6), "n_growth_steps": int(time_evolution.size),
    "growth_window_myr": [float(time_evolution[0]), float(time_evolution[-1])],
    "final_time_myr": float(model.time / sec_per_myr),
    "max_abs_closure_error": float(np.max(np.abs(diagnostics["profile_final"].sum(axis=1) - 1.0))),
    "mass_before_retrograde": {k: float(v) for k, v in diagnostics["mass_before_retro"].items()},
    "mass_after_retrograde": {k: float(v) for k, v in diagnostics["mass_after_retro"].items()},
}
(out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
print(f"[out] wrote CSV/JSON outputs to {out_dir}", flush=True)


def make_plots_output(out_dir: Path, history: dict, model, diagnostics: dict, *, garnet_no, fixed_bc) -> None:
    """Save PT path, growth history and final profile comparison plots (PDF)."""
    matplotlib.use("Agg")

    colours = {"Fe": "tab:red", "Mg": "tab:green", "Mn": "tab:blue", "Ca": "gold"}
    PTt = history["PTt_path"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(PTt[:, 1], PTt[:, 0], "k--", label="PTt path")
    ax1.set_xlabel(r"T [$^\circ$C]")
    ax1.set_ylabel("P [kbar]")
    ax1.set_xlim(250, 850)
    ax1.grid(alpha=0.5)
    ax1.legend()
    ax2.plot(PTt[:, 2], PTt[:, 1], c="tab:red", label="T")
    ax2.plot(PTt[:, 2], PTt[:, 0], c="tab:blue", label="P")
    t0, t1 = diagnostics["time_evolution"][0], diagnostics["time_evolution"][-1]
    ax2.axvspan(t0, t1, alpha=0.2, color="grey", label="growth window")
    ax2.set_xlabel("time (Myr)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.suptitle(f"Glenelg S10 PTt path ({history['source']})")
    fig.tight_layout()
    fig.savefig(out_dir / "PT_path.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["tr"], history["Rr"], "k.-")
    axes[0].set_xlabel("t (Myr)")
    axes[0].set_ylabel(r"R [$\mu$m]")
    axes[0].set_title("growth history")
    for comp, key in zip(("Fe", "Mg", "Mn", "Ca"), ("Fer", "Mgr", "Mnr", "Car")):
        axes[1].plot(history["Rr"], history[key], c=colours[comp], label=comp)
    axes[1].set_xlabel(r"R [$\mu$m]")
    axes[1].set_ylabel("mole fraction")
    axes[1].set_title("initial zoning (growth)")
    axes[1].legend(ncols=2)
    axes[1].grid(alpha=0.3)
    retro = history["retrograde"]
    for comp in ("Fe", "Mg", "Mn", "Ca"):
        axes[2].plot(retro["t"], retro[comp], c=colours[comp], label=comp)
    axes[2].set_xlabel("t (Myr)")
    axes[2].set_ylabel("mole fraction")
    axes[2].set_title("retrograde rim composition")
    axes[2].legend(ncols=2)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "growth_history.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    r_um = model.r * 1.0e6
    initial = diagnostics["initial_profile"]
    prograde = np.column_stack([r_um, diagnostics["profile_prograde"]])
    final = np.column_stack([r_um, diagnostics["profile_final"]])
    for ax, comp, col in zip(axes, components, range(1, 5)):
        ax.plot(initial[:, 0], initial[:, col], ls=":", c=colours[comp], label="initial (growth)")
        ax.plot(prograde[:, 0], prograde[:, col], ls="--", c=colours[comp], label="after prograde")
        ax.plot(final[:, 0], final[:, col], c=colours[comp], label="final (model)")
        ax.set_title(comp)
        ax.set_xlabel(r"r [$\mu$m]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mole fraction")
    epma = _load_epma_profile(garnet_no)
    if epma is not None:
        for ax, comp in zip(axes, components):
            ax.scatter(epma["r"], epma[comp], marker="x", c="k", zorder=0, label="EPMA")
    axes[0].legend(ncols=2, fontsize=8)
    fig.suptitle(
        f"Glenelg S10 1D ({model.geometry}, garnet {garnet_no}, "
        f"fixed_bc={fixed_bc}) — final profiles at t = {model.time / sec_per_myr:.1f} Myr"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "final_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[out] wrote plots to {out_dir}", flush=True)


def _load_epma_profile(garnet_no_idx: int):
    """Load the measured garnet profile for the comparison plot, if present."""
    csv_path = epma_csv_paths.get(garnet_no_idx)
    if csv_path is None or not csv_path.exists():
        return None
    gt_df = pd.read_csv(csv_path)
    gt_data = gt_df[["X", "Py", "Alm", "Gro", "Sp"]].rename(
        columns={"X": "r", "Py": "Mg", "Alm": "Fe", "Gro": "Ca", "Sp": "Mn"}
    )
    gt_data["r"] = gt_data["r"] * 1e3  # mm -> um
    if garnet_no_idx == 0:
        gt_data["r"] = gt_data["r"] - (gt_data["r"].max() / 2)
    return gt_data

# %%
if plot_results:
    make_plots_output(
        out_dir, history, model, diagnostics,
        garnet_no=garnet_no, fixed_bc=fixed_bc,
    )
print("\nretrograde mass budget (geometry-weighted integrals):", flush=True)
for comp in components:
    change = (diagnostics["mass_after_retro"][comp] - diagnostics["mass_before_retro"][comp]) / diagnostics["mass_before_retro"][comp]
    print(f"  {comp}: relative change = {change:+.2e}", flush=True)
