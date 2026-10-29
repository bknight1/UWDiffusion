# %%
"""Which `min_dt` floor resolves RETROGRADE zoning?

Different problem from a diffusing step.  Here the outer rim is held at (and
updated to) a matrix composition to mimic rim exchange with the matrix, so the
feature of interest is the **boundary layer growing inward from the rim** as
the grain cools.  Its length scale is sqrt(D t), not the zone width, so a
timestep that is fine for a step can be much too coarse here -- or vice versa.

Retrograde limb of the supplied path (kbar, degC, Myr):

    t = [0, 20, 50, 100, 150]      P = [1, 14, 10, 8, 6.5]      T = [600, 740, 780, 700, 500]

Run from t = 50 Myr (peak T) to 150 Myr (the cooling limb).  The rim
composition is reset at each snapshot to a matrix value that evolves along the
path, and the resulting inward profile is what "records" the retrograde
history.

Timesteps compared against a fine Backward-Euler reference.  Error is reported
globally and restricted to the rim zone where the zoning actually lives.

Run from the repository root in the UW3 environment:

    python examples/retrograde_rim_dt_study.py
"""

# %%
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from UWDiffusion.fd_1d import MulticomponentDiffusion1D, lasaga_reduced_diffusivity
from UWDiffusion.tracer_diffusivities import CG92

SEC_PER_MYR = 1.0e6 * 365.25 * 24.0 * 3600.0

# %%
# Supplied path, full extent; we run only the cooling limb.
P_KBAR = np.array([1.0, 14.0, 10.0, 8.0, 6.5])
T_DEGC = np.array([600.0, 740.0, 780.0, 700.0, 500.0])
T_MYR = np.array([0.0, 20.0, 50.0, 100.0, 150.0])

P_interp = interp1d(T_MYR, P_KBAR * 1e-4, fill_value="extrapolate")   # -> GPa
T_interp = interp1d(T_MYR, T_DEGC + 273.15, fill_value="extrapolate")  # -> K

T_START_MYR = 50.0
T_END_MYR = 150.0

RADIUS_UM = 100.0
N_POINTS = 201
RIM_DEPTH_UM = 20.0   # report error over the outer 20 um, where zoning lives

COMPONENTS = ["Fe", "Mg", "Mn", "Ca"]
DEPENDENT = "Ca"

# Grain interior at peak T (prograde, equilibrated).
INTERIOR = {"Fe": 0.35, "Mg": 0.20, "Mn": 0.03}

# Matrix composition the rim is set to.  Evolves along the cooling limb to
# mimic continuing exchange: Fe-poor, Mg-rich as T falls.
_MATRIX_A = {"Fe": 0.30, "Mg": 0.28, "Mn": 0.05}
_MATRIX_B = {"Fe": 0.22, "Mg": 0.32, "Mn": 0.09}


def matrix_at(t_myr: float) -> dict:
    frac = (t_myr - T_START_MYR) / (T_END_MYR - T_START_MYR)
    frac = min(max(frac, 0.0), 1.0)
    return {k: (1 - frac) * _MATRIX_A[k] + frac * _MATRIX_B[k] for k in _MATRIX_A}


# Snapshot / rim-reset times.  Shared by every dt so the comparison is fair:
# each run steps exactly to these times and gets the same rim history.
SNAPSHOT_MYR = [55.0, 60.0, 70.0, 80.0, 100.0, 125.0, 150.0]

FLOORS_MYR = [1.0, 0.1, 0.05]
EXTRA_MYR = [0.01]
REF_MYR = 0.005
REF_CHECK_MYR = 0.0025


# %%
# Diffusivity: CG92 tracers -> Lasaga reduced matrix.  T and P depend only on
# model time, but the solver re-evaluates at every face every step, so cache on
# the time value (this alone was a ~90x speedup in the step study).
_TRACER_CACHE = {}


def tracer_diffusivities_m2s(temp_K, pressure_GPa):
    return np.array([
        CG92["Fe"].at(temp_K, pressure_GPa),
        CG92["Mg"].at(temp_K, pressure_GPa),
        CG92["Mn"].at(temp_K, pressure_GPa),
        CG92["Mn"].at(temp_K, pressure_GPa) / 20.0,
    ])


def reduced_matrix(t, X):
    key = float(t)
    d_tracer = _TRACER_CACHE.get(key)
    if d_tracer is None:
        t_myr = key / SEC_PER_MYR
        d_tracer = tracer_diffusivities_m2s(float(T_interp(t_myr)), float(P_interp(t_myr)))
        _TRACER_CACHE[key] = d_tracer
    if isinstance(X, dict):
        x_vec = np.array([float(X[name]) for name in COMPONENTS])
    else:
        x_vec = np.asarray(X, dtype=float)
    return lasaga_reduced_diffusivity(d_tracer, x_vec, dependent_component=COMPONENTS.index(DEPENDENT))


# %%
# What the rim boundary layer looks like along this limb.

print("Retrograde limb")
print(f"  t = {T_START_MYR:g} -> {T_END_MYR:g} Myr")
for t_myr in (50, 60, 75, 100, 125, 150):
    T = float(T_interp(t_myr))
    P = float(P_interp(t_myr))
    d = CG92["Mn"].at(T, P)
    taus = [(l * 1e-6) ** 2 / d / SEC_PER_MYR for l in (5.0, 20.0, RADIUS_UM)]
    print(f"  t={t_myr:5.0f} Myr  T={T-273.15:5.0f} degC  D_Mn={d:.3e}  "
          f"tau(5um)={taus[0]:.3e}  tau(20um)={taus[1]:.3e}  tau(R)={taus[2]:.3e} Myr")

# %%
# One run at a given dt, resetting the rim at each snapshot.


def make_model():
    grid = np.linspace(0.0, RADIUS_UM * 1e-6, N_POINTS)
    model = MulticomponentDiffusion1D(
        grid=grid,
        diffusivity=reduced_matrix,
        geometry="cylindrical",
        components=COMPONENTS,
        dependent_component=DEPENDENT,
        initial_composition={k: np.full(N_POINTS, v) for k, v in INTERIOR.items()},
        bcs={"left": ("neumann", 0.0), "right": ("dirichlet", matrix_at(T_START_MYR))},
    )
    return model


def run_at(dt_myr):
    t0 = time.perf_counter()
    model = make_model()
    model.reset(T_START_MYR * SEC_PER_MYR)
    snaps = []
    for t_myr in SNAPSHOT_MYR:
        model.set_bc("rim", "dirichlet", matrix_at(t_myr))  # rim exchange
        model.run_to_time(t_myr * SEC_PER_MYR, dt_myr * SEC_PER_MYR)
        snaps.append(np.array([model.get_composition(c) for c in COMPONENTS]))
    mass = model.total_mass()
    return {
        "dt_myr": dt_myr,
        "profiles": np.stack(snaps),                 # (n_snap, n_comp, n_pts)
        "snapshots_myr": list(SNAPSHOT_MYR),
        "n_steps": int(round((T_END_MYR - T_START_MYR) / dt_myr)),
        "seconds": time.perf_counter() - t0,
        "mass": {k: float(v) for k, v in mass.items()} if isinstance(mass, dict) else float(mass),
    }


# %%
print()
print("running reference ...")
ref = run_at(REF_MYR)
print(f"  dt={REF_MYR} Myr: {ref['seconds']:.1f}s")
ref_check = run_at(REF_CHECK_MYR)
print(f"  reference self-consistency (0.005 vs 0.0025): "
      f"{float(np.max(np.abs(ref['profiles'] - ref_check['profiles']))):.3e}")
print()

runs = [ref]
for dt in FLOORS_MYR + EXTRA_MYR:
    r = run_at(dt)
    runs.append(r)
    print(f"  dt={dt} Myr: {r['seconds']:.1f}s")

# %%
grid = np.linspace(0.0, RADIUS_UM * 1e-6, N_POINTS)
r_um = grid * 1e6
rim_zone = r_um >= (RADIUS_UM - RIM_DEPTH_UM)
interior = ~rim_zone

print()
print("### Error vs the fine reference  (max over all snapshot times)")
print("| dt (Myr) | steps | wall_s | max abs | RMS | in rim zone | in interior | worst t (Myr) |")
print("|---|---|---|---|---|---|---|---|")
for r in runs:
    if r["dt_myr"] == REF_MYR:
        continue
    d = r["profiles"] - ref["profiles"]
    per_snap = np.max(np.abs(d), axis=(1, 2))
    worst = int(np.argmax(per_snap))
    print(f"| {r['dt_myr']:g} | {r['n_steps']} | {r['seconds']:.1f} | "
          f"{per_snap.max():.3e} | {np.sqrt(np.mean(d**2)):.3e} | "
          f"{np.max(np.abs(d[:, :, rim_zone])):.3e} | "
          f"{np.max(np.abs(d[:, :, interior])):.3e} | {SNAPSHOT_MYR[worst]:g} |")

print()
print("Max error per snapshot time:")
header = [f"{t:g}" for t in SNAPSHOT_MYR]
print("| dt (Myr) | " + " | ".join(header) + " |")
print("|---" * (len(header) + 1) + "|")
for r in runs:
    if r["dt_myr"] == REF_MYR:
        continue
    d = np.max(np.abs(r["profiles"] - ref["profiles"]), axis=(1, 2))
    print(f"| {r['dt_myr']:g} | " + " | ".join(f"{v:.2e}" for v in d) + " |")

print()
print("Per-component max abs error in the rim zone:")
print("| dt (Myr) | " + " | ".join(COMPONENTS) + " |")
print("|---" * (len(COMPONENTS) + 1) + "|")
for r in runs:
    if r["dt_myr"] == REF_MYR:
        continue
    d = np.abs(r["profiles"] - ref["profiles"])[:, :, rim_zone]
    print(f"| {r['dt_myr']:g} | " + " | ".join(f"{v:.3e}" for v in d.max(axis=(0, 2))) + " |")

print()
print("mass at end (per component; should be identical across dt):")
for r in runs:
    print(f"  dt={r['dt_myr']:g} Myr: {r['mass']}")

# %%
out = Path(__file__).resolve().parent / "output"
out.mkdir(exist_ok=True)
serial = [{**r, "profiles": None} for r in runs]
(out / "retrograde_rim_dt_study.json").write_text(json.dumps(serial, indent=2, default=float))
print()
print(f"wrote {out / 'retrograde_rim_dt_study.json'}")
