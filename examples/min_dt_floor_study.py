# %%
"""Which `min_dt` floor resolves the features you care about?

A step initial profile diffusing along a representative PTt path, compared at
several constant timesteps against a fine Backward-Euler reference.  The
question is not "is big dt accurate" in the abstract, but "how much error does
this particular floor introduce on this path".

Path (supplied): P in kbar, T in degC, t in Myr.

    P = [1.0, 14.0, 10.0, 8.0, 6.5]
    T = [600, 740, 780, 700, 500]
    t = [0, 20, 50, 100, 150]

The feature is the smearing of the step.  Error is therefore reported both
globally and restricted to a band around the initial step position, so a
globally small number cannot hide a locally destroyed front.

Reference: the same solver at dt = 0.005 Myr, itself checked against 0.0025.
Both are Backward Euler, so the only difference is timestep.

Run from the repository root in the UW3 environment:

    python examples/min_dt_floor_study.py
"""

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from UWDiffusion.fd_1d import MulticomponentDiffusion1D, lasaga_reduced_diffusivity
from UWDiffusion.tracer_diffusivities import CG92

SEC_PER_MYR = 1.0e6 * 365.25 * 24.0 * 3600.0

# %%
# PTt path as supplied: kbar, degC, Myr.
P_KBAR = np.array([1.0, 14.0, 10.0, 8.0, 6.5])
T_DEGC = np.array([600.0, 740.0, 780.0, 700.0, 500.0])
T_MYR = np.array([0.0, 20.0, 50.0, 100.0, 150.0])

P_interp = interp1d(T_MYR, P_KBAR * 1e-4, fill_value="extrapolate")  # -> GPa
T_interp = interp1d(T_MYR, T_DEGC + 273.15, fill_value="extrapolate")  # -> K

DOMAIN_UM = 100.0  # planar domain width
STEP_UM = 50.0  # initial step position
N_POINTS = 201

COMPONENTS = ["Fe", "Mg", "Mn", "Ca"]
DEPENDENT = "Ca"

# Step: core -> rim.
CORE = {"Fe": 0.30, "Mg": 0.25, "Mn": 0.02}
RIM = {"Fe": 0.45, "Mg": 0.15, "Mn": 0.05}

# Timesteps to test (Myr).  The first three are the requested floors.
FLOORS_MYR = [1.0, 0.1, 0.05]
EXTRA_MYR = [0.01]  # extra context
REF_MYR = 0.005
REF_CHECK_MYR = 0.0025


# %%
def tracer_diffusivities_m2s(temp_K, pressure_GPa):
    """CG92 tracer diffusivities (m^2/s) for [Fe, Mg, Mn, Ca]."""
    return np.array([
        CG92["Fe"].at(temp_K, pressure_GPa),
        CG92["Mg"].at(temp_K, pressure_GPa),
        CG92["Mn"].at(temp_K, pressure_GPa),
        CG92["Mn"].at(temp_K, pressure_GPa) / 20.0,
    ])


# T and P depend only on model time, not position, but the solver re-evaluates
# the diffusivity at every face on every step.  CG92 .at() costs ~62 us and is
# called 4x per evaluation, which dominated the whole run (~90 ms/step).
# Cache the tracer vector on the time value: one lookup per step instead of O(N).
_TRACER_CACHE = {}


def reduced_matrix(t, X):
    """Reduced 3x3 interdiffusion matrix (m^2/s) at model time t (s)."""
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
# Path summary and the diffusion numbers that decide what "resolved" means.

print("PTt path")
print(f"  T: {T_DEGC.min():.0f} - {T_DEGC.max():.0f} degC   P: {P_KBAR.min():.1f} - {P_KBAR.max():.1f} kbar")
print(f"  t: {T_MYR.min():.0f} - {T_MYR.max():.0f} Myr")
print()
print("Diffusion timescale tau = l^2 / D along the path (using D_Mn, the fastest tracer):")
print(f"{'t (Myr)':>8} {'T (degC)':>9} {'D_Mn (m^2/s)':>13} "
      f"{'tau 5um':>10} {'tau 25um':>10} {'tau 50um':>10}")
for t_myr in (0, 20, 35, 50, 75, 100, 125, 150):
    T = float(T_interp(t_myr))
    P = float(P_interp(t_myr))
    d = CG92["Mn"].at(T, P)
    taus = [(l * 1e-6) ** 2 / d / SEC_PER_MYR for l in (5.0, 25.0, 50.0)]
    print(f"{t_myr:8.0f} {T - 273.15:9.0f} {d:13.3e} "
          + " ".join(f"{x:10.3e}" for x in taus))

# %%
# Build and run one model at a given dt.


def make_model():
    x_grid = np.linspace(0.0, DOMAIN_UM * 1e-6, N_POINTS)
    x = x_grid / (DOMAIN_UM * 1e-6)
    left = x < (STEP_UM / DOMAIN_UM)
    model = MulticomponentDiffusion1D(
        grid=x_grid,
        diffusivity=reduced_matrix,
        geometry="planar",
        components=COMPONENTS,
        dependent_component=DEPENDENT,
        initial_composition={
            "Fe": np.where(left, CORE["Fe"], RIM["Fe"]),
            "Mg": np.where(left, CORE["Mg"], RIM["Mg"]),
            "Mn": np.where(left, CORE["Mn"], RIM["Mn"]),
        },
    )
    return model


# Times at which to compare profiles.  Comparing only the final state is
# useless here: the profile homogenises early, so every dt lands on the same
# flat answer and the errors collapse to roundoff.  The question is whether a
# coarse dt reproduces the *transient*, so we must look while it is happening.
SNAPSHOT_MYR = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0]


def run_at(dt_myr):
    t0 = time.perf_counter()
    model = make_model()
    model.reset(0.0)
    snaps = []
    for t_myr in SNAPSHOT_MYR:
        model.run_to_time(t_myr * SEC_PER_MYR, dt_myr * SEC_PER_MYR)
        snaps.append(np.array([model.get_composition(c) for c in COMPONENTS]))
    out = {"dt_myr": dt_myr, "profiles": np.stack(snaps),  # (n_snap, n_comp, n_pts)
           "snapshots_myr": list(SNAPSHOT_MYR),
           "seconds": time.perf_counter() - t0,
           "n_steps": int(round(T_MYR[-1] / dt_myr))}
    mass = model.total_mass()
    out["mass"] = {k: float(v) for k, v in mass.items()} if isinstance(mass, dict) else float(mass)
    return out


# %%
print()
print("running reference (this is the slow one) ...")
ref = run_at(REF_MYR)
print(f"  dt={REF_MYR} Myr: {ref['seconds']:.1f}s")
ref_check = run_at(REF_CHECK_MYR)
ref_vs_check = float(np.max(np.abs(ref["profiles"] - ref_check["profiles"])))
print(f"  dt={REF_CHECK_MYR} Myr vs dt={REF_MYR}: max diff {ref_vs_check:.3e}")
print(f"  -> reference self-consistency (should be << the errors we report)")
print()

runs = [ref]
for dt in FLOORS_MYR + EXTRA_MYR:
    r = run_at(dt)
    runs.append(r)
    print(f"  dt={dt} Myr: {r['seconds']:.1f}s")

# %%
# Error metrics: global, and restricted to a band around the initial step.

x_grid = np.linspace(0.0, DOMAIN_UM * 1e-6, N_POINTS)
x_um = x_grid * 1e6
front_band = np.abs(x_um - STEP_UM) <= 10.0  # +-10 um of the initial step
elsewhere = ~front_band

print()
print("### Error vs the fine reference  (max over all snapshot times)")
print("| dt (Myr) | steps | wall_s | max abs | RMS | worst snapshot (Myr) | near step | elsewhere |")
print("|---|---|---|---|---|---|---|---|")
for r in runs:
    if r["dt_myr"] == REF_MYR:
        continue
    d = r["profiles"] - ref["profiles"]          # (n_snap, n_comp, n_pts)
    per_snap = np.max(np.abs(d), axis=(1, 2))
    worst = int(np.argmax(per_snap))
    print(f"| {r['dt_myr']:g} | {r['n_steps']} | {r['seconds']:.1f} | "
          f"{per_snap.max():.3e} | {np.sqrt(np.mean(d**2)):.3e} | "
          f"{SNAPSHOT_MYR[worst]:g} | "
          f"{np.max(np.abs(d[:, :, front_band])):.3e} | "
          f"{np.max(np.abs(d[:, :, elsewhere])):.3e} |")

print()
print("Max error per snapshot time (where does the coarse run actually go wrong?):")
header = [f"{t:g}" for t in SNAPSHOT_MYR]
print("| dt (Myr) | " + " | ".join(header) + " |")
print("|---" * (len(header) + 1) + "|")
for r in runs:
    if r["dt_myr"] == REF_MYR:
        continue
    d = np.max(np.abs(r["profiles"] - ref["profiles"]), axis=(1, 2))
    print(f"| {r['dt_myr']:g} | " + " | ".join(f"{v:.2e}" for v in d) + " |")

print()
print("Per-component max abs error at the step (max over all snapshots):")
print("| dt (Myr) | " + " | ".join(COMPONENTS) + " |")
print("|---" * (len(COMPONENTS) + 1) + "|")
for r in runs:
    if r["dt_myr"] == REF_MYR:
        continue
    d = np.abs(r["profiles"] - ref["profiles"])[:, :, front_band]
    print(f"| {r['dt_myr']:g} | " + " | ".join(f"{v:.3e}" for v in d.max(axis=(0, 2))) + " |")

# %%
# Mass conservation is the sanity check: a drifting mass is a bug, not a result.

print()
print("mass at end of run (per component; should be identical across all dt):")
for r in runs:
    m = r["mass"]
    shown = {k: f"{v:.8e}" for k, v in m.items()} if isinstance(m, dict) else f"{m:.8e}"
    print(f"  dt={r['dt_myr']:g} Myr: {shown}")

# %%
out = Path(__file__).resolve().parent / "output"
out.mkdir(exist_ok=True)
serial = [{k: v for k, v in r.items()} for r in runs]
for s in serial:
    s["profiles"] = None  # omit the bulk arrays
(out / "min_dt_floor_study.json").write_text(json.dumps(serial, indent=2, default=float))
print()
print(f"wrote {out / 'min_dt_floor_study.json'}")
