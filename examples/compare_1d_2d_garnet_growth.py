# %%
"""Synthetic garnet growth + diffusion: 1D cylindrical FV vs 2D UW3 FEM.

The companion to ``compare_1d_2d_cylindrical_multicomponent.py``, which tests
the *diffusion* operator alone on a smooth eigenmode. This script tests the
rest of the garnet workflow -- concentric growth, interior Dirichlet pins on
not-yet-grown shells, a rim composition that changes at every growth step, and
a retrograde overprint -- against the same synthetic history in both solvers.

Everything is analytic and deliberately small so the whole comparison runs in
seconds. The history is synthetic, not a real rock:

* full disk ``r in [0, 1]``, 17 radial samples, 4 growth shells
* three independent components Fe, Mg, Mn with Ca by closure
* constant tracer diffusivities, reduced through the Lasaga mean-field matrix, so
  the effective ``D`` is genuinely composition dependent
* rim composition drifting linearly through growth, then a retrograde step

Both solvers use the identical growth radii, growth times, rim compositions,
pinning rule and Backward Euler timestep.  The pinning rule: at growth step
``j`` every shell at or beyond the growth front is held at the step
composition, and only the already-grown interior is free to diffuse.

Run in the documented UW3 environment from the repository root:
    python examples/compare_1d_2d_garnet_growth.py
"""

# %%
import sys
from pathlib import Path

import numpy as np
import sympy as sp

import underworld3 as uw

# Allow direct execution from a source checkout without requiring an editable
# UWDiffusion install in the activated Underworld environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from UWDiffusion import (
    MulticomponentDiffusion1D,
    MulticomponentDiffusionModel,
    lasaga_matrix,
    lasaga_reduced_diffusivity,
)
from UWDiffusion.meshing import generate_garnet_annulus_mesh

# %% [markdown]
# Synthetic history
# -----------------
# Rim composition drifts linearly through the prograde growth and then jumps
# for the retrograde overprint. Ca follows from closure.

# %%
COMPONENTS = ["Fe", "Mg", "Mn", "Ca"]
INDEPENDENT = COMPONENTS[:3]
N_TOTAL = 17                 # radial samples in the 1D grid / shell radii
N_GROWTH = 4                 # prograde growth steps
R_OUTER = 1.0
FINAL_TIME = 0.02
DT = 0.002
N_COMMON_RADII = 81
N_ANGLES = 32

# Constant tracer diffusivities; the Lasaga mean field makes D composition
# dependent. Deliberately unequal so cross terms are non-zero.
D_TRACER = np.array([1.0, 1.2, 0.8, 0.4])

# Growth shells occupy the outer N_GROWTH samples of the radial grid, matching
# what generate_garnet_annulus_mesh labels as shell_0..shell_{N_GROWTH-1}.
r_grid = np.linspace(0.0, R_OUTER, N_TOTAL)
growth_radii = r_grid[-N_GROWTH:]

# Growth times: evenly spaced through the first 60% of the run, then a
# retrograde interval with the rim held at the final retrograde composition.
t_growth = np.linspace(0.0, 0.6 * FINAL_TIME, N_GROWTH + 1)
t_retro_end = FINAL_TIME


def rim_composition(t: float) -> np.ndarray:
    """Return [Fe, Mg, Mn, Ca] at the rim for time ``t``."""
    s = t / FINAL_TIME
    if s <= 0.6:
        # Prograde drift.
        frac = s / 0.6
        fe = 0.60 - 0.20 * frac
        mg = 0.15 + 0.10 * frac
        mn = 0.05 + 0.05 * frac
    else:
        # Retrograde overprint: rim resets and Fe climbs back.
        fe = 0.45
        mg = 0.22
        mn = 0.12
    return np.array([fe, mg, mn, 1.0 - fe - mg - mn])


growth_comp = np.array([rim_composition(t) for t in t_growth])
retro_comp = rim_composition(FINAL_TIME)

# %%
# 1D cylindrical FV
# -----------------
# Growth loop: stamp the still-future shells, pin every shell at or beyond
# the growth front, fix the outer rim, then diffuse to the next growth time.


def make_diffusivity():
    """Lasaga-reduced D(state) matching the symbolic matrix used in 2D."""

    def D(state):
        X = np.array([float(state[c]) for c in COMPONENTS])
        return lasaga_reduced_diffusivity(D_TRACER, X, dependent_component=3)

    return D


model_1d = MulticomponentDiffusion1D(
    grid=r_grid,
    diffusivity=make_diffusivity(),
    geometry="cylindrical",
    components=COMPONENTS,
    dependent_component="Ca",
    bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    centre="zero_flux",
    initial_composition={
        "Fe": growth_comp[0, 0], "Mg": growth_comp[0, 1], "Mn": growth_comp[0, 2]
    },
)


def apply_growth_step_1d(step_idx: int) -> None:
    comp = {
        "Fe": float(growth_comp[step_idx, 0]),
        "Mg": float(growth_comp[step_idx, 1]),
        "Mn": float(growth_comp[step_idx, 2]),
    }
    model_1d.apply_growth(growth_radii[step_idx], comp, set_rim_bc=False)
    future = model_1d.r >= growth_radii[step_idx]
    future[-1] = False
    model_1d.set_internal_bcs(future, comp)
    model_1d.set_bc("right", "dirichlet", comp)


model_1d.reset(t_growth[0])
for i in range(N_GROWTH):
    apply_growth_step_1d(i)
    model_1d.run_to_time(t_growth[i + 1], DT)

# Grain complete at t_growth[-1]: drop the interior pins so the whole profile
# can diffuse, and reset the rim to the retrograde composition.
model_1d.clear_internal_bcs()
retro = {
    "Fe": float(retro_comp[0]), "Mg": float(retro_comp[1]), "Mn": float(retro_comp[2])
}
model_1d.set_bc("right", "dirichlet", retro)
model_1d.run_to_time(t_retro_end, DT)

# %%
# 2D coupled UW3 model on a full disk
# -----------------------------------
# Same mean-field matrix, built symbolically exactly as the 2D model builds
# it. The mesh generator lays concentric shells through the radial grid and
# labels the outermost ``N_GROWTH`` of them ``shell_0..shell_{N_GROWTH-1}``.

d_star = sp.symbols("D_Fe D_Mg D_Mn D_Ca")
xi = sp.symbols("xi_1 xi_2 xi_3 xi_4")
D_matrix = lasaga_matrix(list(d_star), list(xi))

mesh_2d = generate_garnet_annulus_mesh(
    radii_nd=r_grid,
    mesh_degree=2,
    total_shell_count=N_TOTAL,
    growth_shell_count=N_GROWTH,
    output_path=Path(__file__).resolve().parent / "output" / "garnet_growth_mesh",
)
model_2d = MulticomponentDiffusionModel(
    COMPONENTS, mesh_2d, diffusion_matrix=D_matrix, degree=2, order=1
)
model_2d.set_diffusion_values(dict(zip(["D_Fe", "D_Mg", "D_Mn", "D_Ca"], D_TRACER)))
for solver in model_2d.solvers:
    solver.petsc_options["snes_rtol"] = 1e-12
    solver.petsc_options["snes_atol"] = 1e-8

coords_2d = model_2d.mesh_vars[0].coords
r_dof = np.sqrt(coords_2d[:, 0] ** 2 + coords_2d[:, 1] ** 2)
vars_2d = list(model_2d.mesh_vars)


def stamp_2d(mask: np.ndarray, comp: np.ndarray) -> None:
    with mesh_2d.access(*vars_2d):
        for k in range(3):
            vars_2d[k].data[mask, 0] = comp[k]
        total = np.zeros_like(vars_2d[-1].data[:, 0])
        for k in range(3):
            total += vars_2d[k].data[:, 0]
        vars_2d[-1].data[:, 0] = 1.0 - total


def apply_growth_step_2d(step_idx: int) -> None:
    comp = growth_comp[step_idx]
    # Stamp the still-future region with the step composition, matching the
    # 1D ``apply_growth`` convention (r strictly beyond the front).
    stamp_2d(r_dof > growth_radii[step_idx], comp)

    # Pin every shell at or beyond the front -- the 2D counterpart of the 1D
    # internal pins plus the rim Dirichlet.
    for solver in model_2d.solvers:
        solver.essential_bcs = []
    for boundary in mesh_2d.boundaries:
        name = boundary.name
        if not name.startswith("shell_"):
            continue
        shell_idx = int(name.split("_")[1])
        if shell_idx >= step_idx:
            for k in range(3):
                model_2d.add_dirichlet_bc({name: float(comp[k])}, component_index=k)
    # The manual field overwrite must be reflected in the time-derivative
    # history, or the next step sees a spurious dC/dt.
    model_2d.sync_history_to_state()


with mesh_2d.access(*vars_2d):
    for k in range(3):
        vars_2d[k].data[:, 0] = growth_comp[0, k]
    total = np.zeros_like(vars_2d[-1].data[:, 0])
    for k in range(3):
        total += vars_2d[k].data[:, 0]
    vars_2d[-1].data[:, 0] = 1.0 - total

for i in range(N_GROWTH):
    apply_growth_step_2d(i)
    # ``duration`` is an absolute target time, not an increment.
    model_2d.run_simulation(duration=t_growth[i + 1], min_dt=DT, max_dt=DT)

# Grain complete at t_growth[-1]: only the outermost shell is still pinned, so
# rebuilding the BCs at the retrograde composition frees the whole interior.
for solver in model_2d.solvers:
    solver.essential_bcs = []
for k in range(3):
    model_2d.add_dirichlet_bc(
        {"shell_%d" % (N_GROWTH - 1): float(retro_comp[k])}, component_index=k
    )
model_2d.run_simulation(duration=t_retro_end, min_dt=DT, max_dt=DT)

# %%
# Compare on common radii
# -----------------------
# Sample the 2D field on a ring of angles and circumferentially average, which
# removes the polygonal-mesh anisotropy from the comparison.

r_common = np.linspace(1e-10, R_OUTER - 1e-10, N_COMMON_RADII)
angles = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
sample_xy = np.column_stack(
    (
        (r_common[:, None] * np.cos(angles)).ravel(),
        (r_common[:, None] * np.sin(angles)).ravel(),
    )
)
profiles_2d_samples = np.stack(
    [
        uw.function.evaluate(var.sym[0], sample_xy).reshape(N_COMMON_RADII, N_ANGLES)
        for var in vars_2d
    ]
)
profiles_2d = np.mean(profiles_2d_samples, axis=2)
angular_spread = np.sqrt(
    np.mean((profiles_2d_samples - profiles_2d[:, :, None]) ** 2, axis=2)
)
profiles_1d = np.vstack(
    [
        np.interp(r_common, model_1d.r, model_1d.get_composition(name))
        for name in COMPONENTS
    ]
)

diff = profiles_1d - profiles_2d
max_diff = np.max(np.abs(diff), axis=1)
rmse = lambda v: float(np.sqrt(np.mean(v**2)))

print("Synthetic garnet growth + diffusion: 1D cylindrical FV / 2D disk FEM")
print(f"  components: {COMPONENTS} (Ca closure); tracer D = {D_TRACER.tolist()}")
print(f"  disk r in [0, {R_OUTER:g}]; {N_TOTAL} radial samples; {N_GROWTH} growth shells")
print(f"  growth times: {np.round(t_growth, 4).tolist()}; retrograde ends {t_retro_end:g}")
print(f"  time: {FINAL_TIME:g}, dt: {DT:g} (both, Backward Euler)")
print(f"  1D vertices: {model_1d.r.size} (dr={r_grid[1]-r_grid[0]:.4g}); 2D shells at dr={r_grid[1]-r_grid[0]:.4g}, Q2")
print(f"  common samples: {N_COMMON_RADII} radii x {N_ANGLES} angles")
for idx, name in enumerate(COMPONENTS):
    j = int(np.argmax(np.abs(diff[idx])))
    print(
        f"  {name}: 1D vs 2D max|difference|={max_diff[idx]:.6e} at r={r_common[j]:.3f}, "
        f"RMSE={rmse(diff[idx]):.6e}"
    )
print(f"  growth-front radii: {np.round(growth_radii, 3).tolist()}")
print(f"  2D circumferential RMS spread max: {np.max(angular_spread, axis=1)}")
print(f"  final times: {model_1d.time:.12g} (1D), {model_2d.current_time:.12g} (2D)")

# %%
# Structural assertions. The tolerance on the profile difference is loose
# enough to absorb FV/FEM discretisation and the sharp compositional steps at
# each growth front, but tight enough to catch a wrong pinning rule, a wrong
# shell index, a missing history sync, or a mismatched mean-field matrix.
assert np.allclose(model_1d.time, FINAL_TIME, rtol=0.0, atol=1e-12)
assert np.allclose(model_2d.current_time, FINAL_TIME, rtol=0.0, atol=1e-12)
assert np.max(np.abs(profiles_1d.sum(axis=0) - 1.0)) < 1e-12
assert np.max(np.abs(profiles_2d.sum(axis=0) - 1.0)) < 1e-12
assert np.max(angular_spread) < 1e-2, np.max(angular_spread)
assert np.all(max_diff < 5e-3), max_diff
