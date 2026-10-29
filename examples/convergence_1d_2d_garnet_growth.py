# %%
"""Mesh and timestep convergence of the synthetic garnet growth comparison.

Companion to ``compare_1d_2d_garnet_growth.py``. That script reports one
number at one resolution; this one tells you what the number means:

* how the 1D/2D discrepancy scales with mesh spacing (the spatial rate),
* how much of it is timestep error rather than mesh error,
* how much of it is the *comparison* rather than either solver -- the 1D
  unknown is a control-volume value and the 2D field is a point value on a
  Q2 basis, and near a sharp compositional step those are not the same
  object.

The discrepancy is therefore split three ways:

``err_point``
    max |1D - 2D| over the whole profile, point sampled. This is the
    headline number from the comparison script.
``err_front``
    the same, restricted to a band of half-width ``dr`` around each growth
    radius and the rim -- where the compositional steps are.
``err_interior``
    the same, restricted to everything outside those bands.
``err_cv``
    max |control-volume-averaged 2D - 1D| over the whole profile. Both
    sides are now region averages, so the interpolation artefact at the
    step is removed.

Known confound: the growth shells are the outermost ``N_GROWTH`` grid
vertices, so their radii move slightly as the mesh is refined. The
*problem class* is identical at every resolution (same growth times, same
rim history, four shells in the outer quarter of the grain) but the shell
positions are a discretisation parameter here, not physics. The fitted rate
should be read as an upper bound on the true spatial rate for a fixed
interface set.

Run in the documented UW3 environment from the repository root:
    python examples/convergence_1d_2d_garnet_growth.py
"""

# %%
from pathlib import Path

import numpy as np
import sympy as sp

import underworld3 as uw

from UWDiffusion import (
    MulticomponentDiffusion1D,
    MulticomponentDiffusionModel,
    lasaga_matrix,
    lasaga_reduced_diffusivity,
)
from UWDiffusion.meshing import generate_garnet_annulus_mesh

# %% [markdown]
# Shared synthetic history. Identical to the comparison script: fixed growth
# timing and rim compositions, only the discretisation varies.

# %%
COMPONENTS = ["Fe", "Mg", "Mn", "Ca"]
N_GROWTH = 4
R_OUTER = 1.0
FINAL_TIME = 0.02
DT_DEFAULT = 0.002
N_ANGLES = 24
N_SAMPLE = 401

D_TRACER = np.array([1.0, 1.2, 0.8, 0.4])
t_growth = np.linspace(0.0, 0.6 * FINAL_TIME, N_GROWTH + 1)
t_retro_end = FINAL_TIME


def rim_composition(t: float) -> np.ndarray:
    s = t / FINAL_TIME
    if s <= 0.6:
        frac = s / 0.6
        fe = 0.60 - 0.20 * frac
        mg = 0.15 + 0.10 * frac
        mn = 0.05 + 0.05 * frac
    else:
        fe, mg, mn = 0.45, 0.22, 0.12
    return np.array([fe, mg, mn, 1.0 - fe - mg - mn])


growth_comp = np.array([rim_composition(t) for t in t_growth])
retro_comp = rim_composition(FINAL_TIME)

d_star = sp.symbols("D_Fe D_Mg D_Mn D_Ca")
xi = sp.symbols("xi_1 xi_2 xi_3 xi_4")
D_MATRIX = lasaga_matrix(list(d_star), list(xi))


# %%
# One coupled 1D / 2D run at a given discretisation.
# -----------------------------------------------


def run_case(n_total: int, dt: float = DT_DEFAULT, tag: str = ""):
    """Run the synthetic growth history in both solvers and return profiles."""
    r_grid = np.linspace(0.0, R_OUTER, n_total)
    growth_radii = r_grid[-N_GROWTH:]

    def D(state):
        X = np.array([float(state[c]) for c in COMPONENTS])
        return lasaga_reduced_diffusivity(D_TRACER, X, dependent_component=3)

    model_1d = MulticomponentDiffusion1D(
        grid=r_grid,
        diffusivity=D,
        geometry="cylindrical",
        components=COMPONENTS,
        dependent_component="Ca",
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
        centre="zero_flux",
        initial_composition={
            "Fe": growth_comp[0, 0], "Mg": growth_comp[0, 1], "Mn": growth_comp[0, 2]
        },
    )

    def grow_1d(j):
        comp = {
            "Fe": float(growth_comp[j, 0]),
            "Mg": float(growth_comp[j, 1]),
            "Mn": float(growth_comp[j, 2]),
        }
        model_1d.apply_growth(growth_radii[j], comp, set_rim_bc=False)
        future = model_1d.r >= growth_radii[j]
        future[-1] = False
        model_1d.set_internal_bcs(future, comp)
        model_1d.set_bc("right", "dirichlet", comp)

    model_1d.reset(t_growth[0])
    for j in range(N_GROWTH):
        grow_1d(j)
        model_1d.run_to_time(t_growth[j + 1], dt)
    model_1d.clear_internal_bcs()
    model_1d.set_bc(
        "right",
        "dirichlet",
        {"Fe": float(retro_comp[0]), "Mg": float(retro_comp[1]), "Mn": float(retro_comp[2])},
    )
    model_1d.run_to_time(t_retro_end, dt)

    mesh = generate_garnet_annulus_mesh(
        radii_nd=r_grid,
        mesh_degree=2,
        total_shell_count=n_total,
        growth_shell_count=N_GROWTH,
        output_path=Path(__file__).resolve().parent / "output" / "conv_garnet_mesh",
        mesh_name=f"garnet_conv_{tag or n_total}",
    )
    model_2d = MulticomponentDiffusionModel(
        COMPONENTS, mesh, diffusion_matrix=D_MATRIX, degree=2, order=1
    )
    model_2d.set_diffusion_values(dict(zip(["D_Fe", "D_Mg", "D_Mn", "D_Ca"], D_TRACER)))
    for solver in model_2d.solvers:
        solver.petsc_options["snes_rtol"] = 1e-12
        solver.petsc_options["snes_atol"] = 1e-8

    coords = model_2d.mesh_vars[0].coords
    r_dof = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    vars_2d = list(model_2d.mesh_vars)

    def stamp(mask, comp):
        with mesh.access(*vars_2d):
            for k in range(3):
                vars_2d[k].data[mask, 0] = comp[k]
            total = np.zeros_like(vars_2d[-1].data[:, 0])
            for k in range(3):
                total += vars_2d[k].data[:, 0]
            vars_2d[-1].data[:, 0] = 1.0 - total

    def grow_2d(j):
        comp = growth_comp[j]
        stamp(r_dof > growth_radii[j], comp)
        for solver in model_2d.solvers:
            solver.essential_bcs = []
        for boundary in mesh.boundaries:
            name = boundary.name
            if not name.startswith("shell_"):
                continue
            if int(name.split("_")[1]) >= j:
                for k in range(3):
                    model_2d.add_dirichlet_bc({name: float(comp[k])}, component_index=k)
        model_2d.sync_history_to_state()

    with mesh.access(*vars_2d):
        for k in range(3):
            vars_2d[k].data[:, 0] = growth_comp[0, k]
        total = np.zeros_like(vars_2d[-1].data[:, 0])
        for k in range(3):
            total += vars_2d[k].data[:, 0]
        vars_2d[-1].data[:, 0] = 1.0 - total

    for j in range(N_GROWTH):
        grow_2d(j)
        model_2d.run_simulation(duration=t_growth[j + 1], min_dt=dt, max_dt=dt)

    for solver in model_2d.solvers:
        solver.essential_bcs = []
    for k in range(3):
        model_2d.add_dirichlet_bc(
            {"shell_%d" % (N_GROWTH - 1): float(retro_comp[k])}, component_index=k
        )
    model_2d.run_simulation(duration=t_retro_end, min_dt=dt, max_dt=dt)

    # Circumferentially averaged 2D radial profile on a fine radial grid.
    r_sample = np.linspace(0.0, R_OUTER, N_SAMPLE)
    angles = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
    xy = np.column_stack(
        (
            (r_sample[:, None] * np.cos(angles)).ravel(),
            (r_sample[:, None] * np.sin(angles)).ravel(),
        )
    )
    samples = np.stack(
        [
            uw.function.evaluate(v.sym[0], xy).reshape(N_SAMPLE, N_ANGLES)
            for v in vars_2d
        ]
    )
    p2d = samples.mean(axis=2)

    return {
        "dr": float(r_grid[1] - r_grid[0]),
        "r_grid": r_grid,
        "growth_radii": growth_radii,
        "r_bounds": model_1d.r_bounds.copy(),
        "p1d": {n: model_1d.get_composition(n) for n in COMPONENTS},
        "r_sample": r_sample,
        "p2d": p2d,
    }


# %%
# Discrepancy metrics
# -------------------


def metrics(case) -> dict:
    dr = case["dr"]
    r_grid, r_b = case["r_grid"], case["r_bounds"]
    r_s, p2d = case["r_sample"], case["p2d"]

    # Band of half-width dr around each growth radius and the rim.
    steps = np.concatenate([case["growth_radii"], [R_OUTER]])
    dist = np.min(np.abs(r_s[:, None] - steps[None, :]), axis=1)
    front_mask = dist <= dr
    interior_mask = ~front_mask

    out = {}
    for k, name in enumerate(COMPONENTS):
        p1d_s = np.interp(r_s, r_grid, case["p1d"][name])
        delta = p1d_s - p2d[k]
        # Control-volume average of the 2D profile over each 1D CV.
        cv2d = np.empty(len(r_b) - 1)
        for m in range(len(r_b) - 1):
            rr = np.linspace(r_b[m], r_b[m + 1], 41)
            w = rr**1  # cylindrical weighting, alpha = 1
            f = np.interp(rr, r_s, p2d[k])
            cv2d[m] = float(np.sum(f * w) / np.sum(w))
        out[name] = {
            "point": float(np.max(np.abs(delta))),
            "front": float(np.max(np.abs(delta[front_mask]))),
            "interior": float(np.max(np.abs(delta[interior_mask]))),
            "cv": float(np.max(np.abs(cv2d - case["p1d"][name]))),
        }
    return out


def rate(dr_values, errors) -> float:
    """Least-squares slope of log(error) against log(dr)."""
    return float(np.polyfit(np.log(dr_values), np.log(errors), 1)[0])


# %% [markdown]
# Mesh sweep at fixed timestep.

# %%
MESH_TOTALS = [17, 25, 33, 41]
mesh_runs = []
for n in MESH_TOTALS:
    case = run_case(n, DT_DEFAULT, tag=f"m{n}")
    mesh_runs.append((case, metrics(case)))
    print(
        f"[mesh] n_total={n:3d} dr={case['dr']:.5f} "
        f"fronts={np.round(case['growth_radii'], 4).tolist()}",
        flush=True,
    )

# %% [markdown]
# Timestep sweep at fixed mesh, to bound the temporal error.

# %%
DT_VALUES = [0.004, 0.002, 0.001]
N_DT_MESH = 25
dt_runs = []
for dt in DT_VALUES:
    case = run_case(N_DT_MESH, dt, tag=f"dt{dt}")
    dt_runs.append((case, metrics(case)))
    print(f"[dt] n_total={N_DT_MESH} dt={dt:g}", flush=True)

# %%
# Report
# ------

print()
print("Synthetic garnet growth: convergence of the 1D cylindrical FV / 2D FEM gap")
print(f"  components {COMPONENTS} (Ca closure); tracer D = {D_TRACER.tolist()}")
print(f"  N_GROWTH={N_GROWTH}, FINAL_TIME={FINAL_TIME:g}, Backward Euler both sides")
print()
print("  Mesh sweep (dt fixed at %g):" % DT_DEFAULT)
print(f"  {'n':>4} {'dr':>9} {'point':>11} {'front':>11} {'interior':>11} {'cv':>11}")
for case, met in mesh_runs:
    row = np.array([[met[n][k] for k in ("point", "front", "interior", "cv")] for n in COMPONENTS])
    peak = row.max(axis=0)
    print(
        f"  {len(case['r_grid']):>4} {case['dr']:>9.5f} "
        f"{peak[0]:>11.4e} {peak[1]:>11.4e} {peak[2]:>11.4e} {peak[3]:>11.4e}"
    )

drs = [c["dr"] for c, _ in mesh_runs]
peak_by_key = {
    key: np.array([
        max(met[n][key] for n in COMPONENTS) for _, met in mesh_runs
    ])
    for key in ("point", "front", "interior", "cv")
}
print()
print("  Fitted spatial rates p in error ~ dr^p:")
for key in ("point", "front", "interior", "cv"):
    print(f"    {key:<9} p = {rate(drs, peak_by_key[key]):+.3f}")
print("    (the 'front' band is one element wide, so it shrinks with dr;")
print("     'front' is the error confined to one element around each step)")

print()
print("  Timestep sweep (n_total = %d):" % N_DT_MESH)
print(f"  {'dt':>9} {'point':>11} {'front':>11} {'interior':>11} {'cv':>11}")
for dt_value, (case, met) in zip(DT_VALUES, dt_runs):
    row = np.array([[met[n][k] for k in ("point", "front", "interior", "cv")] for n in COMPONENTS])
    peak = row.max(axis=0)
    print(
        f"  {dt_value:>9g} "
        f"{peak[0]:>11.4e} {peak[1]:>11.4e} {peak[2]:>11.4e} {peak[3]:>11.4e}"
    )
dts = DT_VALUES
peak_dt = {
    key: np.array([
        max(met[n][key] for n in COMPONENTS) for _, met in dt_runs
    ])
    for key in ("point", "front", "interior", "cv")
}
print()
print("  Fitted temporal rates q in error ~ dt^q:")
for key in ("point", "front", "interior", "cv"):
    print(f"    {key:<9} q = {rate(dts, peak_dt[key]):+.3f}")

coarse, fine = mesh_runs[0][1], mesh_runs[-1][1]
print()
print("  What the headline number is made of (coarsest -> finest mesh):")
for key in ("point", "front", "interior", "cv"):
    a = max(coarse[n][key] for n in COMPONENTS)
    b = max(fine[n][key] for n in COMPONENTS)
    print(f"    {key:<9} {a:.4e} -> {b:.4e}  ({a / b:.1f}x)")
