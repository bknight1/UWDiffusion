# %%
"""Solver tolerance and preconditioner sweep on a NONLINEAR time-dependent solve.

``examples/solver_tolerance_sweep.py`` (the linear variant) showed that a
constant-coefficient operator cannot answer this question at all: Newton
converges in one iteration whatever the tolerance, so there is no knee to find.
This version makes the operator genuinely nonlinear and time-dependent in the
same way the garnet growth models are, so solver settings actually bite.

What makes it representative of the production problem
-----------------------------------------------------
* A Lasaga mean-field matrix via ``lasaga_matrix_from_components``, whose
  entries depend on the composition unknowns through ``xi_1..xi_4`` -- the
  residual is genuinely nonlinear, exactly as in the garnet models.
* Four components with a strongly fractionating tracer set (Mn fast, Ca slow),
  giving the same large non-symmetry the production matrix has.
* Diffusivity re-evaluated every step from a synthetic PTt path, so the system
  is rebuilt each step (the ``pre_solve_diffusivity_only`` pattern).
* A time-varying Dirichlet rim, rebuilt each step (the
  ``update_retrograde_boundary`` pattern).

No analytical reference is used -- the question here is purely which solver
configuration converges robustly and fastest.  The tolerance knee is found by
SELF-CONSISTENCY: the tightest run is the reference and coarser tolerances are
compared against it.

Predictions
-----------
P1  Because the operator is nonlinear, ``snes_its`` is > 1 and grows as the
    tolerance tightens.  (The linear variant reported exactly 1 everywhere.)
P2  ``snes_atol`` has a knee: below it the final state stops changing while
    the iteration count keeps climbing.  That knee is the setting to use.
P3  Inverting the tolerance hierarchy -- a ``ksp_rtol`` LOOSER than
    ``snes_atol`` -- is what makes Newton stall and the line search fail.
    This is the pathology seen in the production granulite run.
P4  All preconditioners agree on the final state (solver settings are not a
    physical knob) but differ in wall-clock.

Run from the repository root in an environment with the UW3/PETSc/MPI stack
available (see AGENTS.md):

    python examples/solver_tolerance_sweep.py

Machine-readable results go to
``examples/output/solver_tolerance_sweep_nonlinear.json`` (gitignored).
"""

# %%
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import underworld3 as uw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from UWDiffusion import MulticomponentDiffusionModel, lasaga_matrix_from_components

# %%
# Geometry and physics.  The tracer set is chosen to reproduce the strong
# non-symmetry of the production matrix: Mn roughly 9x faster than Fe, Ca 20x
# slower than Mn.  That is what makes the operator hard, not the mesh.

R_INNER = 0.5
R_OUTER = 1.0
COMPONENTS = ["Fe", "Mg", "Mn", "Ca"]  # Ca is dependent under closure
N_INDEP = len(COMPONENTS) - 1

# Relative tracer diffusivities, matching the spread of CG92 at garnet
# conditions (D_Mn / D_Ca = 20 by the package convention).
D_BASE = {"Fe": 1.00, "Mg": 0.63, "Mn": 9.41, "Ca": 0.47}

FINAL_TIME = 0.02
TIME_STEP_FACTOR = 0.1  # the library default
DEGREE = 1
ORDER = 1
CELL_SIZE = 0.05

# Pin the step count so every run does the same amount of work; otherwise the
# accuracy heuristic picks dt and the comparison is confounded.
N_STEPS = 24
MAX_DT = FINAL_TIME / N_STEPS
MIN_DT = MAX_DT / 10.0

N_COMMON_RADII = 81
N_ANGLES = 32

# %%
# Nonlinear operator: the Lasaga matrix carries xi_1..xi_4, which
# _composition_substitutions resolves to the unknowns, so the residual depends
# on the solution.  This is the property the linear variant lacked.

D_MATRIX = lasaga_matrix_from_components(COMPONENTS)

# %%
# Smooth non-uniform initial state -- a zoned grain, so the composition
# dependence of D is actually exercised.


def rim_shape(r: np.ndarray) -> np.ndarray:
    """Smooth radial profile in [-1, 1], flat-gradient at both rims."""
    return np.cos(np.pi * (r - R_INNER) / (R_OUTER - R_INNER))


@contextlib.contextmanager
def quiet_stdout():
    """Silence Python and C-level (PETSc printf) stdout during a run."""
    sys.stdout.flush()
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)


# %%
# One nonlinear time-dependent run at a chosen solver configuration.


def run_case(snes_atol: float, pc_type: str, sub_pc_type: str | None = None,
             ksp_rtol_factor: float = 1e-2, label: str = "",
             cell_size: float | None = None, n_steps: int | None = None) -> dict:
    """Run to FINAL_TIME and report solver behaviour.

    ``ksp_rtol_factor`` scales the linear tolerance RELATIVE to ``snes_atol``.
    1e-2 keeps the linear solve tighter than the nonlinear one; values > 1
    invert the hierarchy and reproduce the production pathology.
    ``cell_size`` and ``n_steps`` override the fixed discretisation, for the
    robustness sweep.
    """
    cell_size = CELL_SIZE if cell_size is None else cell_size
    n_steps = N_STEPS if n_steps is None else n_steps
    max_dt = FINAL_TIME / n_steps
    min_dt = max_dt / 10.0
    ksp_rtol = snes_atol * ksp_rtol_factor
    t0 = time.perf_counter()
    mesh = uw.meshing.Annulus(
        radiusOuter=R_OUTER, radiusInner=R_INNER, cellSize=cell_size, qdegree=2
    )
    n_cells = int(mesh.dm.getDepthStratum(0)[1] - mesh.dm.getDepthStratum(0)[0])
    h_eff = float(np.sqrt(np.pi * (R_OUTER**2 - R_INNER**2) / n_cells))

    try:
        model = MulticomponentDiffusionModel(
            COMPONENTS, mesh, diffusion_matrix=D_MATRIX, degree=DEGREE, order=ORDER
        )
    except Exception as exc:
        return {"label": label, "snes_atol": snes_atol, "pc_type": pc_type,
                "sub_pc_type": sub_pc_type, "ksp_rtol": ksp_rtol, "h_eff": h_eff,
                "status": f"FAILED: {type(exc).__name__}: {exc}"}

    # Diffusivities must be set BEFORE run_simulation: the timestep heuristic
    # evaluates the matrix at the top of the step loop, before any pre-solve
    # hook has run.  With unsubstituted D_* symbols it silently returns 0.0,
    # which makes max_time_step infinite and the whole run a single step.
    model.set_diffusion_values(D_BASE)

    for solver in model.solvers:
        solver.petsc_options["snes_rtol"] = snes_atol
        solver.petsc_options["snes_atol"] = snes_atol
        solver.petsc_options["ksp_rtol"] = ksp_rtol
        solver.petsc_options["pc_type"] = pc_type
        if sub_pc_type:
            solver.petsc_options["sub_pc_type"] = sub_pc_type

    # Non-uniform initial state.
    coords = model.mesh_vars[0].coords
    r_nodes = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    shape = rim_shape(r_nodes)
    with mesh.access(*model.mesh_vars):
        model.mesh_vars[0].data[:, 0] = 0.55 + 0.12 * shape   # Fe
        model.mesh_vars[1].data[:, 0] = 0.32 - 0.12 * shape   # Mg
        model.mesh_vars[2].data[:, 0] = 0.030 + 0.01 * shape  # Mn
        model.mesh_vars[3].data[:, 0] = (
            1.0 - model.mesh_vars[0].data[:, 0]
            - model.mesh_vars[1].data[:, 0] - model.mesh_vars[2].data[:, 0]
        )

    # Time-dependent forcing, rebuilt before every solve: the production
    # pattern of pre_solve_diffusivity_only + update_retrograde_boundary.
    counter = {"its": 0, "solves": 0}

    def pre_solve():
        frac = min(model.current_time / FINAL_TIME, 1.0) if FINAL_TIME else 0.0
        scale = 1.0 + 0.8 * frac                      # synthetic PTt path
        model.set_diffusion_values({k: v * scale for k, v in D_BASE.items()})
        model.clear_boundary_conditions()
        rim_fe = 0.55 + 0.10 * np.sin(2.0 * np.pi * frac)
        rim_mg = 0.32 - 0.10 * np.sin(2.0 * np.pi * frac)
        model.add_dirichlet_bc({"Upper": float(rim_fe)}, component_index=0)
        model.add_dirichlet_bc({"Upper": float(rim_mg)}, component_index=1)
        model.add_dirichlet_bc({"Upper": 0.030}, component_index=2)

    def post_solve():
        try:
            counter["its"] += int(model.solver.snes.getIterationNumber())
            counter["solves"] += 1
        except Exception:
            pass

    model.register_pre_solve_hook("forcing", pre_solve)
    model.register_post_solve_hook("count", post_solve)

    t_solve = time.perf_counter()
    try:
        with quiet_stdout():
            # max_dt pins the step count: without it the accuracy heuristic
            # chooses dt and the run length varies with kappa, which would
            # confound the comparison.  Same pattern as the garnet models.
            model.run_simulation(duration=FINAL_TIME, max_dt=max_dt, min_dt=min_dt)
        status = "ok"
    except RuntimeError as exc:
        status = f"DIVERGED: {exc}"
    seconds_solve = time.perf_counter() - t_solve

    n_steps = int(model.step)
    dt_used = FINAL_TIME / n_steps if n_steps else float("nan")

    # Snapshot the final state on a common grid for cross-run comparison.
    r_common = np.linspace(R_INNER + 1e-8, R_OUTER - 1e-8, N_COMMON_RADII)
    angles = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
    sample_xy = np.column_stack(
        ((r_common[:, None] * np.cos(angles)).ravel(),
         (r_common[:, None] * np.sin(angles)).ravel())
    )
    profiles = {}
    for name, var in zip(COMPONENTS, model.mesh_vars):
        s = uw.function.evaluate(var.sym[0], sample_xy).reshape(N_COMMON_RADII, N_ANGLES)
        profiles[name] = np.mean(s, axis=1).tolist()

    out = {
        "label": label, "snes_atol": snes_atol, "ksp_rtol": ksp_rtol,
        "ksp_rtol_factor": ksp_rtol_factor,
        "pc_type": pc_type, "sub_pc_type": sub_pc_type,
        "cell_size": cell_size, "degree": DEGREE, "order": ORDER,
        "h_eff": h_eff, "n_cells": n_cells,
        "dt_used": dt_used, "n_steps": n_steps,
        "snes_its": counter["its"], "snes_solves": counter["solves"],
        "its_per_solve": counter["its"] / counter["solves"] if counter["solves"] else float("nan"),
        "seconds_solve": seconds_solve, "seconds": time.perf_counter() - t0,
        "profiles": profiles, "status": status,
    }
    del model, mesh
    gc.collect()
    return out


def state_delta(a: dict, b: dict) -> float:
    """Max |difference| of the two runs' final profiles, or NaN."""
    if a.get("status") != "ok" or b.get("status") != "ok":
        return float("nan")
    return float(max(np.max(np.abs(np.array(a["profiles"][k]) - np.array(b["profiles"][k])))
                     for k in COMPONENTS))


# %%
# Sweep A -- tolerance, with the CORRECT hierarchy (linear tighter than nonlinear).

SNES_ATOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

print("NONLINEAR solver tolerance sweep: time-dependent Lasaga operator")
print(f"  fixed: cellSize={CELL_SIZE} degree={DEGREE} order={ORDER} "
      f"components={COMPONENTS} D={D_BASE}")
print(f"  sweep A: snes_atol {SNES_ATOLS}, pc='gamg', ksp_rtol = 1e-2 * snes_atol")
print()

t_sweep_start = time.perf_counter()
sweep_a = []
for snes_atol in SNES_ATOLS:
    rec = run_case(snes_atol, pc_type="gamg", label=f"atol={snes_atol:.0e}")
    sweep_a.append(rec)
    if rec["status"] != "ok":
        print(f"  atol={snes_atol:.0e}: {rec['status']}")
    else:
        print(f"  atol={snes_atol:.0e}: its/solve={rec['its_per_solve']:.2f} "
              f"tot_its={rec['snes_its']} solve={rec['seconds_solve']:.1f}s "
              f"steps={rec['n_steps']}")
    sys.stdout.flush()

ok_a = [r for r in sweep_a if r["status"] == "ok"]
ok_a.sort(key=lambda r: r["snes_atol"])

# The tightest successful run is the self-consistency reference.
reference = ok_a[0] if ok_a else None
for r in ok_a:
    r["delta_vs_tightest"] = state_delta(r, reference)

print()
print("self-consistency vs the tightest successful tolerance")
for r in ok_a:
    print(f"  atol={r['snes_atol']:.0e}: delta={r['delta_vs_tightest']:.3e}  "
          f"its/solve={r['its_per_solve']:.2f}  solve={r['seconds_solve']:.1f}s")

TOL_SLACK = 1e-6  # final states must agree to this level to count as unchanged
BEST_ATOL = None
# Coarsest first: the knee is the loosest tolerance whose answer is already
# converged, not the tightest one.
for r in sorted(ok_a, key=lambda r: -r["snes_atol"]):
    if r["delta_vs_tightest"] <= TOL_SLACK:
        BEST_ATOL = r["snes_atol"]
        break

print()
print(f"knee: coarsest snes_atol whose final state matches the tightest run "
      f"to {TOL_SLACK:.0e} -> {BEST_ATOL if BEST_ATOL else 'none found'}")

# %%
# Sweep B -- preconditioner at the knee tolerance.

USE_ATOL = BEST_ATOL if BEST_ATOL is not None else 1e-6
PC_CHOICES = [("gamg", None), ("lu", None), ("bjacobi", "ilu"), ("asm", "ilu")]

print()
print(f"  sweep B: pc {PC_CHOICES} at snes_atol={USE_ATOL:.0e}")
print()

sweep_b = []
for pc_type, sub_pc_type in PC_CHOICES:
    rec = run_case(USE_ATOL, pc_type=pc_type, sub_pc_type=sub_pc_type,
                   label=f"pc={pc_type}")
    sweep_b.append(rec)
    if rec["status"] != "ok":
        print(f"  pc={pc_type}: {rec['status']}")
    else:
        print(f"  pc={pc_type}/{sub_pc_type or '-'}: its/solve={rec['its_per_solve']:.2f} "
              f"solve={rec['seconds_solve']:.1f}s delta={state_delta(rec, reference):.3e}")
    sys.stdout.flush()

# %%
# Sweep C -- the production pathology: linear tolerance LOOSER than nonlinear.

BAD_CONFIGS = [
    ("inverted (ksp=1e3 x atol)", 1e-6, 1e3),
    ("production (ksp=1e-6 fixed, atol=1e-8)", 1e-8, 1e2),
]

print()
print("  sweep C: inverted tolerance hierarchy (the production failure mode)")
print()

sweep_c = []
for label, atol, factor in BAD_CONFIGS:
    rec = run_case(atol, pc_type="gamg", ksp_rtol_factor=factor, label=label)
    sweep_c.append(rec)
    if rec["status"] != "ok":
        print(f"  {label}: {rec['status']}")
    else:
        print(f"  {label}: its/solve={rec['its_per_solve']:.2f} "
              f"solve={rec['seconds_solve']:.1f}s delta={state_delta(rec, reference):.3e}")
    sys.stdout.flush()

SWEEP_SECONDS = time.perf_counter() - t_sweep_start

# %%
# Tables.

print()
print("### Sweep A -- tolerance (pc='gamg', ksp_rtol = 1e-2 * snes_atol)")
print("| snes_atol | ksp_rtol | status | its/solve | tot_its | solve_s | steps | delta |")
print("|---|---|---|---|---|---|---|---|")
for r in sweep_a:
    if r["status"] != "ok":
        print(f"| {r['snes_atol']:.0e} | {r['ksp_rtol']:.0e} | {r['status'][:40]} | -- | -- | -- | -- | -- |")
        continue
    print(f"| {r['snes_atol']:.0e} | {r['ksp_rtol']:.0e} | ok | {r['its_per_solve']:.2f} | "
          f"{r['snes_its']} | {r['seconds_solve']:.1f} | {r['n_steps']} | {r['delta_vs_tightest']:.2e} |")

print()
print(f"### Sweep B -- preconditioner at snes_atol={USE_ATOL:.0e}")
print("| pc_type | sub_pc | status | its/solve | solve_s | delta |")
print("|---|---|---|---|---|---|")
for r in sorted(sweep_b, key=lambda r: r.get("seconds_solve", float("inf"))):
    if r["status"] != "ok":
        print(f"| {r['pc_type']} | {r.get('sub_pc_type') or '-'} | {r['status'][:40]} | -- | -- | -- |")
        continue
    print(f"| {r['pc_type']} | {r.get('sub_pc_type') or '-'} | ok | {r['its_per_solve']:.2f} | "
          f"{r['seconds_solve']:.1f} | {state_delta(r, reference):.2e} |")

print()
print("### Sweep C -- inverted hierarchy")
print("| config | snes_atol | ksp_rtol | status | its/solve | solve_s |")
print("|---|---|---|---|---|---|")
for r in sweep_c:
    its = f"{r['its_per_solve']:.2f}" if r.get("status") == "ok" else "--"
    sec = f"{r['seconds_solve']:.1f}" if r.get("status") == "ok" else "--"
    print(f"| {r['label']} | {r['snes_atol']:.0e} | {r['ksp_rtol']:.0e} | {r['status'][:40]} | {its} | {sec} |")

# %%
# Verdicts.

print()
print("### Verdicts")

ok_a_sorted = [r for r in sweep_a if r["status"] == "ok"]
if ok_a_sorted:
    lo, hi = ok_a_sorted[0], ok_a_sorted[-1]
    print(f"P1  its/solve {hi['its_per_solve']:.2f} at atol={hi['snes_atol']:.0e} vs "
          f"{lo['its_per_solve']:.2f} at atol={lo['snes_atol']:.0e} -> "
          f"{'CONFIRMED (nonlinear, iterations respond to tolerance)' if hi['its_per_solve'] > lo['its_per_solve'] or hi['snes_its'] > lo['snes_its'] else 'REFUTED'}")

if BEST_ATOL is not None:
    knee = next(r for r in ok_a_sorted if r["snes_atol"] == BEST_ATOL)
    print(f"P2  knee at snes_atol={BEST_ATOL:.0e}: delta {knee['delta_vs_tightest']:.2e}, "
          f"{knee['its_per_solve']:.2f} its/solve, {knee['seconds_solve']:.1f}s")

bad_ok = [r for r in sweep_c if r["status"] == "ok"]
bad_fail = [r for r in sweep_c if r["status"] != "ok"]
if bad_fail:
    print(f"P3  CONFIRMED: inverted hierarchy diverged in {len(bad_fail)}/{len(sweep_c)} "
          f"config(s) -> {[r['label'] for r in bad_fail]}")
elif bad_ok and ok_a_sorted:
    ref_cost = next(r for r in ok_a_sorted if r["snes_atol"] == USE_ATOL)
    worst = max(bad_ok, key=lambda r: r["seconds_solve"])
    print(f"P3  inverted hierarchy converged but cost {worst['seconds_solve']:.1f}s vs "
          f"{ref_cost['seconds_solve']:.1f}s correctly ordered "
          f"({worst['seconds_solve'] / ref_cost['seconds_solve']:.2f}x)")

ok_b = [r for r in sweep_b if r["status"] == "ok"]
if ok_b:
    spread = max(state_delta(r, reference) for r in ok_b)
    fastest = min(ok_b, key=lambda r: r["seconds_solve"])
    slowest = max(ok_b, key=lambda r: r["seconds_solve"])
    print(f"P4  fastest pc = {fastest['pc_type']}/{fastest.get('sub_pc_type') or '-'} "
          f"({fastest['seconds_solve']:.1f}s) vs slowest {slowest['pc_type']} "
          f"({slowest['seconds_solve']:.1f}s) -> {slowest['seconds_solve'] / fastest['seconds_solve']:.2f}x; "
          f"max state delta {spread:.2e}")

# %%
out = Path(__file__).resolve().parent / "output"
out.mkdir(exist_ok=True)
payload = {"sweep_a": sweep_a, "sweep_b": sweep_b, "sweep_c": sweep_c,
           "knee_snes_atol": BEST_ATOL,
           "fixed": {"cell_size": CELL_SIZE, "degree": DEGREE, "order": ORDER,
                     "components": COMPONENTS, "D_base": D_BASE}}
(out / "solver_tolerance_sweep_nonlinear.json").write_text(
    json.dumps(payload, indent=2, default=float))
print()
print(f"wrote {out / 'solver_tolerance_sweep_nonlinear.json'}")
print(f"total sweep wall time: {SWEEP_SECONDS:.1f} s")
