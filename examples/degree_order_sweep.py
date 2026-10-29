# %%
"""Degree x order sweep of the 2D UW3 solver against the exact eigenmode.

End-to-end test of a prediction made from two analytical floors:

* the **approximation floor** -- L2 projection of the exact radial eigenmode
  onto piecewise degree-p polynomials (`examples/p_vs_h_projection_study.py`),
  which no solver can beat;
* the **exact modal time error** for Backward Euler / BDF2 / BDF3, computable
  in closed form from the amplification factor.

Together those predict a complete error budget for every (degree, order, h).
This script runs the real solver and checks the budget.

Predictions
-----------
P1  At `degree=2, order=1` with the library's default timestep heuristic
    (`dt = 0.1 h^2 / kappa`), total error is dominated by TIME error by
    2-4 orders of magnitude, and does not fall to the spatial floor.
P2  Raising `order` to 2 or 3 drops the error by 1-2 orders of magnitude and
    moves the fitted rate toward the Q2 floor (~3).
P3  At fixed `order=1`, `degree=2 -> 3` buys almost nothing (both buried under
    the same time error).  At `order=3`, `degree=2 -> 3` buys clearly.
P4  Measured error is bracketed by (time error + spatial floor), within a
    factor of a few.

Run from the repository root in an environment with the UW3/PETSc/MPI stack
available (see AGENTS.md):

    python examples/degree_order_sweep.py

Machine-readable results go to ``examples/output/degree_order_sweep.json``
(gitignored).
"""

# %%
import contextlib
import gc
import json
import os
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

import underworld3 as uw

from UWDiffusion import MulticomponentDiffusionModel

# %%
# Matched physics -- identical to examples/convergence_1d_2d_cylindrical_multicomponent.py
R_INNER = 0.5
R_OUTER = 1.0
DIFFUSION_MATRIX = np.array([[1.0, 0.3], [0.2, 0.5]])
COMPONENTS = ["Mg", "Fe", "Ca"]
MEAN_INDEPENDENT = np.array([0.5, 0.25])
MODE_AMPLITUDE = 0.15
FINAL_TIME = 0.02
N_COMMON_RADII = 101
N_ANGLES = 64

DEGREES = [1, 2, 3]
ORDERS = [1, 2, 3]
CELL_SIZES = [0.08, 0.04]

TIME_STEP_FACTOR = 0.1  # the library default -- deliberately NOT overridden


# %%
# Analytical reference helpers (verbatim from the convergence study).


def neumann_wavenumber(r_inner: float, r_outer: float) -> float:
    """First positive annulus radial eigenvalue satisfying u'=0 at both rims."""

    def determinant(k: float) -> float:
        return j1(k * r_inner) * y1(k * r_outer) - j1(k * r_outer) * y1(k * r_inner)

    return float(brentq(determinant, 6.0, 7.0, xtol=1e-14))


def radial_mode(r: np.ndarray, k: float) -> np.ndarray:
    """Neumann radial Laplacian eigenfunction, normalised to max(abs(u))=1."""
    unscaled = y1(k * R_INNER) * j0(k * r) - j1(k * R_INNER) * y0(k * r)
    dense_r = np.linspace(R_INNER, R_OUTER, 20001)
    dense_u = y1(k * R_INNER) * j0(k * dense_r) - j1(k * R_INNER) * y0(k * dense_r)
    return unscaled / np.max(np.abs(dense_u))


def matrix_eigenpairs() -> list[dict]:
    """Both real eigenpairs of D sorted by decreasing eigenvalue."""
    eigenvalues, eigenvectors = np.linalg.eig(DIFFUSION_MATRIX)
    assert np.all(np.isreal(eigenvalues)), eigenvalues
    order = np.argsort(eigenvalues.real)[::-1]
    pairs = []
    for idx in order:
        eigenvector = np.real(eigenvectors[:, idx])
        eigenvector /= np.linalg.norm(eigenvector)
        if eigenvector[0] < 0:
            eigenvector *= -1
        pairs.append({"lambda": float(np.real(eigenvalues[idx])), "eigenvector": eigenvector})
    return pairs


def exact_profiles(r: np.ndarray, time: float, k: float, pair: dict) -> np.ndarray:
    """Exact [Mg, Fe, Ca] profiles for one coupled eigenmode."""
    decay = np.exp(-pair["lambda"] * k**2 * time)
    perturbation = MODE_AMPLITUDE * radial_mode(r, k) * decay
    independent = MEAN_INDEPENDENT[:, None] + pair["eigenvector"][:, None] * perturbation
    dependent = 1.0 - independent.sum(axis=0)
    return np.vstack((independent, dependent))


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
# Closed-form time-error prediction for constant-step BDF of a given order.


def bdf_amplification(order: int, dt: float, mu: float) -> float:
    """Modal amplification over FINAL_TIME for constant-step BDF of `order`.

    BDF-p:  sum_{j=1..p} (1/j) grad^j u^{n+1} = -mu*dt*u^{n+1}
    with grad^j u^{n+1} = sum_{i=0..j} (-1)^i C(j,i) u^{n+1-i}.
    """
    z = mu * dt
    p = order
    coef = np.zeros(p + 2)  # coef[m] is the coefficient of g**m
    for j in range(1, p + 1):
        for i in range(0, j + 1):
            coef[p - i] += (1.0 / j) * ((-1) ** i) * comb(j, i)
    coef[p] += z
    roots = np.roots(coef[::-1])  # np.roots wants highest power first
    g = roots[np.argmin(np.abs(roots - 1.0))]
    n = int(round(FINAL_TIME / dt))
    return float(np.real(g)) ** n


def predicted_time_error(order: int, dt: float, pair: dict) -> float:
    """Peak |component| time error over the profile, for the modal amplitude."""
    mu = pair["lambda"] * k_wavenumber**2
    exact = np.exp(-mu * FINAL_TIME)
    num = bdf_amplification(order, dt, mu)
    v = pair["eigenvector"]
    peak_scale = MODE_AMPLITUDE * np.max(np.abs(np.array([v[0], v[1], -(v[0] + v[1])])))
    return float(abs(num - exact) * peak_scale)


# %%
# Spatial approximation floor, fitted from examples/p_vs_h_projection_study.py.
# Values are max-norm error on the unit-normalised radial mode at h = 0.003125,
# with the fitted rates; scaled here to the modal amplitude and final decay.

_PROJ = {
    1: {"rate": 1.97, "err_at": 8.305e-06},
    2: {"rate": 2.99, "err_at": 6.815e-09},
    3: {"rate": 3.99, "err_at": 4.180e-12},
}
_PROJ_H = 0.003125


def predicted_spatial_error(degree: int, h: float, pair: dict) -> float:
    c = _PROJ[degree]
    unit = c["err_at"] * (h / _PROJ_H) ** c["rate"]
    decay = np.exp(-pair["lambda"] * k_wavenumber**2 * FINAL_TIME)
    return float(unit * MODE_AMPLITUDE * decay)


# %%
# One 2D run at a chosen (cellSize, degree, order), using the DEFAULT heuristic.

ANNULUS_AREA = np.pi * (R_OUTER**2 - R_INNER**2)
D_SYMBOLS = sp.Matrix(
    [[sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")], [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")]]
)
D_VALUES = {
    "D_MgMg": float(DIFFUSION_MATRIX[0, 0]),
    "D_MgFe": float(DIFFUSION_MATRIX[0, 1]),
    "D_FeMg": float(DIFFUSION_MATRIX[1, 0]),
    "D_FeFe": float(DIFFUSION_MATRIX[1, 1]),
}

k_wavenumber = neumann_wavenumber(R_INNER, R_OUTER)
EIGENPAIRS = matrix_eigenpairs()
PAIR = EIGENPAIRS[0]  # dominant, lambda = 1.1

r_common = np.linspace(R_INNER + 1e-10, R_OUTER - 1e-10, N_COMMON_RADII)
angles = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
sample_xy = np.column_stack(
    (
        (r_common[:, None] * np.cos(angles)).ravel(),
        (r_common[:, None] * np.sin(angles)).ravel(),
    )
)


def run_case(cell_size: float, degree: int, order: int) -> dict:
    t0 = time.perf_counter()
    mesh = uw.meshing.Annulus(
        radiusOuter=R_OUTER, radiusInner=R_INNER, cellSize=cell_size, qdegree=2
    )
    n_cells = int(mesh.dm.getDepthStratum(0)[1] - mesh.dm.getDepthStratum(0)[0])
    h_eff = float(np.sqrt(ANNULUS_AREA / n_cells))
    min_radius = float(mesh.get_min_radius())

    try:
        model = MulticomponentDiffusionModel(
            COMPONENTS, mesh, diffusion_matrix=D_SYMBOLS, degree=degree, order=order
        )
    except Exception as exc:  # degree != 2 is untested territory
        return {"cell_size": cell_size, "degree": degree, "order": order,
                "h_eff": h_eff, "n_cells": n_cells, "status": f"FAILED: {type(exc).__name__}: {exc}"}

    model.set_diffusion_values(D_VALUES)
    for solver in model.solvers:
        solver.petsc_options["snes_rtol"] = 1e-12
        solver.petsc_options["snes_atol"] = 1e-6

    coords = model.mesh_vars[0].coords
    r_nodes = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    initial = exact_profiles(r_nodes, 0.0, k_wavenumber, PAIR)
    with mesh.access(*model.mesh_vars):
        for idx, mesh_var in enumerate(model.mesh_vars):
            mesh_var.data[:, 0] = initial[idx]

    mass_0 = [float(uw.maths.Integral(mesh, fn=v.sym[0]).evaluate()) for v in model.mesh_vars]

    with quiet_stdout():
        model.run_simulation(duration=FINAL_TIME)  # DEFAULT timestep heuristic

    n_steps = int(model.step)
    dt_used = FINAL_TIME / n_steps if n_steps else float("nan")

    mass_T = [float(uw.maths.Integral(mesh, fn=v.sym[0]).evaluate()) for v in model.mesh_vars]
    samples = np.stack(
        [uw.function.evaluate(v.sym[0], sample_xy).reshape(N_COMMON_RADII, N_ANGLES)
         for v in model.mesh_vars]
    )
    profiles = np.mean(samples, axis=2)
    closure = float(np.max(np.abs(samples.sum(axis=0) - 1.0)))

    ref = exact_profiles(r_common, FINAL_TIME, k_wavenumber, PAIR)
    diff = profiles - ref
    err_max = float(np.max(np.abs(diff)))
    err_l2 = float(np.sqrt(np.mean(diff**2)))

    out = {
        "cell_size": cell_size, "degree": degree, "order": order,
        "h_eff": h_eff, "n_cells": n_cells,
        "n_dof": int(model.mesh_vars[0].data.shape[0]) * degree,  # approximate
        "min_radius": min_radius, "dt_used": dt_used, "n_steps": n_steps,
        "err_max": err_max, "err_l2": err_l2,
        "mass_drift": float(max(abs(b - a) for a, b in zip(mass_0, mass_T))),
        "closure_residual": closure,
        "seconds": time.perf_counter() - t0,
        "status": "ok",
    }
    out["pred_time"] = predicted_time_error(order, dt_used, PAIR)
    out["pred_space"] = predicted_spatial_error(degree, h_eff, PAIR)
    out["pred_total"] = out["pred_time"] + out["pred_space"]
    out["measured_over_pred"] = err_max / out["pred_total"] if out["pred_total"] > 0 else float("nan")

    del model, mesh
    gc.collect()
    return out


# %%
# Sweep.

kappa_max = float(np.max(np.abs(np.linalg.eigvals(DIFFUSION_MATRIX))))
print("Degree x order sweep: 2D UW3 vs exact eigenmode")
print(f"  k = {k_wavenumber:.12g}; lambda = {PAIR['lambda']}; kappa_max = {kappa_max}")
print(f"  default heuristic dt = {TIME_STEP_FACTOR} * h^2 / kappa_max")
print(f"  grid: degree {DEGREES} x order {ORDERS} x cellSize {CELL_SIZES}")
print()

t_sweep_start = time.perf_counter()
results = []
for cell_size in CELL_SIZES:
    for degree in DEGREES:
        for order in ORDERS:
            rec = run_case(cell_size, degree, order)
            results.append(rec)
            if rec["status"] != "ok":
                print(f"  cs={cell_size} deg={degree} ord={order}: {rec['status']}")
            else:
                print(
                    f"  cs={cell_size} deg={degree} ord={order}: "
                    f"h_eff={rec['h_eff']:.4g} dt={rec['dt_used']:.3e} steps={rec['n_steps']:5d} "
                    f"err={rec['err_max']:.3e} pred={rec['pred_total']:.3e} "
                    f"ratio={rec['measured_over_pred']:.2f} ({rec['seconds']:.1f}s)"
                )
            sys.stdout.flush()

SWEEP_SECONDS = time.perf_counter() - t_sweep_start

# %%
# Tables.

ok = [r for r in results if r["status"] == "ok"]

print()
print("### Main table")
print("| degree | order | cellSize | h_eff | dt_used | steps | err_max | err_l2 | wall_s |")
print("|---|---|---|---|---|---|---|---|---|")
for r in ok:
    print(f"| {r['degree']} | {r['order']} | {r['cell_size']} | {r['h_eff']:.4g} | "
          f"{r['dt_used']:.3e} | {r['n_steps']} | {r['err_max']:.3e} | {r['err_l2']:.3e} | {r['seconds']:.1f} |")

print()
print("### Prediction table (P4)")
print("| degree | order | cellSize | measured | pred_time | pred_space | measured/pred |")
print("|---|---|---|---|---|---|---|")
for r in ok:
    print(f"| {r['degree']} | {r['order']} | {r['cell_size']} | {r['err_max']:.3e} | "
          f"{r['pred_time']:.3e} | {r['pred_space']:.3e} | {r['measured_over_pred']:.2f} |")

print()
print("### Fitted rates (two finest levels)")
print("| degree | order | rate (err_max vs h_eff) |")
print("|---|---|---|")
for degree in DEGREES:
    for order in ORDERS:
        sub = sorted([r for r in ok if r["degree"] == degree and r["order"] == order],
                     key=lambda r: -r["h_eff"])
        if len(sub) >= 2:
            hs = np.array([r["h_eff"] for r in sub[:2]])
            es = np.array([r["err_max"] for r in sub[:2]])
            rate = float(np.log(es[0] / es[1]) / np.log(hs[0] / hs[1]))
            print(f"| {degree} | {order} | {rate:.2f} |")

# %%
# Cost / accuracy trade-off: the wall-clock axis.

print()
print(f"### Cost vs accuracy  (whole sweep took {SWEEP_SECONDS:.1f} s)")
print()
print("Pareto frontier -- no other configuration is BOTH faster and more accurate:")
print("| degree | order | cellSize | h_eff | steps | err_max | wall_s | err x wall (lower better) |")
print("|---|---|---|---|---|---|---|---|")
frontier = []
for r in ok:
    dominated = any(
        (o is not r) and (o["err_max"] <= r["err_max"]) and (o["seconds"] <= r["seconds"])
        and (o["err_max"] < r["err_max"] or o["seconds"] < r["seconds"])
        for o in ok
    )
    if not dominated:
        frontier.append(r)
for r in sorted(frontier, key=lambda r: r["err_max"]):
    print(f"| {r['degree']} | {r['order']} | {r['cell_size']} | {r['h_eff']:.4g} | {r['n_steps']} | "
          f"{r['err_max']:.3e} | {r['seconds']:.1f} | {r['err_max']*r['seconds']:.2e} |")

print()
print("Cheapest configuration to reach each accuracy target:")
print("| target err_max | degree | order | cellSize | steps | err_max | wall_s |")
print("|---|---|---|---|---|---|---|")
for target in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7):
    cands = [r for r in ok if r["err_max"] <= target]
    if not cands:
        print(f"| <= {target:.0e} | -- | -- | -- | -- | not reached | -- |")
        continue
    best = min(cands, key=lambda r: r["seconds"])
    print(f"| <= {target:.0e} | {best['degree']} | {best['order']} | {best['cell_size']} | "
          f"{best['n_steps']} | {best['err_max']:.3e} | {best['seconds']:.1f} |")

print()
print("Full cost ranking (cheapest first):")
print("| # | degree | order | cellSize | err_max | wall_s | steps |")
print("|---|---|---|---|---|---|---|")
for i, r in enumerate(sorted(ok, key=lambda r: r["seconds"]), 1):
    print(f"| {i} | {r['degree']} | {r['order']} | {r['cell_size']} | "
          f"{r['err_max']:.3e} | {r['seconds']:.1f} | {r['n_steps']} |")

# %%
# P1-P4 verdicts.

print()
print("### Verdicts")


def find(degree, order, cell_size):
    for r in ok:
        if r["degree"] == degree and r["order"] == order and r["cell_size"] == cell_size:
            return r
    return None


p1 = find(2, 1, 0.04)
if p1:
    print(f"P1  degree=2, order=1, cs=0.04: measured {p1['err_max']:.2e} vs time {p1['pred_time']:.2e} "
          f"vs space {p1['pred_space']:.2e}. time/space = {p1['pred_time']/p1['pred_space']:.0f}x "
          f"-> {'CONFIRMED (time dominated)' if p1['pred_time'] > 10 * p1['pred_space'] else 'REFUTED'}")

for order in (2, 3):
    a, b = find(2, 1, 0.04), find(2, order, 0.04)
    if a and b:
        print(f"P2  order 1 -> {order} at degree=2, cs=0.04: {a['err_max']:.2e} -> {b['err_max']:.2e} "
              f"(x{a['err_max']/b['err_max']:.0f} better)")

for order in (1, 3):
    a, b = find(2, order, 0.04), find(3, order, 0.04)
    if a and b:
        print(f"P3  degree 2 -> 3 at order={order}, cs=0.04: {a['err_max']:.2e} -> {b['err_max']:.2e} "
              f"(x{a['err_max']/b['err_max']:.2f})")

ratios = [r["measured_over_pred"] for r in ok if np.isfinite(r["measured_over_pred"])]
if ratios:
    print(f"P4  measured/predicted across {len(ratios)} runs: "
          f"min {min(ratios):.2f}, median {np.median(ratios):.2f}, max {max(ratios):.2f}")

# %%
out = Path(__file__).resolve().parent / "output"
out.mkdir(exist_ok=True)
(out / "degree_order_sweep.json").write_text(json.dumps(results, indent=2, default=float))
print()
print(f"wrote {out / 'degree_order_sweep.json'}")
print(f"total sweep wall time: {SWEEP_SECONDS:.1f} s")
