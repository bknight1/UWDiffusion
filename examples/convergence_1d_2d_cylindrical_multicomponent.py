# %%
"""Spatial and temporal convergence study: 1D cylindrical FV vs 2D UW3 FEM.

Reproducible companion to the one-shot comparisons
``examples/compare_1d_2d_cylindrical.py`` and
``examples/compare_1d_2d_cylindrical_multicomponent.py`` (both preserved
unchanged).  This script quantifies *how* the genuinely coupled multicomponent
annulus problem converges, using the exact continuum eigenmode solution as
reference and reporting the two codes separately (1D-vs-2D differences are
secondary cross-checks, not the convergence metric).

Physics (identical to the multicomponent one-shot): components Mg, Fe
independent with Ca = 1 - Mg - Fe closure, reduced diffusion matrix
``D = [[1.0, 0.3], [0.2, 0.5]]`` (nonsymmetric, genuinely coupled), annulus
``0.5 <= r <= 1.0``, homogeneous-Neumann rims (explicit in 1D, natural in 2D),
and a radially symmetric Bessel eigenmode so the exact solution is known for
any time:

    x_i(r, t) = x0_i + a * v_i * u(r) * exp(-lambda * k**2 * t),

where ``(lambda, v)`` is an eigenpair of D, ``u(r)`` is the first radial
Neumann eigenfunction of the annulus, and Ca follows by closure.  BOTH
eigenpairs of D are studied (lambda = 1.1 dominant, lambda = 0.4
subdominant).

Study design
------------
* SPATIAL (both codes, per eigenmode): refine space independently of each
  other at controlled (small) time error.  Both codes use a first-order
  Backward-Euler Richardson pair X(0) ~ 2X(dt/2) - X(dt) at every level to
  remove the leading O(dt) time error (the leading term cancels exactly for
  the modal decay), leaving spatial + angular discretisation error against
  the exact solution:
  - 1D: N = 26, 51, 101, 201 vertices (dr = 0.02 ... 0.0025, exact ratio 2)
    at dt = 1e-5 and 5e-6 (raw BE error at dt=1e-5 is ~1.2e-5 on the Mg peak
    -- larger than the finest spatial error -- which is why Richardson is
    used; its residual floor is ~1e-9).
  - 2D: cellSize = 0.08, 0.04, 0.02 (actual effective ratios reported from
    h_eff = sqrt(area / n_cells)), each at dt = 5e-4 AND 2.5e-4 (Richardson
    residual floor ~2.8e-6 on the Mg peak, below the finest spatial error).
    The un-extrapolated ||X(dt) - X(dt/2)|| time content is reported per
    level, and the analytic Richardson floor is computed exactly from the
    modal BE amplification factor.
* TEMPORAL (both codes, per eigenmode): fixed fine spatial resolution
  (1D: N = 201, dr = 0.0025; 2D: cellSize = 0.02), dt ladder
  {0.002, 0.001, 0.0005, 0.00025} against a finer-step reference on the SAME
  mesh (dt_ref = 6.25e-5 = T/320, dt_min/4).  Observed rates use the actual
  refinement ratios; because the reference is finite, e(dt) ~ C (dt - dt_ref),
  and the constant C = e/(dt - dt_ref) is also reported to expose clean
  first-order BE behaviour.
* CHECKS: closure residual (Mg+Fe+Ca - 1 at sampled points), per-component
  mass drift (exact conservation expected for homogeneous Neumann), SNES
  convergence diagnostics per 2D step (post-solve hooks), and a 1D linear
  system residual probe (assemble + one additional step at the finest grid).

Run in the documented UW3 environment from the repository root:

    source /Users/benknight/software/venv/uw3-dev/bin/activate && \
        python examples/convergence_1d_2d_cylindrical_multicomponent.py

Machine-readable results are written to
``examples/output/convergence_study_summary.json`` (gitignored).
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
import sympy as sp
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

import underworld3 as uw

# Allow direct execution from a source checkout without requiring an editable
# UWDiffusion install in the activated Underworld environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from UWDiffusion import MulticomponentDiffusion1D, MulticomponentDiffusionModel

# %%
# Matched physics (as in the multicomponent one-shot comparison).
R_INNER = 0.5
R_OUTER = 1.0
DIFFUSION_MATRIX = np.array([[1.0, 0.3], [0.2, 0.5]])
COMPONENTS = ["Mg", "Fe", "Ca"]  # Ca is the dependent closure component
MEAN_INDEPENDENT = np.array([0.5, 0.25])
MODE_AMPLITUDE = 0.15
FINAL_TIME = 0.02
N_COMMON_RADII = 101
N_ANGLES = 64

# Spatial study configuration (Richardson-in-dt pair at every level).
SPATIAL_1D_N = [26, 51, 101, 201]  # dr = 0.02, 0.01, 0.005, 0.0025 (ratio 2)
DT_1D_SPATIAL = 1.0e-5  # coarse member of the 1D Richardson pair
DT_1D_SPATIAL_HALVED = 5.0e-6  # fine member; Richardson floor ~1e-9 (Mg)
SPATIAL_2D_CELLSIZE = [0.08, 0.04, 0.02]
DT_2D_SPATIAL = 5.0e-4  # coarse member of the 2D Richardson pair
DT_2D_SPATIAL_HALVED = DT_2D_SPATIAL / 2.0  # Richardson floor ~2.8e-6 (Mg)

# Temporal study configuration (fixed fine spatial resolution per code).
TEMPORAL_1D_N = 201  # dr = 0.0025
TEMPORAL_2D_CELLSIZE = 0.02
TEMPORAL_DT_LADDER = [0.002, 0.001, 0.0005, 0.00025]
TEMPORAL_DT_REF = 0.0000625  # T/320 = dt_min/4, same-mesh finer-step reference


# %%
# Analytical reference helpers (shared with the one-shot comparison).
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
    """Return both real eigenpairs of D sorted by decreasing eigenvalue."""
    eigenvalues, eigenvectors = np.linalg.eig(DIFFUSION_MATRIX)
    assert np.all(np.isreal(eigenvalues)), eigenvalues
    order = np.argsort(eigenvalues.real)[::-1]
    pairs = []
    for idx in order:
        eigenvector = np.real(eigenvectors[:, idx])
        eigenvector /= np.linalg.norm(eigenvector)
        if eigenvector[0] < 0:
            eigenvector *= -1  # keep the Mg component positive
        pairs.append(
            {
                "lambda": float(np.real(eigenvalues[idx])),
                "eigenvector": eigenvector,
            }
        )
    return pairs


def exact_profiles(r: np.ndarray, time: float, k: float, pair: dict) -> np.ndarray:
    """Return exact [Mg, Fe, Ca] profiles for one coupled eigenmode."""
    decay = np.exp(-pair["lambda"] * k**2 * time)
    perturbation = MODE_AMPLITUDE * radial_mode(r, k) * decay
    independent = MEAN_INDEPENDENT[:, None] + pair["eigenvector"][:, None] * perturbation
    dependent = 1.0 - independent.sum(axis=0)
    return np.vstack((independent, dependent))


def be_modal_amplitude(dt: float, mu: float) -> float:
    """Exact Backward-Euler decay factor (1 + mu dt)^(-n) for n = T/dt steps."""
    n_steps = int(round(FINAL_TIME / dt))
    return float((1.0 + mu * dt) ** (-n_steps))


def richardson_time_floor(pair: dict, k: float, dt: float) -> np.ndarray:
    """Peak |component| residual time error after dt -> 0 Richardson in BE."""
    mu = pair["lambda"] * k**2
    exact = np.exp(-mu * FINAL_TIME)
    re = 2.0 * be_modal_amplitude(dt / 2.0, mu) - be_modal_amplitude(dt, mu)
    v = pair["eigenvector"]
    # Ca = 1 - Mg - Fe, so its modal weight is -(v_Mg + v_Fe).
    peak_scale = MODE_AMPLITUDE * np.abs(np.array([v[0], v[1], -(v[0] + v[1])]))
    return np.abs(re - exact) * peak_scale


def error_metrics(num: np.ndarray, ref: np.ndarray) -> dict:
    """Component-wise max-abs and RMSE of (num - ref) for 3 x N profiles."""
    diff = num - ref
    return {
        "max_abs": [float(v) for v in np.max(np.abs(diff), axis=1)],
        "rmse": [float(np.sqrt(np.mean(diff[i] ** 2))) for i in range(diff.shape[0])],
    }


def observed_rate(err_coarse: float, err_fine: float, ratio: float) -> float:
    """p from e ~ h^p using the actual (not assumed) refinement ratio."""
    if err_fine <= 0.0 or err_coarse <= 0.0:
        return float("nan")
    return float(np.log(err_coarse / err_fine) / np.log(ratio))


@contextlib.contextmanager
def quiet_stdout():
    """Silence both Python and C-level (PETSc printf) stdout during a run."""
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


k_wavenumber = neumann_wavenumber(R_INNER, R_OUTER)
EIGENPAIRS = matrix_eigenpairs()
r_common = np.linspace(R_INNER + 1e-10, R_OUTER - 1e-10, N_COMMON_RADII)
angles = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
sample_xy = np.column_stack(
    (
        (r_common[:, None] * np.cos(angles)).ravel(),
        (r_common[:, None] * np.sin(angles)).ravel(),
    )
)
ANNULUS_AREA = np.pi * (R_OUTER**2 - R_INNER**2)
D_SYMBOLS = sp.Matrix(
    [
        [sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")],
        [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")],
    ]
)
D_VALUES = {
    "D_MgMg": float(DIFFUSION_MATRIX[0, 0]),
    "D_MgFe": float(DIFFUSION_MATRIX[0, 1]),
    "D_FeMg": float(DIFFUSION_MATRIX[1, 0]),
    "D_FeFe": float(DIFFUSION_MATRIX[1, 1]),
}

print("Convergence study: coupled multicomponent annulus, exact eigenmode reference")
print(f"  D = {DIFFUSION_MATRIX.tolist()}; components {COMPONENTS} (Ca closure)")
print(f"  annulus radii [{R_INNER:g}, {R_OUTER:g}]; T = {FINAL_TIME:g}")
print(f"  radial Neumann wavenumber k = {k_wavenumber:.12g}")
for pair in EIGENPAIRS:
    print(
        f"  eigenmode lambda = {pair['lambda']:.6g}, "
        f"v = {pair['eigenvector']}, "
        f"mu = lambda k^2 = {pair['lambda'] * k_wavenumber**2:.6g}"
    )


# %%
# 1D runner: cylindrical FV with homogeneous-Neumann rims, constant coupled D.
def run_1d(n_points: int, dt: float, pair: dict) -> dict:
    """Run one 1D FV case; return final profiles and conservation diagnostics."""
    t0 = time.perf_counter()
    grid = np.linspace(R_INNER, R_OUTER, n_points)
    independent_initial = (
        MEAN_INDEPENDENT[:, None]
        + MODE_AMPLITUDE * pair["eigenvector"][:, None] * radial_mode(grid, k_wavenumber)
    )
    initial = np.vstack(
        (independent_initial, 1.0 - independent_initial.sum(axis=0))
    ).T
    assert initial.min() > 0.0 and initial.max() < 1.0, initial.min()

    model_1d = MulticomponentDiffusion1D(
        grid=grid,
        diffusivity=DIFFUSION_MATRIX,
        geometry="cylindrical",
        components=COMPONENTS,
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
        initial_composition=initial,
    )
    result = model_1d.run_simulation(dt=dt, total_time=FINAL_TIME)

    volumes = model_1d.cell_volumes
    mass_0 = volumes @ result.compositions[0]
    mass_T = volumes @ result.compositions[-1]
    closure = float(np.max(np.abs(result.compositions[-1].sum(axis=1) - 1.0)))
    assert np.isclose(result.times[-1], FINAL_TIME, rtol=0.0, atol=1e-12)
    # Default save_interval=None records only the initial and final states.
    assert result.compositions.shape == (2, n_points, len(COMPONENTS))

    # Linear-solve residual probe: assemble the BE system from the final state
    # (constant D, so A is dt-dependent only), take one extra step, and measure
    # ||A x_new - b||_inf / ||b||_inf of that probe solve.
    x_final = model_1d._X.copy()
    A_probe, b_probe = model_1d._assemble_system(dt, x_final)
    model_1d.step(dt)
    x_probe = np.column_stack(
        [model_1d._X[:, c] for c in model_1d.independent_indices]
    ).ravel()
    resid = float(
        np.max(np.abs(A_probe @ x_probe - b_probe)) / np.max(np.abs(b_probe))
    )
    model_1d._X = x_final  # undo the probe step
    model_1d.time = FINAL_TIME

    return {
        "n_points": n_points,
        "dr": float(grid[1] - grid[0]),
        "dt": dt,
        "profiles": result.compositions[-1].T.copy(),  # (3, n_points)
        "r": grid.copy(),
        "mass_drift": [float(v) for v in np.abs(mass_T - mass_0)],
        "closure_residual": closure,
        "linear_residual_probe": resid,
        "seconds": time.perf_counter() - t0,
    }


def profiles_1d_on_common(run: dict) -> np.ndarray:
    """Interpolate 1D final profiles (3, n_points) onto the common radii."""
    return np.vstack(
        [np.interp(r_common, run["r"], run["profiles"][i]) for i in range(3)]
    )


# %%
# 2D runner: Q2 annulus FEM, natural (unconstrained) rims, coupled matrix.
def run_2d(cell_size: float, dt: float, pair: dict) -> dict:
    """Run one 2D UW3 case; return sampled profiles and solver diagnostics."""
    t0 = time.perf_counter()
    mesh = uw.meshing.Annulus(
        radiusOuter=R_OUTER,
        radiusInner=R_INNER,
        cellSize=cell_size,
        qdegree=2,
    )
    n_cells = int(mesh.dm.getDepthStratum(0)[1] - mesh.dm.getDepthStratum(0)[0])
    model_2d = MulticomponentDiffusionModel(
        COMPONENTS, mesh, diffusion_matrix=D_SYMBOLS, degree=2, order=1
    )
    model_2d.set_diffusion_values(D_VALUES)
    for solver in model_2d.solvers:
        solver.petsc_options["snes_rtol"] = 1e-12
        solver.petsc_options["snes_atol"] = 1e-6
        solver.petsc_options["snes_converged_reason"] = None

    diag_log: list[dict] = []

    def record_diagnostics() -> None:
        for solver in model_2d.solvers:
            diag_log.append(solver.get_snes_diagnostics())

    model_2d.post_solve_hooks.append(("diag", record_diagnostics))

    coords = model_2d.mesh_vars[0].coords
    r_nodes = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    initial = exact_profiles(r_nodes, 0.0, k_wavenumber, pair)
    assert initial.min() > 0.0 and initial.max() < 1.0, initial.min()
    with mesh.access(*model_2d.mesh_vars):
        for idx, mesh_var in enumerate(model_2d.mesh_vars):
            mesh_var.data[:, 0] = initial[idx]

    mass_0 = [
        float(uw.maths.Integral(mesh, fn=var.sym[0]).evaluate())
        for var in model_2d.mesh_vars
    ]

    with quiet_stdout():
        model_2d.run_simulation(duration=FINAL_TIME, min_dt=dt, max_dt=dt)

    n_steps_expected = int(round(FINAL_TIME / dt))
    # Allow the off-by-one that IEEE accumulation can introduce in the 2D
    # while-loop stepper; the final-time landing is the strict requirement.
    assert model_2d.step in (n_steps_expected, n_steps_expected + 1), (
        model_2d.step,
        n_steps_expected,
    )
    assert np.isclose(model_2d.current_time, FINAL_TIME, rtol=0.0, atol=1e-12)

    mass_T = [
        float(uw.maths.Integral(mesh, fn=var.sym[0]).evaluate())
        for var in model_2d.mesh_vars
    ]

    samples = np.stack(
        [
            uw.function.evaluate(mesh_var.sym[0], sample_xy).reshape(
                N_COMMON_RADII, N_ANGLES
            )
            for mesh_var in model_2d.mesh_vars
        ]
    )
    profiles = np.mean(samples, axis=2)
    angular_spread = np.sqrt(np.mean((samples - profiles[:, :, None]) ** 2, axis=2))
    closure = float(np.max(np.abs(samples.sum(axis=0) - 1.0)))

    snes_ok = all(d["converged"] for d in diag_log)
    max_snes_its = max(d["snes_iterations"] for d in diag_log)
    max_lin_its = max(d["linear_iterations"] for d in diag_log)

    h_eff = float(np.sqrt(ANNULUS_AREA / n_cells))
    out = {
        "cell_size": cell_size,
        "n_cells": n_cells,
        "n_vertices": int(mesh.X.coords.shape[0]),
        "h_eff": h_eff,
        "min_radius": float(mesh.get_min_radius()),
        "dt": dt,
        "profiles": profiles.copy(),  # (3, N_COMMON_RADII)
        "angular_spread_max": [float(v) for v in np.max(angular_spread, axis=1)],
        "mass_drift": [float(abs(b - a)) for a, b in zip(mass_0, mass_T)],
        "closure_residual": closure,
        "snes_all_converged": bool(snes_ok),
        "snes_max_iterations": int(max_snes_its),
        "linear_max_iterations": int(max_lin_its),
        "n_steps": int(model_2d.step),
        "seconds": time.perf_counter() - t0,
    }
    del model_2d, mesh
    gc.collect()
    return out


# %%
# SPATIAL STUDY - 1D code (exact dr ratio 2.0; BE Richardson pair at each level).
study_1d_spatial = {"per_eigenmode": [], "seconds": 0.0}
t_study = time.perf_counter()
print("\n=== SPATIAL CONVERGENCE: 1D cylindrical FV (vs exact eigenmode) ===")
print(f"  levels N = {SPATIAL_1D_N} (dr ratio exactly 2); BE Richardson pair "
      f"({DT_1D_SPATIAL:g}, {DT_1D_SPATIAL_HALVED:g}) at every level")
for pair in EIGENPAIRS:
    runs_a = [run_1d(n, DT_1D_SPATIAL, pair) for n in SPATIAL_1D_N]
    runs_b = [run_1d(n, DT_1D_SPATIAL_HALVED, pair) for n in SPATIAL_1D_N]
    exact_final = exact_profiles(r_common, FINAL_TIME, k_wavenumber, pair)

    metrics_raw, metrics_re, time_content, rates_max, rates_rmse = [], [], [], [], []
    for ra, rb in zip(runs_a, runs_b):
        metrics_raw.append(error_metrics(profiles_1d_on_common(ra), exact_final))
        prof_re = 2.0 * profiles_1d_on_common(rb) - profiles_1d_on_common(ra)
        metrics_re.append(error_metrics(prof_re, exact_final))
        time_content.append(
            error_metrics(profiles_1d_on_common(ra), profiles_1d_on_common(rb))
        )
    for i in range(len(runs_a) - 1):
        ratio = runs_a[i]["dr"] / runs_a[i + 1]["dr"]
        rates_max.append(
            [observed_rate(metrics_re[i]["max_abs"][c], metrics_re[i + 1]["max_abs"][c], ratio)
             for c in range(3)]
        )
        rates_rmse.append(
            [observed_rate(metrics_re[i]["rmse"][c], metrics_re[i + 1]["rmse"][c], ratio)
             for c in range(3)]
        )

    floor = richardson_time_floor(pair, k_wavenumber, DT_1D_SPATIAL)
    label = f"lambda={pair['lambda']:g}"
    print(f"\n  eigenmode {label}, v = {pair['eigenvector']}")
    print("    level |    dr   | Mg max    Fe max    Ca max | Mg rate  Fe rate  Ca rate")
    for i, ra in enumerate(runs_a):
        rate_row = (
            "  n/a"
            if i == 0
            else " ".join(f"{p:7.3f}" for p in rates_max[i - 1])
        )
        print(
            f"    N={ra['n_points']:4d} | {ra['dr']:.5f} | "
            + " ".join(f"{m:.3e}" for m in metrics_re[i]["max_abs"])
            + " | " + rate_row
        )
    print("    raw errors at dt (before Richardson), finest level: "
          + " ".join(f"{m:.3e}" for m in metrics_raw[-1]["max_abs"]))
    print(f"    Richardson time floor (analytic, dt={DT_1D_SPATIAL:g}): "
          + " ".join(f"{v:.2e}" for v in floor))
    print("    floor / Richardson spatial error at finest level: "
          + " ".join(
              f"{floor[c] / metrics_re[-1]['max_abs'][c]:.2e}" for c in range(3)
            ))
    print("    ||X(dt)-X(dt/2)|| raw time content by level (max, Mg/Fe/Ca):")
    for ra, tc in zip(runs_a, time_content):
        print(
            f"      N={ra['n_points']:4d}: "
            + " ".join(f"{v:.2e}" for v in tc["max_abs"])
            + "  (Richardson spatial err: "
            + " ".join(f"{m:.2e}" for m in metrics_re[runs_a.index(ra)]["max_abs"])
            + ")"
        )
    fin = runs_a[-1]
    print(f"    checks finest level: mass drift max={max(fin['mass_drift']):.2e}, "
          f"closure={fin['closure_residual']:.2e}, "
          f"linear residual probe={fin['linear_residual_probe']:.2e}")
    study_1d_spatial["per_eigenmode"].append(
        {
            "eigenmode": label,
            "lambda": pair["lambda"],
            "levels_dt": [
                {k: v for k, v in ra.items() if k not in ("profiles", "r")}
                | {"errors_vs_exact_raw": metrics_raw[i],
                   "errors_vs_exact_richardson": metrics_re[i],
                   "time_content_vs_dt_halved": time_content[i]}
                for i, ra in enumerate(runs_a)
            ],
            "levels_dt_halved": [
                {k: v for k, v in rb.items() if k not in ("profiles", "r")}
                for rb in runs_b
            ],
            "observed_rates_max_richardson": rates_max,
            "observed_rates_rmse_richardson": rates_rmse,
            "dr_ratios": [
                runs_a[i]["dr"] / runs_a[i + 1]["dr"]
                for i in range(len(runs_a) - 1)
            ],
            "richardson_time_floor": [float(v) for v in floor],
        }
    )
study_1d_spatial["seconds"] = time.perf_counter() - t_study
print(f"\n  [1D spatial study wall time: {study_1d_spatial['seconds']:.1f} s]")

# %%
# SPATIAL STUDY - 2D code (Richardson-in-dt at each level; actual h_eff ratios).
study_2d_spatial = {"per_eigenmode": [], "seconds": 0.0}
t_study = time.perf_counter()
print("\n=== SPATIAL CONVERGENCE: 2D annulus FEM (vs exact eigenmode) ===")
print(f"  levels cellSize = {SPATIAL_2D_CELLSIZE}; BE Richardson pair "
      f"({DT_2D_SPATIAL:g}, {DT_2D_SPATIAL_HALVED:g}) at every level")
for pair in EIGENPAIRS:
    runs_a = [run_2d(h, DT_2D_SPATIAL, pair) for h in SPATIAL_2D_CELLSIZE]
    runs_b = [run_2d(h, DT_2D_SPATIAL_HALVED, pair) for h in SPATIAL_2D_CELLSIZE]
    exact_final = exact_profiles(r_common, FINAL_TIME, k_wavenumber, pair)

    metrics_raw, metrics_re, time_content, rates_max, rates_rmse = [], [], [], [], []
    for ra, rb in zip(runs_a, runs_b):
        metrics_raw.append(error_metrics(ra["profiles"], exact_final))
        prof_re = 2.0 * rb["profiles"] - ra["profiles"]
        metrics_re.append(error_metrics(prof_re, exact_final))
        time_content.append(error_metrics(ra["profiles"], rb["profiles"]))
    for i in range(len(runs_a) - 1):
        ratio = runs_a[i]["h_eff"] / runs_a[i + 1]["h_eff"]
        rates_max.append(
            [observed_rate(metrics_re[i]["max_abs"][c], metrics_re[i + 1]["max_abs"][c], ratio)
             for c in range(3)]
        )
        rates_rmse.append(
            [observed_rate(metrics_re[i]["rmse"][c], metrics_re[i + 1]["rmse"][c], ratio)
             for c in range(3)]
        )

    floor = richardson_time_floor(pair, k_wavenumber, DT_2D_SPATIAL)
    label = f"lambda={pair['lambda']:g}"
    print(f"\n  eigenmode {label}, v = {pair['eigenvector']}")
    print("    level |   h_eff  | ratio | Mg max    Fe max    Ca max | Mg rate  Fe rate  Ca rate")
    for i, ra in enumerate(runs_a):
        rate_row = (
            "  n/a"
            if i == 0
            else " ".join(f"{p:7.3f}" for p in rates_max[i - 1])
        )
        ratio_row = "  n/a" if i == 0 else f"{runs_a[i-1]['h_eff']/ra['h_eff']:5.3f}"
        print(
            f"    h={ra['cell_size']:4.2f} | {ra['h_eff']:.5f} | {ratio_row} | "
            + " ".join(f"{m:.3e}" for m in metrics_re[i]["max_abs"])
            + " | " + rate_row
        )
    print("    raw errors at dt (before Richardson), finest level: "
          + " ".join(f"{m:.3e}" for m in metrics_raw[-1]["max_abs"]))
    print(f"    Richardson time floor (analytic, dt={DT_2D_SPATIAL:g}): "
          + " ".join(f"{v:.2e}" for v in floor))
    print("    ||X(dt)-X(dt/2)|| time content by level (max, Mg/Fe/Ca):")
    for ra, tc in zip(runs_a, time_content):
        print(
            f"      h={ra['cell_size']:4.2f}: "
            + " ".join(f"{v:.2e}" for v in tc["max_abs"])
            + f"  (spatial err: "
            + " ".join(f"{m:.2e}" for m in metrics_re[runs_a.index(ra)]["max_abs"])
            + ")"
        )
    fin = runs_a[-1]
    print(f"    checks finest level: SNES all converged={fin['snes_all_converged']}, "
          f"max SNES its={fin['snes_max_iterations']}, "
          f"max linear its={fin['linear_max_iterations']}, "
          f"mass drift max={max(fin['mass_drift']):.2e}, "
          f"closure={fin['closure_residual']:.2e}")
    print("    2D angular RMS spread max (Mg/Fe/Ca) finest level: "
          + " ".join(f"{v:.2e}" for v in fin["angular_spread_max"]))
    study_2d_spatial["per_eigenmode"].append(
        {
            "eigenmode": label,
            "lambda": pair["lambda"],
            "levels_dt": [
                {k: v for k, v in ra.items() if k not in ("profiles",)}
                | {"errors_vs_exact_raw": metrics_raw[i],
                   "errors_vs_exact_richardson": metrics_re[i],
                   "time_content_vs_dt_halved": time_content[i]}
                for i, ra in enumerate(runs_a)
            ],
            "levels_dt_halved": [
                {k: v for k, v in rb.items() if k not in ("profiles",)}
                for rb in runs_b
            ],
            "observed_rates_max_richardson": rates_max,
            "observed_rates_rmse_richardson": rates_rmse,
            "h_eff_ratios": [
                runs_a[i]["h_eff"] / runs_a[i + 1]["h_eff"]
                for i in range(len(runs_a) - 1)
            ],
            "richardson_time_floor": [float(v) for v in floor],
        }
    )
study_2d_spatial["seconds"] = time.perf_counter() - t_study
print(f"\n  [2D spatial study wall time: {study_2d_spatial['seconds']:.1f} s]")

# %%
# TEMPORAL STUDY - 1D code (fixed dr = 0.0025; same-mesh finer-step reference).
study_1d_temporal = {"per_eigenmode": [], "seconds": 0.0}
t_study = time.perf_counter()
print("\n=== TEMPORAL CONVERGENCE: 1D cylindrical FV (fixed N=201, vs dt-ref) ===")
print(f"  dt ladder {TEMPORAL_DT_LADDER} vs reference dt_ref={TEMPORAL_DT_REF:g} "
      f"({int(round(FINAL_TIME / TEMPORAL_DT_REF))} steps, same mesh)")
for pair in EIGENPAIRS:
    ref = run_1d(TEMPORAL_1D_N, TEMPORAL_DT_REF, pair)
    ref_common = profiles_1d_on_common(ref)
    ladder_runs = [run_1d(TEMPORAL_1D_N, dt, pair) for dt in TEMPORAL_DT_LADDER]

    rows, rates_max, rates_rmse, const_c = [], [], [], []
    for run in ladder_runs:
        prof = profiles_1d_on_common(run)
        met = error_metrics(prof, ref_common)
        rows.append({"dt": run["dt"], "errors_vs_ref": met, "seconds": run["seconds"]})
        const_c.append(
            [met["max_abs"][c] / (run["dt"] - TEMPORAL_DT_REF) for c in range(3)]
        )
    for i in range(len(ladder_runs) - 1):
        ratio = ladder_runs[i]["dt"] / ladder_runs[i + 1]["dt"]
        rates_max.append(
            [observed_rate(rows[i]["errors_vs_ref"]["max_abs"][c],
                           rows[i + 1]["errors_vs_ref"]["max_abs"][c], ratio)
             for c in range(3)]
        )
        rates_rmse.append(
            [observed_rate(rows[i]["errors_vs_ref"]["rmse"][c],
                           rows[i + 1]["errors_vs_ref"]["rmse"][c], ratio)
             for c in range(3)]
        )

    label = f"lambda={pair['lambda']:g}"
    print(f"\n  eigenmode {label}")
    print("      dt     | Mg max    Fe max    Ca max | Mg rate  Fe rate  Ca rate | Mg C      Fe C      Ca C")
    for i, row in enumerate(rows):
        rate_row = (
            "  n/a"
            if i == 0
            else " ".join(f"{p:7.3f}" for p in rates_max[i - 1])
        )
        print(
            f"    {row['dt']:8.6f} | "
            + " ".join(f"{m:.3e}" for m in row["errors_vs_ref"]["max_abs"])
            + " | " + rate_row
            + " | " + " ".join(f"{c:.3e}" for c in const_c[i])
        )
    print("    (C = e/(dt - dt_ref); constant C exposes clean first-order BE;")
    print("     raw rates drift upward at the finest dt because dt -> dt_ref)")
    print(f"    checks: mass drift max={max(ladder_runs[-1]['mass_drift']):.2e}, "
          f"closure={ladder_runs[-1]['closure_residual']:.2e}")
    study_1d_temporal["per_eigenmode"].append(
        {
            "eigenmode": label,
            "lambda": pair["lambda"],
            "dt_ref": TEMPORAL_DT_REF,
            "n_ref_steps": int(round(FINAL_TIME / TEMPORAL_DT_REF)),
            "rows": rows,
            "observed_rates_max": rates_max,
            "observed_rates_rmse": rates_rmse,
            "dt_ratios": [
                ladder_runs[i]["dt"] / ladder_runs[i + 1]["dt"]
                for i in range(len(ladder_runs) - 1)
            ],
            "constant_C_e_over_dt_minus_ref": const_c,
        }
    )
study_1d_temporal["seconds"] = time.perf_counter() - t_study
print(f"\n  [1D temporal study wall time: {study_1d_temporal['seconds']:.1f} s]")

# %%
# TEMPORAL STUDY - 2D code (fixed cellSize 0.02; same-mesh finer-step reference).
study_2d_temporal = {"per_eigenmode": [], "seconds": 0.0}
t_study = time.perf_counter()
print("\n=== TEMPORAL CONVERGENCE: 2D annulus FEM (fixed cellSize=0.02, vs dt-ref) ===")
print(f"  dt ladder {TEMPORAL_DT_LADDER} vs reference dt_ref={TEMPORAL_DT_REF:g} "
      f"({int(round(FINAL_TIME / TEMPORAL_DT_REF))} steps, same mesh)")
for pair in EIGENPAIRS:
    ref = run_2d(TEMPORAL_2D_CELLSIZE, TEMPORAL_DT_REF, pair)
    ladder_runs = [run_2d(TEMPORAL_2D_CELLSIZE, dt, pair) for dt in TEMPORAL_DT_LADDER]

    rows, rates_max, rates_rmse, const_c = [], [], [], []
    for run in ladder_runs:
        met = error_metrics(run["profiles"], ref["profiles"])
        rows.append(
            {"dt": run["dt"], "errors_vs_ref": met, "seconds": run["seconds"],
             "snes_all_converged": run["snes_all_converged"],
             "snes_max_iterations": run["snes_max_iterations"]}
        )
        const_c.append(
            [met["max_abs"][c] / (run["dt"] - TEMPORAL_DT_REF) for c in range(3)]
        )
    for i in range(len(ladder_runs) - 1):
        ratio = ladder_runs[i]["dt"] / ladder_runs[i + 1]["dt"]
        rates_max.append(
            [observed_rate(rows[i]["errors_vs_ref"]["max_abs"][c],
                           rows[i + 1]["errors_vs_ref"]["max_abs"][c], ratio)
             for c in range(3)]
        )
        rates_rmse.append(
            [observed_rate(rows[i]["errors_vs_ref"]["rmse"][c],
                           rows[i + 1]["errors_vs_ref"]["rmse"][c], ratio)
             for c in range(3)]
        )

    label = f"lambda={pair['lambda']:g}"
    print(f"\n  eigenmode {label}")
    print("      dt     | Mg max    Fe max    Ca max | Mg rate  Fe rate  Ca rate | Mg C      Fe C      Ca C")
    for i, row in enumerate(rows):
        rate_row = (
            "  n/a"
            if i == 0
            else " ".join(f"{p:7.3f}" for p in rates_max[i - 1])
        )
        print(
            f"    {row['dt']:8.6f} | "
            + " ".join(f"{m:.3e}" for m in row["errors_vs_ref"]["max_abs"])
            + " | " + rate_row
            + " | " + " ".join(f"{c:.3e}" for c in const_c[i])
        )
    print(f"    reference run: SNES all converged={ref['snes_all_converged']}, "
          f"mass drift max={max(ref['mass_drift']):.2e}, "
          f"closure={ref['closure_residual']:.2e}")
    print("    2D angular RMS spread max (Mg/Fe/Ca) at dt-ref: "
          + " ".join(f"{v:.2e}" for v in ref["angular_spread_max"]))
    study_2d_temporal["per_eigenmode"].append(
        {
            "eigenmode": label,
            "lambda": pair["lambda"],
            "dt_ref": TEMPORAL_DT_REF,
            "n_ref_steps": int(round(FINAL_TIME / TEMPORAL_DT_REF)),
            "ref_seconds": ref["seconds"],
            "ref_snes_all_converged": ref["snes_all_converged"],
            "rows": rows,
            "observed_rates_max": rates_max,
            "observed_rates_rmse": rates_rmse,
            "dt_ratios": [
                ladder_runs[i]["dt"] / ladder_runs[i + 1]["dt"]
                for i in range(len(ladder_runs) - 1)
            ],
            "constant_C_e_over_dt_minus_ref": const_c,
        }
    )
study_2d_temporal["seconds"] = time.perf_counter() - t_study
print(f"\n  [2D temporal study wall time: {study_2d_temporal['seconds']:.1f} s]")

# %%
# SECONDARY cross-check: 1D vs 2D at each code's finest spatial setting.
cross_check = []
print("\n=== SECONDARY: 1D vs 2D at finest spatial settings (not the rate metric) ===")
for pair in EIGENPAIRS:
    a1 = run_1d(SPATIAL_1D_N[-1], DT_1D_SPATIAL, pair)
    b1 = run_1d(SPATIAL_1D_N[-1], DT_1D_SPATIAL_HALVED, pair)
    ra = run_2d(SPATIAL_2D_CELLSIZE[-1], DT_2D_SPATIAL, pair)
    rb = run_2d(SPATIAL_2D_CELLSIZE[-1], DT_2D_SPATIAL_HALVED, pair)
    prof_1d = 2.0 * profiles_1d_on_common(b1) - profiles_1d_on_common(a1)
    prof_2d = 2.0 * rb["profiles"] - ra["profiles"]
    diff = error_metrics(prof_1d, prof_2d)
    exact_final = exact_profiles(r_common, FINAL_TIME, k_wavenumber, pair)
    err_1d = error_metrics(prof_1d, exact_final)
    err_2d = error_metrics(prof_2d, exact_final)
    label = f"lambda={pair['lambda']:g}"
    print(f"  eigenmode {label}: max|1D-2D| = "
          + " ".join(f"{v:.3e}" for v in diff["max_abs"])
          + " (Mg/Fe/Ca); 1D vs exact max = "
          + " ".join(f"{v:.3e}" for v in err_1d["max_abs"])
          + "; 2D vs exact max = "
          + " ".join(f"{v:.3e}" for v in err_2d["max_abs"]))
    cross_check.append(
        {"eigenmode": label, "diff_1d_2d": diff, "err_1d_vs_exact": err_1d,
         "err_2d_vs_exact": err_2d}
    )

# %%
# Machine-readable summary and final accounting.
total_start_to_here = (
    study_1d_spatial["seconds"] + study_2d_spatial["seconds"]
    + study_1d_temporal["seconds"] + study_2d_temporal["seconds"]
)
summary = {
    "meta": {
        "script": str(Path(__file__).name),
        "command": (
            "source /Users/benknight/software/venv/uw3-dev/bin/activate && "
            "python examples/convergence_1d_2d_cylindrical_multicomponent.py"
        ),
        "components": COMPONENTS,
        "diffusion_matrix": DIFFUSION_MATRIX.tolist(),
        "eigenvalues": [p["lambda"] for p in EIGENPAIRS],
        "eigenvectors": [p["eigenvector"].tolist() for p in EIGENPAIRS],
        "k_wavenumber": k_wavenumber,
        "final_time": FINAL_TIME,
        "spatial": {
            "1d_n_levels": SPATIAL_1D_N,
            "1d_dt_richardson_pair": [DT_1D_SPATIAL, DT_1D_SPATIAL_HALVED],
            "2d_cellsizes": SPATIAL_2D_CELLSIZE,
            "2d_dt_richardson_pair": [DT_2D_SPATIAL, DT_2D_SPATIAL_HALVED],
        },
        "temporal": {
            "1d_n": TEMPORAL_1D_N,
            "2d_cellsize": TEMPORAL_2D_CELLSIZE,
            "dt_ladder": TEMPORAL_DT_LADDER,
            "dt_ref": TEMPORAL_DT_REF,
        },
        "study_wall_seconds": total_start_to_here,
        "underworld3_version": getattr(uw, "__version__", "unknown"),
    },
    "spatial_1d": study_1d_spatial,
    "spatial_2d": study_2d_spatial,
    "temporal_1d": study_1d_temporal,
    "temporal_2d": study_2d_temporal,
    "cross_check_1d_vs_2d_finest": cross_check,
}
out_path = Path(__file__).resolve().parents[1] / "examples" / "output"
out_path.mkdir(parents=True, exist_ok=True)
json_path = out_path / "convergence_study_summary.json"


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


with open(json_path, "w") as fh:
    json.dump(summary, fh, indent=2, default=_json_default)

print("\n=== STUDY ACCOUNTING ===")
print(f"  1D spatial : {study_1d_spatial['seconds']:8.1f} s")
print(f"  2D spatial : {study_2d_spatial['seconds']:8.1f} s")
print(f"  1D temporal: {study_1d_temporal['seconds']:8.1f} s")
print(f"  2D temporal: {study_2d_temporal['seconds']:8.1f} s")
print(f"  study total: {total_start_to_here:8.1f} s")
print(f"  summary JSON: {json_path}")
print("\nScope notes:")
print("  - Both real eigenmodes of D tested (lambda = 1.1 and 0.4).")
print("  - 2D observed rates use actual h_eff = sqrt(area/n_cells) ratios, not")
print("    nominal cellSize ratios; 1D rates use exact dr ratios.")
print("  - Temporal observed rates vs a finite dt_ref are upward-biased as")
print("    dt -> dt_ref (e ~ C(dt - dt_ref)); the reported C table makes the")
print("    first-order BE behaviour explicit.")
print("  - Spatial errors (both codes) include sampling/averaging + discretisation")
print("    error; the Richardson pair removes only the O(dt) part, with residual")
print("    floors reported analytically per study.")
