# %%
"""Matched coupled multicomponent diffusion: 1D cylindrical FV vs 2D UW3 FEM.

Both solvers advance the same two independent components (Mg, Fe) with Ca
implicit by closure, and the same nonsymmetric, non-diagonal reduced diffusion
matrix. This is a true cross-diffusion comparison, not separate scalar
DiffusionModel runs. Both use a radially symmetric Neumann eigenmode in an
annulus and Backward Euler with a shared fixed timestep.

Run in the documented UW3 environment from the repository root:
    python examples/compare_1d_2d_cylindrical_multicomponent.py
"""

# %% [markdown]
# Physics and comparison design
# -----------------------------
# The 1D solver advances `dX_i/dt = (1/r) d/dr[r sum_j D_ij dX_j/dr]` using a
# conservative vertex-centred finite-volume discretisation. The 2D
# `MulticomponentDiffusionModel` uses the same reduced component flux matrix
# `J_i = -sum_j D_ij grad(X_j)` on a Cartesian annulus; for radial fields its
# continuum operator is identical. The matrix is genuinely coupled and
# nonsymmetric: `D = [[1.0, 0.3], [0.2, 0.5]]`. The third component Ca is
# computed as `1 - Mg - Fe` in both solvers.
#
# Both runs use natural/homogeneous-Neumann boundaries, the identical Bessel
# eigenmode initial condition, annular radii, matrix, duration and BE timestep.
# A mesh-refined 1D radial grid and circumferentially averaged Q2 values are
# compared on common radii; all three component profiles are reported against
# the exact continuum eigenmode. Remaining errors include spatial discretisation
# and angular mesh anisotropy.

# %%
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

import underworld3 as uw

from UWDiffusion import MulticomponentDiffusion1D, MulticomponentDiffusionModel

# %%
# Matched physical and numerical parameters.
R_INNER = 0.5
R_OUTER = 1.0
DIFFUSION_MATRIX = np.array([[1.0, 0.3], [0.2, 0.5]])
COMPONENTS = ["Mg", "Fe", "Ca"]  # Ca is the dependent closure component
FINAL_TIME = 0.02
DT = 0.001
N_1D = 51
MESH_CELL_SIZE = 0.025
N_COMMON_RADII = 101
N_ANGLES = 64
MEAN_INDEPENDENT = np.array([0.5, 0.25])
MODE_AMPLITUDE = 0.15
N_STEPS = int(round(FINAL_TIME / DT))


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


def eigenmode_parameters() -> tuple[float, np.ndarray]:
    """Return the dominant eigenvalue/eigenvector of the shared D matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(DIFFUSION_MATRIX)
    idx = int(np.argmax(eigenvalues.real))
    eigenvalue = float(eigenvalues[idx].real)
    eigenvector = eigenvectors[:, idx].real
    eigenvector /= np.linalg.norm(eigenvector)
    if eigenvector[0] < 0:
        eigenvector *= -1
    return eigenvalue, eigenvector


def exact_profiles(r: np.ndarray, time: float, k: float) -> np.ndarray:
    """Return exact [Mg, Fe, Ca] profiles for the coupled diffusion eigenmode."""
    eigenvalue, eigenvector = eigenmode_parameters()
    decay = np.exp(-eigenvalue * k**2 * time)
    perturbation = MODE_AMPLITUDE * radial_mode(r, k) * decay
    independent = MEAN_INDEPENDENT[:, None] + eigenvector[:, None] * perturbation
    dependent = 1.0 - independent.sum(axis=0)
    return np.vstack((independent, dependent))


k = neumann_wavenumber(R_INNER, R_OUTER)
eigenvalue, eigenvector = eigenmode_parameters()
grid_1d = np.linspace(R_INNER, R_OUTER, N_1D)
independent_initial_1d = (
    MEAN_INDEPENDENT[:, None]
    + MODE_AMPLITUDE * eigenvector[:, None] * radial_mode(grid_1d, k)
)
initial_1d = np.vstack(
    (independent_initial_1d, 1.0 - independent_initial_1d.sum(axis=0))
).T

# %%
# 1D cylindrical FV: shared reduced matrix, coupled independent components,
# Ca closure and explicit homogeneous-Neumann conditions at both rims.
model_1d = MulticomponentDiffusion1D(
    grid=grid_1d,
    diffusivity=DIFFUSION_MATRIX,
    geometry="cylindrical",
    components=COMPONENTS,
    bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    initial_composition=initial_1d,
)
result_1d = model_1d.run_simulation(dt=DT, total_time=FINAL_TIME)

# %%
# 2D coupled UW3 model on a full annulus. Leaving both rims unconstrained gives
# natural zero-normal-flux BCs for the matrix-valued component fluxes.
mesh_2d = uw.meshing.Annulus(
    radiusOuter=R_OUTER,
    radiusInner=R_INNER,
    cellSize=MESH_CELL_SIZE,
    qdegree=2,
)
d_symbols = sp.Matrix(
    [
        [sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")],
        [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")],
    ]
)
model_2d = MulticomponentDiffusionModel(
    COMPONENTS, mesh_2d, diffusion_matrix=d_symbols, degree=2, order=1
)
model_2d.set_diffusion_values(
    {
        "D_MgMg": DIFFUSION_MATRIX[0, 0],
        "D_MgFe": DIFFUSION_MATRIX[0, 1],
        "D_FeMg": DIFFUSION_MATRIX[1, 0],
        "D_FeFe": DIFFUSION_MATRIX[1, 1],
    }
)
for solver in model_2d.solvers:
    solver.petsc_options["snes_rtol"] = 1e-12
    solver.petsc_options["snes_atol"] = 1e-6
    solver.petsc_options["snes_converged_reason"] = None

coords_2d = model_2d.mesh_vars[0].coords
r_nodes_2d = np.sqrt(coords_2d[:, 0] ** 2 + coords_2d[:, 1] ** 2)
initial_2d = exact_profiles(r_nodes_2d, 0.0, k)
with mesh_2d.access(*model_2d.mesh_vars):
    for idx, mesh_var in enumerate(model_2d.mesh_vars):
        mesh_var.data[:, 0] = initial_2d[idx]

model_2d.run_simulation(duration=FINAL_TIME, min_dt=DT, max_dt=DT)
assert np.isclose(model_1d.time, FINAL_TIME, rtol=0.0, atol=1e-12)
assert np.isclose(model_2d.current_time, FINAL_TIME, rtol=0.0, atol=1e-12)
assert model_2d.step == N_STEPS

# %%
# Compare all component radial profiles at identical radii, avoiding just-outside
# boundary queries caused by floating-point roundoff on the curved mesh.
r_common = np.linspace(R_INNER + 1e-10, R_OUTER - 1e-10, N_COMMON_RADII)
angles = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
sample_xy = np.column_stack(
    (
        (r_common[:, None] * np.cos(angles)).ravel(),
        (r_common[:, None] * np.sin(angles)).ravel(),
    )
)
profiles_2d_samples = np.stack(
    [
        uw.function.evaluate(mesh_var.sym[0], sample_xy).reshape(
            N_COMMON_RADII, N_ANGLES
        )
        for mesh_var in model_2d.mesh_vars
    ]
)
profiles_2d = np.mean(profiles_2d_samples, axis=2)
angular_spread_2d = np.sqrt(
    np.mean((profiles_2d_samples - profiles_2d[:, :, None]) ** 2, axis=2)
)
profiles_1d = np.vstack(
    [
        np.interp(r_common, result_1d.r, result_1d.get_profile(name, -1))
        for name in COMPONENTS
    ]
)
profiles_exact = exact_profiles(r_common, FINAL_TIME, k)

diff_1d_2d = profiles_1d - profiles_2d
err_1d = profiles_1d - profiles_exact
err_2d = profiles_2d - profiles_exact
rmse = lambda values: float(np.sqrt(np.mean(values**2)))
max_diff_1d_2d = np.max(np.abs(diff_1d_2d), axis=1)
max_err_1d = np.max(np.abs(err_1d), axis=1)
max_err_2d = np.max(np.abs(err_2d), axis=1)

# Keep the script useful as an executable verification, not only a plotting
# aid. Thresholds allow the observed FV/FEM and angular discretisation errors
# while detecting wrong coupling, closure, geometry, or timestep physics.
assert np.all(max_diff_1d_2d < 1e-3), max_diff_1d_2d
assert np.all(max_err_1d < 2e-3), max_err_1d
assert np.all(max_err_2d < 2e-3), max_err_2d
assert np.max(np.abs(profiles_1d.sum(axis=0) - 1.0)) < 1e-12
assert np.max(np.abs(profiles_2d.sum(axis=0) - 1.0)) < 1e-12

# %%
print("Direct coupled 1D cylindrical FV / 2D annulus FEM comparison")
print(f"  components: {COMPONENTS} (Ca closure); D = {DIFFUSION_MATRIX.tolist()}")
print(f"  radii: [{R_INNER:g}, {R_OUTER:g}]")
print(f"  time: {FINAL_TIME:g}, dt: {DT:g}, steps: {N_STEPS} (both, Backward Euler)")
print(f"  1D vertices: {N_1D} (dr={grid_1d[1] - grid_1d[0]:.4g}); 2D cellSize: {MESH_CELL_SIZE:g}, Q2")
print(f"  common profile samples: {N_COMMON_RADII} radii x {N_ANGLES} angular samples")
print(f"  coupled eigenvalue: {eigenvalue:.12g}; eigenvector: {eigenvector}; k: {k:.12g}")
for idx, name in enumerate(COMPONENTS):
    print(
        f"  {name}: 1D vs 2D max|difference|={max_diff_1d_2d[idx]:.6e}, "
        f"RMSE={rmse(diff_1d_2d[idx]):.6e}; "
        f"1D vs exact max={max_err_1d[idx]:.6e}, RMSE={rmse(err_1d[idx]):.6e}; "
        f"2D vs exact max={max_err_2d[idx]:.6e}, RMSE={rmse(err_2d[idx]):.6e}"
    )
print(
    "  2D circumferential RMS spread max by component: "
    f"{np.max(angular_spread_2d, axis=1)}"
)
print(f"  final times: {model_1d.time:.12g} (1D), {model_2d.current_time:.12g} (2D)")
