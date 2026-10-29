# %%
"""Direct radial diffusion comparison: 1D cylindrical FV vs 2D UW3 FEM.

Both models solve the same scalar constant-D diffusion problem on the annulus
0.5 <= r <= 1.0 with zero flux at both rims. The 1D model uses its cylindrical
geometry factor; the 2D model uses a Cartesian annulus and a radially symmetric
field. A radial Neumann Bessel mode supplies a known transient solution.

Run in the documented UW3 environment from the repository root:
    python examples/compare_1d_2d_cylindrical.py
"""

# %% [markdown]
# Physics and comparison design
# -----------------------------
# The 1D solver advances `dC/dt = (1/r) d/dr(r D dC/dr)` (`alpha=1`), using a
# conservative vertex-centred finite-volume discretisation and Backward Euler.
# `DiffusionModel` advances `dC/dt = div(D grad(C))` on a 2D Cartesian annulus,
# with continuous Q2 elements and first-order (Backward Euler) time stepping.
# For a radial field, `div(D grad(C)) = (1/r) d/dr(r D dC/dr)`, so the continuum
# equations agree. The 1D endpoints have explicit homogeneous Neumann BCs; the
# 2D annulus rims are left unconstrained, hence natural zero-normal-flux BCs.
#
# Both runs use the same annular radii, D, initial profile, duration, and fixed
# timestep. The 1D grid spacing is chosen to be close to the Q2 nodal spacing
# on the 2D mesh. The 2D field is sampled on a circle at many angles and
# circumferentially averaged before comparison at common radii. Differences
# therefore include the distinct spatial discretisations (and residual mesh
# anisotropy), not a change in physics or time integration.

# %%
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

import underworld3 as uw

from UWDiffusion import DiffusionModel, MulticomponentDiffusion1D

# %%
# Matched physical and numerical parameters.
R_INNER = 0.5
R_OUTER = 1.0
DIFFUSIVITY = 1.0
FINAL_TIME = 0.02
DT = 0.001
N_1D = 26  # dr = 0.02, comparable to Q2 nodal spacing for 0.04 mesh cells
MESH_CELL_SIZE = 0.04
N_COMMON_RADII = 101
N_ANGLES = 64
MEAN_CONCENTRATION = 0.5
MODE_AMPLITUDE = 0.25
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


k = neumann_wavenumber(R_INNER, R_OUTER)
grid_1d = np.linspace(R_INNER, R_OUTER, N_1D)
initial_1d = MEAN_CONCENTRATION + MODE_AMPLITUDE * radial_mode(grid_1d, k)

# %%
# 1D cylindrical finite-volume model. One field is used as a scalar tracer,
# avoiding any difference in closure treatment between the two solvers.
model_1d = MulticomponentDiffusion1D(
    grid=grid_1d,
    diffusivity=DIFFUSIVITY,
    geometry="cylindrical",
    components=1,
    bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    initial_composition=initial_1d,
)
result_1d = model_1d.run_simulation(dt=DT, total_time=FINAL_TIME)

# %%
# 2D finite-element model on a full annulus. No rim BC calls are made: the
# natural finite-element boundary condition is zero normal diffusive flux.
mesh_2d = uw.meshing.Annulus(
    radiusOuter=R_OUTER,
    radiusInner=R_INNER,
    cellSize=MESH_CELL_SIZE,
    qdegree=2,
)
model_2d = DiffusionModel("C_compare", mesh_2d, degree=2, order=1)
model_2d.diffusivity = DIFFUSIVITY
coords_2d = model_2d.mesh_var.coords
r_nodes_2d = np.sqrt(coords_2d[:, 0] ** 2 + coords_2d[:, 1] ** 2)
initial_2d = MEAN_CONCENTRATION + MODE_AMPLITUDE * radial_mode(r_nodes_2d, k)
with mesh_2d.access(model_2d.mesh_var):
    model_2d.mesh_var.data[:, 0] = initial_2d

model_2d.run_simulation(
    duration=FINAL_TIME,
    min_dt=DT,
    max_dt=DT,
)
assert np.isclose(model_1d.time, FINAL_TIME, rtol=0.0, atol=1e-12)
assert np.isclose(model_2d.current_time, FINAL_TIME, rtol=0.0, atol=1e-12)
assert model_2d.step == N_STEPS

# %%
# Compare at identical radii, away from the boundary by a tiny margin to avoid
# querying just outside the curved mesh due to floating-point roundoff.
r_common = np.linspace(R_INNER + 1e-10, R_OUTER - 1e-10, N_COMMON_RADII)
angles = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
sample_xy = np.column_stack(
    (
        (r_common[:, None] * np.cos(angles)).ravel(),
        (r_common[:, None] * np.sin(angles)).ravel(),
    )
)
samples_2d = uw.function.evaluate(model_2d.mesh_var.sym[0], sample_xy).reshape(
    N_COMMON_RADII, N_ANGLES
)
profile_2d = np.mean(samples_2d, axis=1)
angular_spread_2d = np.sqrt(np.mean((samples_2d - profile_2d[:, None]) ** 2, axis=1))
profile_1d = np.interp(r_common, result_1d.r, result_1d.get_profile(0, -1))

# Exact continuum solution of the same homogeneous-Neumann eigenmode.
decay = np.exp(-DIFFUSIVITY * k**2 * FINAL_TIME)
profile_exact = MEAN_CONCENTRATION + MODE_AMPLITUDE * radial_mode(r_common, k) * decay

diff_1d_2d = profile_1d - profile_2d
err_1d = profile_1d - profile_exact
err_2d = profile_2d - profile_exact
rmse = lambda values: float(np.sqrt(np.mean(values**2)))

# %%
print("Direct 1D cylindrical FV / 2D annulus FEM comparison")
print(f"  radii: [{R_INNER:g}, {R_OUTER:g}], D: {DIFFUSIVITY:g}")
print(f"  time: {FINAL_TIME:g}, dt: {DT:g}, steps: {N_STEPS} (both)")
print(f"  1D vertices: {N_1D} (dr={grid_1d[1] - grid_1d[0]:.4g}); 2D cellSize: {MESH_CELL_SIZE:g}, Q2")
print(f"  common profile samples: {N_COMMON_RADII} radii x {N_ANGLES} angular samples")
print(f"  first Neumann eigen-wavenumber k: {k:.12g}; analytic modal decay: {decay:.8g}")
print(f"  1D vs 2D: max|difference|={np.max(np.abs(diff_1d_2d)):.6e}, RMSE={rmse(diff_1d_2d):.6e}")
print(f"  1D vs analytic: max|error|={np.max(np.abs(err_1d)):.6e}, RMSE={rmse(err_1d):.6e}")
print(f"  2D vs analytic: max|error|={np.max(np.abs(err_2d)):.6e}, RMSE={rmse(err_2d):.6e}")
print(f"  2D circumferential RMS spread: max={np.max(angular_spread_2d):.6e}, mean={np.mean(angular_spread_2d):.6e}")
print(f"  final times: {model_1d.time:.12g} (1D), {model_2d.current_time:.12g} (2D)")
