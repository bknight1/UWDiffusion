"""Fast exact-solution verification for the 2D UW3-based diffusion models.

These tests exercise :class:`UWDiffusion.DiffusionModel` and
:class:`UWDiffusion.MulticomponentDiffusionModel` (the Underworld3/PETSc path
used by the tutorials) on small meshes where the exact solution is known, so
each run finishes in seconds.

**Zero-flux boundary verification is done with natural (unconstrained)
boundaries.**  In the finite-element weak form, a boundary with no imposed
condition is the *natural* boundary: it contributes no boundary term, which
is exactly zero normal diffusive flux.  No Dirichlet value is ever used to
substantiate a zero-flux claim here (a prescribed rim value constrains the
solution, it does not verify flux-free boundaries).  The installed UW3
``add_neumann_bc`` is unreliable, so the tests deliberately leave every
natural boundary completely unconstrained (no ``add_dirichlet_bc`` and no
``add_neumann_bc`` calls) and verify the resulting zero-flux behaviour
against exact solutions.

Cartesian tests (``uw.meshing.StructuredQuadBox`` on the unit square, Q2
elements, backward Euler ``order=1``):

1. **Constant uniform state, natural boundaries** -- C = 1 is an exact fixed
   point of the discrete problem (K * 1 = 0 for the FE stiffness matrix with
   natural boundaries), so it must be preserved to solver-tolerance
   precision (measured drift ~4e-11).  Any spurious boundary flux would
   immediately drain the interior.

2. **Zero-flux Laplacian eigenmode** --

        C(x, y, t) = cos(pi x) cos(pi y) exp(-2 kappa pi^2 t)

    on [0, 1]^2 with *no* boundary conditions.  dC/dx = -pi sin(pi x) cos(pi y)
    vanishes on the Left/Right walls and dC/dy vanishes on Top/Bottom, so the
    exact solution satisfies the natural (zero-flux) condition on all four
    sides.  With kappa = 1, T = 0.05 and dt = 1e-3 (50 backward-Euler steps)
    the temporal amplitude error is bounded by ~ (1/2) lambda^2 T dt
    (lambda = 2 pi^2); measured amplitude error is 3.6e-3.  The field must
    also conserve mass: the mode integrates to zero and the normal flux is
    zero on every wall, so the total mass must remain 0 (measured drift
    < 1e-8 via grid sampling of the FE field).

Annulus tests (``uw.meshing.Annulus``, radii 0.5-1.0, natural boundaries on
both rims; ``Lower`` = inner rim, ``Upper`` = outer rim):

3. **Constant uniform state** -- same fixed-point argument as test 1 on the
    gmsh-generated unstructured mesh (measured drift ~3e-10).

4. **Radial Neumann-compatible transient eigenmode** -- the radial
    eigenfunction of the planar 2D Laplacian with u'(r_i) = u'(r_o) = 0:

        u(r) = Y_1(k r_i) J_0(k r) - J_1(k r_i) Y_0(k r),
        C(r, t) = u(r)/max|u| * exp(-kappa k^2 t),

    where k is the first positive root of
    J_1(k r_i) Y_1(k r_o) - J_1(k r_o) Y_1(k r_i) = 0
    (k = 6.3931567616... for r_i = 0.5, r_o = 1.0, found by bisection).
    The mode has exactly zero normal gradient on both rims, so it is
    compatible with the natural boundaries.  Evolved with kappa = 1,
    T = 0.03, dt = 5e-4 (60 backward-Euler steps) at cellSize 0.05:
    measured amplitude error 5.0e-3 and rescaled shape error 1.7e-2.

Multicomponent tests (``MulticomponentDiffusionModel``):

5. **Cartesian coupled cross-diffusion eigenmode (Mg / Fe / Ca)** -- components
    ["Mg", "Fe", "Ca"] with Ca as the implicit closure (Ca = 1 - Mg - Fe)
    and a constant 2x2 reduced diffusion matrix with genuine cross terms

        D = [[1.0, 0.3], [0.2, 0.5]]   (nonsymmetric, SPD)

    whose eigenpairs are lambda_1 = 1.1 with eigenvector (3, 1)/sqrt(10) and
    lambda_2 = 0.4.  Starting from

        x(r, 0) = x0 + a * v_1 * cos(pi x) cos(pi y),  x0 = (0.5, 0.25), a = 0.15,

    the constant part is preserved and the eigenmode part decays at the
    coupled rate 2 pi^2 lambda_1:

        x(r, t) = x0 + a v_1 cos(pi x) cos(pi y) exp(-2 pi^2 lambda_1 t),
        Ca(r, t) = 1 - Mg(r, t) - Fe(r, t).

    The cosine mode has zero normal derivative on all four walls, so every
    component flux J_i = (D v_1)_i grad(f) . n vanishes on the boundary and
    each component's mass is conserved exactly.  Both independent components
    and the closure are compared against the exact coupled decay (measured
    max errors 4.5e-4 / 2.0e-4 / 6.5e-4); the closure identity holds to
    ~1e-16 and per-component masses drift < 1e-8 (tolerance 1e-6).  This
    test fails qualitatively if cross-diffusion is dropped (Mg would then
    decay at rate D_11 = 1.0 instead of lambda_1 = 1.1, a ~5e-3 absolute
    error on a 2e-3 tolerance), if the diffusion matrix is transposed (the
    eigenvector ratio Mg:Fe changes from 3:1 to 2:1), or if the closure is
    not updated.

6. **Annulus coupled cross-diffusion eigenmode (Mg / Fe / Ca)** -- components
    ["Mg", "Fe", "Ca"] with Ca closure on ``uw.meshing.Annulus`` with
    natural unconstrained boundaries on both rims. The radial Bessel mode
    u(r) (with u'(r_i) = u'(r_o) = 0) decays at coupled rate k^2 lambda_1,
    conserving mass and satisfying the closure identity.

No application code is modified by this suite; mesh files are written to the
gitignored ``.meshes/`` cache directory.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

try:
    import sympy as sp

    from scipy.special import j0, j1, y0, y1

    import underworld3 as uw

    from UWDiffusion import DiffusionModel, MulticomponentDiffusionModel
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"Underworld3/MPI/Gmsh environment unavailable: {exc}",
        allow_module_level=True,
    )

PI = np.pi


# ==============================================================================
# Helpers
# ==============================================================================


def _tighten_solver(model) -> None:
    """Tighten SNES tolerances exactly as the tutorial notebooks do.

    The default nonlinear-solver tolerances leave O(1e-6) noise on trivially
    steady problems; the tutorials set snes_rtol=1e-12 / snes_atol=1e-6 for
    the same reason.  Works for both single- and multicomponent models.
    """
    solvers = (
        model.solvers if hasattr(model, "solvers") else [model.diffusion_solver]
    )
    for solver in solvers:
        solver.petsc_options["snes_rtol"] = 1e-12
        solver.petsc_options["snes_atol"] = 1e-6
        solver.petsc_options["snes_converged_reason"] = None


def _set_initial_condition(model: DiffusionModel, expression) -> None:
    """Set the single-component mesh variable from data.

    ``expression`` may be a plain float, a numpy array of nodal values (one
    per entry of ``model.mesh_var.coords``), or a sympy expression of mesh
    coordinates.
    """
    with model.mesh.access(model.mesh_var):
        if isinstance(expression, (int, float)):
            model.mesh_var.data[:, 0] = float(expression)
        elif isinstance(expression, np.ndarray):
            model.mesh_var.data[:, 0] = expression
        else:
            model.mesh_var.data[:, 0] = uw.function.evaluate(
                expression, model.mesh_var.coords
            ).flatten()


def _evaluate(model: DiffusionModel) -> tuple[np.ndarray, np.ndarray]:
    """Return (dof coordinates, nodal values) of the model variable."""
    coords = model.mesh_var.coords
    values = uw.function.evaluate(model.mesh_var.sym[0], coords).flatten()
    return coords, values


def _make_model(mesh, kappa: float = 1.0) -> DiffusionModel:
    """Single-component model with unit non-dimensional scaling."""
    model = DiffusionModel("C", mesh, degree=2, order=1)
    # Plain floats pass through uw.scaling unchanged, so kappa is used as-is.
    model.diffusivity = kappa
    _tighten_solver(model)
    return model


def _grid_mass(mesh_var, n: int = 121) -> float:
    """Integrate the FE field over the unit square by grid sampling.

    Evaluates the FE interpolant on a regular n x n grid and applies the
    trapezoid rule.  For the integrands used here the odd derivatives vanish
    on the walls, so the rule is exact up to round-off / interpolation
    effects well below the asserted tolerances.
    """
    grid = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    values = uw.function.evaluate(mesh_var.sym[0], points)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(trapz_fn(values.reshape(n, n), grid, axis=1), grid))


def _annulus_neumann_wavenumber(r_i: float, r_o: float) -> float:
    """First positive root k of J_1(k r_i) Y_1(k r_o) - J_1(k r_o) Y_1(k r_i).

    This is the solvability condition for a radial Laplacian eigenfunction
    A J_0(kr) + B Y_0(kr) with u'(r_i) = u'(r_o) = 0.  For r_i = 0.5,
    r_o = 1.0 the first root lies in (6, 7); bisected to machine precision.
    """
    def determinant(k: float) -> float:
        return j1(k * r_i) * y1(k * r_o) - j1(k * r_o) * y1(k * r_i)

    lo, hi = 6.0, 7.0
    if not determinant(lo) * determinant(hi) < 0:  # pragma: no cover
        pytest.skip("annulus Neumann wavenumber bracket not found")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if determinant(lo) * determinant(mid) <= 0:
            hi = mid
        else:
            lo = mid
    k = 0.5 * (lo + hi)
    assert abs(determinant(k)) < 1e-10
    return k


def _annulus_neumann_mode(r: np.ndarray, k: float, r_i: float) -> np.ndarray:
    """Radial eigenfunction with u'(r_i) = u'(r_o) = 0, normalised to max|u| = 1.

    u(r) = Y_1(k r_i) J_0(k r) - J_1(k r_i) Y_0(k r) satisfies u'(r_i) = 0
    identically and u'(r_o) = k * [J_1(k r_i) Y_1(k r_o) - J_1(k r_o) Y_1(k r_i)]
    = 0 at the root k.  Its Laplacian is -k^2 u, so
    C(r, t) = u(r) exp(-kappa k^2 t) solves the diffusion equation with zero
    normal flux on both rims.
    """
    u = y1(k * r_i) * j0(k * r) - j1(k * r_i) * y0(k * r)
    r_dense = np.linspace(r_i, 1.0, 20001)
    u_max = float(np.max(np.abs(y1(k * r_i) * j0(k * r_dense) - j1(k * r_i) * y0(k * r_dense))))
    return u / u_max


# ==============================================================================
# 1. Cartesian constant uniform state, natural boundaries (StructuredQuadBox)
# ==============================================================================


def test_cartesian_constant_state_natural_boundaries() -> None:
    """A uniform state must be preserved with unconstrained (natural) walls.

    C = 1 is an exact fixed point of the implicit diffusion step with natural
    boundaries (zero source, zero flux, K * 1 = 0), so the field must remain
    uniform to solver-tolerance precision.  No boundary condition of any kind
    is applied: if the solver imposed any spurious flux at the walls, the
    interior would drain and the drift would grow to O(1).
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(12, 12), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    model = _make_model(mesh, kappa=1.0)
    _set_initial_condition(model, 1.0)

    model.run_simulation(duration=0.03, min_dt=0.01, max_dt=0.01)

    # Three exact 0.01 steps land on the requested duration.
    assert model.step == 3
    assert model.current_time == pytest.approx(0.03, abs=1e-14)

    _, values = _evaluate(model)
    drift = float(np.max(np.abs(values - 1.0)))
    # Measured drift ~4e-11 with tightened SNES tolerances; 1e-8 still leaves
    # two orders of margin while catching any real boundary flux.
    assert drift < 1e-8, f"uniform state drifted by {drift:.3e}"


# ==============================================================================
# 2. Cartesian zero-flux eigenmode, natural boundaries (StructuredQuadBox)
# ==============================================================================


def test_cartesian_cosine_eigenmode_natural_boundaries() -> None:
    """Fundamental zero-flux eigenmode on the unit square.

    Exact solution with kappa = 1 and *no* boundary conditions (natural =
    zero normal flux on all four walls):

        C(x, y, t) = cos(pi x) cos(pi y) exp(-2 pi^2 t)

    since dC/dx vanishes at x = 0, 1 and dC/dy vanishes at y = 0, 1.  The
    simulation runs to T = 0.05 with dt = 1e-3 (50 backward-Euler steps).
    The backward-Euler amplitude of the mode is (1 + lambda dt)^(-T/dt)
    instead of exp(-lambda T), a relative deficit of ~ (1/2) lambda^2 T dt;
    together with the Q2 spatial error the measured amplitude error is
    3.6e-3.  Tolerance 0.02 (5x measured) still separates a correct solve
    from qualitative failures: a frozen field gives 0.39, a 2x-wrong decay
    rate gives ~0.23.
    """
    kappa = 1.0
    lam = 2.0 * kappa * PI**2

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    model = _make_model(mesh, kappa=kappa)
    coords = model.mesh_var.coords
    _set_initial_condition(
        model, np.cos(PI * coords[:, 0]) * np.cos(PI * coords[:, 1])
    )

    T = 0.05
    model.run_simulation(duration=T, min_dt=0.001, max_dt=0.001)
    assert model.step == 50
    assert model.current_time == pytest.approx(T, abs=1e-12)

    coords_out, values = _evaluate(model)

    mode = np.cos(PI * coords_out[:, 0]) * np.cos(PI * coords_out[:, 1])
    exact = mode * np.exp(-lam * model.current_time)
    amp_err = float(np.max(np.abs(values - exact)))
    assert amp_err < 0.02, f"eigenmode amplitude error {amp_err:.3e} >= 0.02"

    # The field must actually have decayed to e^{-lambda T} ~ 0.34 (guards
    # against a frozen field or a spurious sink).
    assert 0.15 < float(np.max(np.abs(values))) < 0.6

    # Shape check: rescaled by the exact decay, the field must still be the
    # eigenfunction.
    shape_err = float(
        np.max(np.abs(values / np.exp(-lam * model.current_time) - mode))
    )
    assert shape_err < 0.04, f"eigenmode shape error {shape_err:.3e} >= 0.04"

    # Zero normal flux on every wall means the mass of this zero-integral
    # mode must be conserved exactly.  Sampled mass drift measured < 1e-8.
    mass_T = _grid_mass(model.mesh_var)
    assert abs(mass_T) < 1e-6, f"zero-flux mode gained/lost mass: {mass_T:.3e}"


# ==============================================================================
# 3. Annulus constant uniform state, natural rims
# ==============================================================================


def test_annulus_constant_state_natural_boundaries() -> None:
    """Uniform state preserved with unconstrained rims on the annulus mesh.

    Same fixed-point argument as the Cartesian constant-state test, but on
    the gmsh-generated unstructured annulus mesh.  No boundary condition is
    applied on either rim: any spurious rim flux would drain the interior.
    """
    mesh = uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5, cellSize=0.08, qdegree=2
    )
    model = _make_model(mesh, kappa=1.0)
    _set_initial_condition(model, 1.0)

    model.run_simulation(duration=0.02, min_dt=0.01, max_dt=0.01)
    assert model.step == 2
    assert model.current_time == pytest.approx(0.02, abs=1e-14)

    _, values = _evaluate(model)
    drift = float(np.max(np.abs(values - 1.0)))
    # Measured drift ~3e-10 with tightened SNES tolerances.
    assert drift < 1e-8, f"uniform state drifted by {drift:.3e}"


# ==============================================================================
# 4. Annulus radial Neumann-compatible transient eigenmode, natural rims
# ==============================================================================


def test_annulus_radial_neumann_eigenmode() -> None:
    """Radial Laplacian eigenmode with zero normal gradient on both rims.

    Exact solution with kappa = 1 and no boundary conditions:

        C(r, t) = u(r)/max|u| * exp(-k^2 t),
        u(r) = Y_1(k r_i) J_0(k r) - J_1(k r_i) Y_0(k r),

    where k = 6.3931567616... is the first root of
    J_1(k r_i) Y_1(k r_o) - J_1(k r_o) Y_1(k r_i) = 0, which enforces
    u'(r_i) = u'(r_o) = 0 -- exactly the natural (zero-flux) condition on
    the two circular rims.  Evolved to T = 0.03 with dt = 5e-4 (60
    backward-Euler steps) at cellSize 0.05: measured amplitude error 5.0e-3
    and rescaled shape error 1.7e-2.  Tolerances are >= 4x the measured
    values and still separate a correct solve from a frozen field (~0.7) or
    a 2x-wrong decay rate (~0.25).
    """
    kappa = 1.0
    r_i, r_o = 0.5, 1.0

    k = _annulus_neumann_wavenumber(r_i, r_o)

    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_i, cellSize=0.05, qdegree=2
    )
    model = _make_model(mesh, kappa=kappa)

    coords = model.mesh_var.coords
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    _set_initial_condition(model, _annulus_neumann_mode(r, k, r_i))

    T = 0.03
    model.run_simulation(duration=T, min_dt=0.0005, max_dt=0.0005)
    assert model.step == 60
    assert model.current_time == pytest.approx(T, abs=1e-10)

    coords_out, values = _evaluate(model)
    r_out = np.sqrt(coords_out[:, 0] ** 2 + coords_out[:, 1] ** 2)

    decay = np.exp(-(k**2) * kappa * model.current_time)
    mode = _annulus_neumann_mode(r_out, k, r_i)
    exact = mode * decay

    amp_err = float(np.max(np.abs(values - exact)))
    assert amp_err < 0.025, f"Neumann-mode amplitude error {amp_err:.3e} >= 0.025"

    # The mode must actually have decayed to e^{-k^2 T} ~ 0.29.
    assert 0.1 < float(np.max(np.abs(values))) < 0.6

    shape_err = float(np.max(np.abs(values / decay - mode)))
    assert shape_err < 0.07, f"Neumann-mode shape error {shape_err:.3e} >= 0.07"


# ==============================================================================
# 5. Multicomponent cross-diffusion eigenmode (Mg / Fe / Ca closure)
# ==============================================================================


def test_multicomponent_cross_diffusion_eigenmode() -> None:
    """Exact coupled decay of a 2x2 cross-diffusion eigenmode with closure.

    ``MulticomponentDiffusionModel`` with components ["Mg", "Fe", "Ca"]
    (Ca = 1 - Mg - Fe implicit) and constant 2x2 reduced diffusion matrix

        D = [[1.0, 0.3],          D_MgMg = 1.0, D_MgFe = 0.3,
             [0.2, 0.5]]          D_FeMg = 0.2, D_FeFe = 0.5

    (nonsymmetric, positive definite: eigenvalues 1.1 and 0.4 with real
    eigenvectors).  The dominant eigenpair is lambda_1 = 1.1,
    v_1 = (3, 1)/sqrt(10).

    With no boundary conditions (natural = zero normal flux everywhere) and
    initial data

        x(r, 0) = x0 + a v_1 cos(pi x) cos(pi y),  x0 = (0.5, 0.25), a = 0.15,

    the exact solution is

        Mg(r, t) = 0.5  + a v_1[0] f(r) exp(-2 pi^2 lambda_1 t)
        Fe(r, t) = 0.25 + a v_1[1] f(r) exp(-2 pi^2 lambda_1 t)
        Ca(r, t) = 1 - Mg - Fe,   f = cos(pi x) cos(pi y)

    because the constant part has zero gradient (preserved exactly by the FE
    operator) and the eigenmode part decays at the coupled rate
    2 pi^2 lambda_1.  The cosine mode has zero normal derivative on all four
    walls, so every component flux J_i = (D v_1)_i grad(f) . n vanishes on
    the boundary and each component's mass is conserved.

    Measured at T = 0.05, dt = 1e-3 (50 backward-Euler steps), Q2,
    elementRes (12, 12): max errors 4.5e-4 (Mg), 2.0e-4 (Fe), 6.5e-4 (Ca);
    closure identity ~1e-16; per-component mass drift <= 4e-9.  Tolerances
    are >= 4x measured yet still fail qualitatively wrong physics: dropping
    cross-diffusion (Mg decaying at D_11 = 1.0) gives ~5e-3 error,
    transposing D moves the Mg:Fe mode ratio from 3:1 to 2:1 (~7e-3 error on
    Fe), and a stale closure gives ~0.13.
    """
    component_names = ["Mg", "Fe", "Ca"]  # Ca is the implicit closure
    d_values = {"D_MgMg": 1.0, "D_MgFe": 0.3, "D_FeMg": 0.2, "D_FeFe": 0.5}
    d_numeric = np.array([[1.0, 0.3], [0.2, 0.5]])

    # Dominant (fastest-decaying) eigenpair of the constant matrix; real by
    # construction (D is positive definite).
    eigvals, eigvecs = np.linalg.eig(d_numeric)
    order = np.argsort(eigvals)[::-1]
    lambda_1 = float(np.real(eigvals[order[0]]))
    v1 = np.real(eigvecs[:, order[0]])
    v1 = v1 / np.linalg.norm(v1)
    if v1[0] < 0:
        v1 = -v1
    assert lambda_1 > 0 and np.all(np.isreal(eigvals))

    a_amplitude = 0.15
    x0 = np.array([0.5, 0.25])  # constant offset (Mg, Fe)

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(12, 12), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    d_matrix = sp.Matrix(
        [
            [sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")],
            [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")],
        ]
    )
    model = MulticomponentDiffusionModel(
        component_names, mesh, diffusion_matrix=d_matrix, degree=2, order=1
    )
    model.set_diffusion_values(d_values)
    _tighten_solver(model)

    coords = model.mesh_vars[0].coords
    mode = np.cos(PI * coords[:, 0]) * np.cos(PI * coords[:, 1])
    mg0 = x0[0] + a_amplitude * v1[0] * mode
    fe0 = x0[1] + a_amplitude * v1[1] * mode
    with mesh.access(*model.mesh_vars):
        model.mesh_vars[0].data[:, 0] = mg0
        model.mesh_vars[1].data[:, 0] = fe0
        model.mesh_vars[2].data[:, 0] = 1.0 - mg0 - fe0

    # Initial masses (zero-flux => these must not change; the cosine mode
    # integrates to zero, so each mass equals the constant offset volume).
    mass_0 = [_grid_mass(var) for var in model.mesh_vars]

    T = 0.05
    model.run_simulation(duration=T, min_dt=0.001, max_dt=0.001)
    assert model.step == 50
    assert model.current_time == pytest.approx(T, abs=1e-12)

    decay = np.exp(-2.0 * PI**2 * lambda_1 * model.current_time)
    coords_out = model.mesh_vars[0].coords
    mode_out = np.cos(PI * coords_out[:, 0]) * np.cos(PI * coords_out[:, 1])
    mg_exact = x0[0] + a_amplitude * v1[0] * mode_out * decay
    fe_exact = x0[1] + a_amplitude * v1[1] * mode_out * decay
    ca_exact = 1.0 - mg_exact - fe_exact

    with mesh.access(*model.mesh_vars):
        mg = model.mesh_vars[0].data[:, 0].copy()
        fe = model.mesh_vars[1].data[:, 0].copy()
        ca = model.mesh_vars[2].data[:, 0].copy()

    err_mg = float(np.max(np.abs(mg - mg_exact)))
    err_fe = float(np.max(np.abs(fe - fe_exact)))
    err_ca = float(np.max(np.abs(ca - ca_exact)))
    assert err_mg < 2e-3, f"Mg exact coupled-decay error {err_mg:.3e} >= 2e-3"
    assert err_fe < 2e-3, f"Fe exact coupled-decay error {err_fe:.3e} >= 2e-3"
    assert err_ca < 3e-3, f"Ca exact closure-decay error {err_ca:.3e} >= 3e-3"

    # Closure identity Ca = 1 - Mg - Fe must hold at the nodes (measured
    # ~1e-16).
    closure_residual = float(np.max(np.abs(mg + fe + ca - 1.0)))
    assert closure_residual < 1e-12, (
        f"closure residual {closure_residual:.3e} >= 1e-12"
    )

    # Zero normal flux on every wall conserves each component's mass exactly.
    # Measured drift <= 4e-9; 1e-6 leaves two-plus orders of margin while
    # catching any real boundary leakage (a wrongly imposed boundary value
    # would move O(0.1) of mass).
    mass_T = [_grid_mass(var) for var in model.mesh_vars]
    for name, m0, m1 in zip(component_names, mass_0, mass_T):
        drift = abs(m1 - m0)
        assert drift < 1e-6, f"{name} mass drift {drift:.3e} >= 1e-6"
        assert abs(m1 - m0) == pytest.approx(0.0, abs=1e-6)


# ==============================================================================
# 6. Annulus multicomponent cross-diffusion eigenmode (Mg / Fe / Ca closure)
# ==============================================================================


def test_annulus_multicomponent_cross_diffusion_eigenmode() -> None:
    """Exact coupled decay of a 2x2 cross-diffusion Bessel mode on an annulus.

    ``MulticomponentDiffusionModel`` with components ["Mg", "Fe", "Ca"]
    (Ca = 1 - Mg - Fe implicit) and constant 2x2 reduced diffusion matrix

        D = [[1.0, 0.3],          D_MgMg = 1.0, D_MgFe = 0.3,
             [0.2, 0.5]]          D_FeMg = 0.2, D_FeFe = 0.5

    on ``uw.meshing.Annulus`` (r_i = 0.5, r_o = 1.0) with natural
    unconstrained boundaries on both rims.

    The dominant eigenpair of D is lambda_1 = 1.1, v_1 = (3, 1)/sqrt(10).
    Using the radial Bessel eigenfunction u(r) with u'(r_i) = u'(r_o) = 0
    at wavenumber k = 6.3931567616..., the initial data

        x(r, 0) = x0 + a v_1 u(r),  x0 = (0.5, 0.25), a = 0.15,

    decays at the coupled rate k^2 lambda_1:

        Mg(r, t) = 0.5  + a v_1[0] u(r) exp(-k^2 lambda_1 t)
        Fe(r, t) = 0.25 + a v_1[1] u(r) exp(-k^2 lambda_1 t)
        Ca(r, t) = 1 - Mg - Fe.

    Because u'(r_i) = u'(r_o) = 0, every component flux J_i . n vanishes
    on both circular rims, conserving total mass and preserving the closure.
    """
    component_names = ["Mg", "Fe", "Ca"]  # Ca is the implicit closure
    d_values = {"D_MgMg": 1.0, "D_MgFe": 0.3, "D_FeMg": 0.2, "D_FeFe": 0.5}
    d_numeric = np.array([[1.0, 0.3], [0.2, 0.5]])

    eigvals, eigvecs = np.linalg.eig(d_numeric)
    order = np.argsort(eigvals)[::-1]
    lambda_1 = float(np.real(eigvals[order[0]]))
    v1 = np.real(eigvecs[:, order[0]])
    v1 = v1 / np.linalg.norm(v1)
    if v1[0] < 0:
        v1 = -v1
    assert lambda_1 > 0 and np.all(np.isreal(eigvals))

    a_amplitude = 0.15
    x0 = np.array([0.5, 0.25])

    r_i, r_o = 0.5, 1.0
    k = _annulus_neumann_wavenumber(r_i, r_o)

    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_i, cellSize=0.06, qdegree=2
    )
    d_matrix = sp.Matrix(
        [
            [sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")],
            [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")],
        ]
    )
    model = MulticomponentDiffusionModel(
        component_names, mesh, diffusion_matrix=d_matrix, degree=2, order=1
    )
    model.set_diffusion_values(d_values)
    _tighten_solver(model)

    coords = model.mesh_vars[0].coords
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    mode = _annulus_neumann_mode(r, k, r_i)

    mg0 = x0[0] + a_amplitude * v1[0] * mode
    fe0 = x0[1] + a_amplitude * v1[1] * mode
    with mesh.access(*model.mesh_vars):
        model.mesh_vars[0].data[:, 0] = mg0
        model.mesh_vars[1].data[:, 0] = fe0
        model.mesh_vars[2].data[:, 0] = 1.0 - mg0 - fe0

    # Initial masses on the annulus via FE quadrature
    mass_0 = [
        float(uw.maths.Integral(mesh, fn=var.sym[0]).evaluate())
        for var in model.mesh_vars
    ]

    T = 0.02
    dt = 0.001
    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)
    assert model.step == 20
    assert model.current_time == pytest.approx(T, abs=1e-12)

    decay = np.exp(-(k**2) * lambda_1 * model.current_time)
    coords_out = model.mesh_vars[0].coords
    r_out = np.sqrt(coords_out[:, 0] ** 2 + coords_out[:, 1] ** 2)
    mode_out = _annulus_neumann_mode(r_out, k, r_i)
    mg_exact = x0[0] + a_amplitude * v1[0] * mode_out * decay
    fe_exact = x0[1] + a_amplitude * v1[1] * mode_out * decay
    ca_exact = 1.0 - mg_exact - fe_exact

    with mesh.access(*model.mesh_vars):
        mg = model.mesh_vars[0].data[:, 0].copy()
        fe = model.mesh_vars[1].data[:, 0].copy()
        ca = model.mesh_vars[2].data[:, 0].copy()

    err_mg = float(np.max(np.abs(mg - mg_exact)))
    err_fe = float(np.max(np.abs(fe - fe_exact)))
    err_ca = float(np.max(np.abs(ca - ca_exact)))
    assert err_mg < 3e-3, f"Mg exact coupled-decay error {err_mg:.3e} >= 3e-3"
    assert err_fe < 2e-3, f"Fe exact coupled-decay error {err_fe:.3e} >= 2e-3"
    assert err_ca < 4e-3, f"Ca exact closure-decay error {err_ca:.3e} >= 4e-3"

    # Verify that the mode has actually decayed (decay ~ 0.41)
    assert 0.3 < decay < 0.5

    # Shape check: rescaled by coupled decay, perturbations match eigenmode
    shape_err_mg = float(
        np.max(np.abs((mg - x0[0]) / (a_amplitude * v1[0] * decay) - mode_out))
    )
    shape_err_fe = float(
        np.max(np.abs((fe - x0[1]) / (a_amplitude * v1[1] * decay) - mode_out))
    )
    assert shape_err_mg < 0.05, f"Mg shape error {shape_err_mg:.3e} >= 0.05"
    assert shape_err_fe < 0.05, f"Fe shape error {shape_err_fe:.3e} >= 0.05"

    # Closure identity Ca = 1 - Mg - Fe must hold at the nodes
    closure_residual = float(np.max(np.abs(mg + fe + ca - 1.0)))
    assert closure_residual < 1e-12, (
        f"closure residual {closure_residual:.3e} >= 1e-12"
    )

    # Zero normal flux on both circular rims conserves each component's mass
    mass_T = [
        float(uw.maths.Integral(mesh, fn=var.sym[0]).evaluate())
        for var in model.mesh_vars
    ]
    for name, m0, m1 in zip(component_names, mass_0, mass_T):
        drift = abs(m1 - m0)
        assert drift < 1e-6, f"{name} mass drift {drift:.3e} >= 1e-6"
        assert abs(m1 - m0) == pytest.approx(0.0, abs=1e-6)

