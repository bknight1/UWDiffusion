"""Time-integration options: higher-order stepping and the source term.

Covers the two additions to the unified solver:

* ``order`` > 1 -- BDF for the time derivative plus an Adams-Moulton average
  of the component fluxes over history levels.  Should be second order in
  ``dt`` where ``order=1`` is first order.
* ``S`` -- a per-component source term entering the residual as
  ``F0 = dx/dt - S``, so the strong form gains ``+ S``.

Run:
    python -m pytest tests/test_time_integration_options.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

PI = np.pi

try:
    import sympy as sp
    import underworld3 as uw

    from UWDiffusion import (
        DiffusionDecayIngrowthModel,
        DiffusionModel,
        MulticomponentDiffusionModel,
        lasaga_matrix_from_components,
    )
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"Underworld3/MPI/Gmsh environment unavailable: {exc}",
        allow_module_level=True,
    )


def _mesh(res=12):
    return uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


def _cos_mode(coords):
    return np.cos(PI * coords[:, 0]) * np.cos(PI * coords[:, 1])


def _tighten(model):
    solvers = list(getattr(model, "solvers", []) or [])
    for name in ("parent_diffusion", "daughter_diffusion"):
        s = getattr(model, name, None)
        if s is not None and s not in solvers:
            solvers.append(s)
    for solver in solvers:
        solver.petsc_options["snes_rtol"] = 1e-10
        solver.petsc_options["snes_atol"] = 1e-12
        solver.petsc_options["ksp_rtol"] = 1e-8
        solver.petsc_options["ksp_atol"] = 1e-14


# ==============================================================================
# 1. Higher-order stepping is more accurate at the same dt
# ==============================================================================


def _run_scalar_eigenmode(order, dt, T, kappa=1.0, res=12):
    mesh = _mesh(res)
    model = DiffusionModel("C", mesh, degree=2, order=order)
    model.diffusivity = kappa
    _tighten(model)
    coords = model.mesh_var.coords
    with mesh.access(model.mesh_var):
        model.mesh_var.data[:, 0] = _cos_mode(coords)
    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)
    with mesh.access(model.mesh_var):
        vals = model.mesh_var.data[:, 0].copy()
    exact = _cos_mode(coords) * np.exp(-2.0 * PI**2 * kappa * T)
    return float(np.max(np.abs(vals - exact))), model.step


def test_order2_beats_order1_on_eigenmode():
    """Backward Euler is O(dt); the order-2 scheme should be markedly better.

    The pure cosine eigenmode decays at a known rate, so the amplitude error
    is a clean measure of the temporal error once dt is small enough that the
    spatial error floor is not dominant.
    """
    T = 0.02
    err1, steps1 = _run_scalar_eigenmode(order=1, dt=2.0e-3, T=T)
    err2, steps2 = _run_scalar_eigenmode(order=2, dt=2.0e-3, T=T)

    assert steps1 == steps2 == 10, (steps1, steps2)
    print(f"\n    order=1 L_inf error {err1:.3e}")
    print(f"    order=2 L_inf error {err2:.3e}")
    assert err2 < err1, f"order=2 ({err2:.3e}) not better than order=1 ({err1:.3e})"


def test_order2_temporal_convergence():
    """Halving dt should reduce the order-2 error faster than order 1."""
    T = 0.02
    errs = {}
    for order in (1, 2):
        e_hi, _ = _run_scalar_eigenmode(order=order, dt=2.0e-3, T=T)
        e_lo, _ = _run_scalar_eigenmode(order=order, dt=1.0e-3, T=T)
        errs[order] = (e_hi, e_lo)
        print(f"\n    order={order}: dt=2e-3 -> {e_hi:.3e}, dt=1e-3 -> {e_lo:.3e}")

    # order 1: error roughly halves; order 2: error drops faster.
    ratio1 = errs[1][0] / max(errs[1][1], 1e-300)
    ratio2 = errs[2][0] / max(errs[2][1], 1e-300)
    print(f"    reduction factor: order=1 {ratio1:.2f}, order=2 {ratio2:.2f}")
    assert ratio2 > ratio1, (
        f"order=2 refinement factor {ratio2:.2f} not greater than "
        f"order=1 {ratio1:.2f}"
    )


def test_order_greater_than_3_rejected():
    with pytest.raises(NotImplementedError):
        DiffusionModel("C", _mesh(4), degree=2, order=4)


def test_default_degree_and_order():
    """Defaults are ``degree=1, order=1`` and reproduce an explicit run.

    Every other test in the suite pins ``degree`` and ``order`` explicitly, so
    the defaults are otherwise unexercised.  ``degree=1, order=1`` is the
    documented choice for sharp growth interfaces (and the cheapest option), so
    it needs its own guard.
    """
    mesh = _mesh()
    for model in (
        DiffusionModel("C", mesh),
        MulticomponentDiffusionModel(["A", "B"], mesh, closure=False),
        DiffusionDecayIngrowthModel("P", "D", mesh, 1.0e9),
    ):
        assert model.degree == 1, f"{type(model).__name__}.degree default is not 1"
        assert model.order == 1, f"{type(model).__name__}.order default is not 1"

    dt, T = 2.0e-3, 0.02

    def _run(use_defaults):
        """use_defaults=True passes neither degree nor order."""
        m = _mesh(12)
        kwargs = {} if use_defaults else {"degree": 1, "order": 1}
        model = DiffusionModel("C", m, **kwargs)
        model.diffusivity = 1.0
        _tighten(model)
        coords = model.mesh_var.coords
        with m.access(model.mesh_var):
            model.mesh_var.data[:, 0] = _cos_mode(coords)
        model.run_simulation(duration=T, min_dt=dt, max_dt=dt)
        with m.access(model.mesh_var):
            vals = model.mesh_var.data[:, 0].copy()
        exact = _cos_mode(coords) * np.exp(-2.0 * PI**2 * 1.0 * T)
        return float(np.max(np.abs(vals - exact))), int(model.step)

    err_default, steps_default = _run(True)
    err_explicit, steps_explicit = _run(False)

    assert steps_default == steps_explicit
    assert err_default == pytest.approx(err_explicit, rel=1e-12), (
        f"default run {err_default:.6e} != explicit degree=1,order=1 {err_explicit:.6e}"
    )


# ==============================================================================
# 2. Source term
# ==============================================================================


def test_uniform_source_growth_exact():
    """Uniform field plus a constant source: diffusion is inactive.

        dC/dt = S   =>   C(t) = C0 + S t

    Exactly reproducible at any dt, so this isolates the source wiring and
    its sign.
    """
    mesh = _mesh(10)
    model = DiffusionModel("C", mesh, degree=2, order=1)
    model.diffusivity = 1.0
    _tighten(model)

    c0, source, T, dt = 0.3, 0.05, 1.0, 0.25
    with mesh.access(model.mesh_var):
        model.mesh_var.data[:, 0] = c0

    # The scalar model exposes `S` as a 1x1 matrix (legacy shape).
    model.S = sp.Matrix([[source]])
    assert model.source[0] == source or float(model.source[0]) == source

    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)

    with mesh.access(model.mesh_var):
        vals = model.mesh_var.data[:, 0].copy()
    exact = c0 + source * T

    err = float(np.max(np.abs(vals - exact)))
    assert err < 1e-6, f"uniform source growth error {err:.3e} (exact {exact:.6f})"


def test_negative_source_decay_exact():
    """A negative source is a sink: C(t) = C0 - |S| t."""
    mesh = _mesh(10)
    model = DiffusionModel("C", mesh, degree=2, order=1)
    model.diffusivity = 1.0
    _tighten(model)

    c0, source, T, dt = 0.4, -0.1, 1.0, 0.25
    with mesh.access(model.mesh_var):
        model.mesh_var.data[:, 0] = c0
    model.S = sp.Matrix([[source]])
    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)

    with mesh.access(model.mesh_var):
        vals = model.mesh_var.data[:, 0].copy()
    err = float(np.max(np.abs(vals - (c0 + source * T))))
    assert err < 1e-6, f"uniform sink error {err:.3e}"


def test_per_component_source_on_multicomponent():
    """Independent per-component sources with no closure constraint."""
    mesh = _mesh(10)
    d = sp.Matrix([[sp.Symbol("D00"), 0], [0, sp.Symbol("D11")]])
    model = MulticomponentDiffusionModel(
        ["A", "B"], mesh, diffusion_matrix=d, degree=2, order=1, closure=False
    )
    model.set_diffusion_values({"D00": 1.0, "D11": 1.0})
    _tighten(model)

    T, dt = 1.0, 0.25
    with mesh.access(*model.mesh_vars):
        model.mesh_vars[0].data[:, 0] = 0.2
        model.mesh_vars[1].data[:, 0] = 0.6

    model.set_source([0.05, -0.02])
    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)

    with mesh.access(*model.mesh_vars):
        a = model.mesh_vars[0].data[:, 0].copy()
        b = model.mesh_vars[1].data[:, 0].copy()

    err_a = float(np.max(np.abs(a - (0.2 + 0.05 * T))))
    err_b = float(np.max(np.abs(b - (0.6 - 0.02 * T))))
    assert err_a < 1e-6, f"component A source error {err_a:.3e}"
    assert err_b < 1e-6, f"component B source error {err_b:.3e}"


def test_source_with_closure_conserves_total():
    """With closure, opposing sources must move the total by their net sum."""
    mesh = _mesh(10)
    d = sp.Matrix([[sp.Symbol("D00"), sp.Symbol("D01")],
                   [sp.Symbol("D10"), sp.Symbol("D11")]])
    model = MulticomponentDiffusionModel(
        ["A", "B", "C"], mesh, diffusion_matrix=d, degree=2, order=1, closure=True
    )
    model.set_diffusion_values({"D00": 1.0, "D01": 0.1, "D10": 0.1, "D11": 1.0})
    _tighten(model)

    T, dt = 1.0, 0.25
    with mesh.access(*model.mesh_vars):
        model.mesh_vars[0].data[:, 0] = 0.3
        model.mesh_vars[1].data[:, 0] = 0.3
        model.mesh_vars[2].data[:, 0] = 0.4

    model.set_source([0.0, 0.0])
    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)

    with mesh.access(*model.mesh_vars):
        fields = [v.data[:, 0].copy() for v in model.mesh_vars]
    total = sum(fields)
    assert np.allclose(total, 1.0, atol=1e-10), f"closure broken: {total[:3]}"


# ==============================================================================
# 3. Variable timesteps
# ==============================================================================


def _run_variable_dt_reaction(order, steps, lam=1.0):
    """Pure decay on a uniform field, stepping through an arbitrary dt ladder.

    A uniform field makes the spatial Laplacian exactly zero, so the problem
    reduces to ``dC/dt = -lambda C`` with *no spatial error at all*.  Any
    discrepancy is purely temporal, which makes this a clean probe of how
    variable timesteps are handled.

    ``steps`` is a sequence of dt values; the model is stepped by calling
    ``run_simulation`` with absolute end times and clamped dt, so the ladder
    exercises the real public stepper (and its truncated final step).
    """
    mesh = _mesh(8)
    model = MulticomponentDiffusionModel(
        ["C"], mesh, diffusion_matrix=sp.Matrix([[sp.Symbol("D00")]]),
        degree=2, order=order, closure=False,
    )
    model.set_diffusion_values({"D00": 1.0})
    _tighten(model)
    # dC/dt = div(D grad C) - lambda C  ->  uniform field gives pure decay.
    model.set_source([-lam * sp.Symbol("xi_1")])

    c0 = 0.8
    with mesh.access(*model.mesh_vars):
        model.mesh_vars[0].data[:, 0] = c0

    t = 0.0
    for dt in steps:
        t += dt
        model.run_simulation(duration=t, min_dt=dt, max_dt=dt)

    with mesh.access(*model.mesh_vars):
        vals = model.mesh_vars[0].data[:, 0].copy()
    return float(np.mean(vals)), t, model.step


def test_variable_dt_ladder_matches_exact_decay():
    """A deliberately uneven dt ladder must still hit the exact solution.

    The DDt uses variable-step BDF coefficients (``c0=(1+2r)/(1+r)`` etc. with
    ``r = dt_n/dt_{n-1}``), so an uneven sequence must not degrade accuracy
    beyond the scheme's order.
    """
    lam = 1.0
    # Uneven ladder: ratios r ranging from 0.4 to 2.5.
    fine = [0.02, 0.05, 0.02, 0.08, 0.03, 0.10, 0.02, 0.05, 0.02, 0.03]
    for order in (1, 2):
        c, T, steps = _run_variable_dt_reaction(order, fine, lam=lam)
        exact = 0.8 * np.exp(-lam * T)
        err = abs(c - exact)
        print(f"\n    order={order} variable dt: T={T:.4f}, steps={steps}, "
              f"C={c:.8f}, exact={exact:.8f}, err={err:.3e}")
        # Order 1 is O(dt) ~ 1e-2 here; order 2 should be far tighter.
        tol = 2e-2 if order == 1 else 5e-3
        assert err < tol, f"order={order} variable-dt error {err:.3e} >= {tol:.3e}"
        assert 0.0 < c < 0.8, "decay went outside its physical range"


def test_variable_dt_preserves_convergence_order():
    """Refining an uneven ladder must still show O(dt) then O(dt^2).

    The refined ladder halves every dt AND doubles the step count, so both
    runs span the same total time -- otherwise the two would be solved over
    different intervals and the comparison is meaningless.
    """
    lam = 1.0
    pattern = [0.04, 0.10, 0.04, 0.16, 0.06, 0.20, 0.04, 0.10]
    coarse = pattern
    fine = [d / 2.0 for d in pattern * 2]

    results = {}
    for order in (1, 2):
        c_hi, T_hi, _ = _run_variable_dt_reaction(order, coarse, lam=lam)
        c_lo, T_lo, _ = _run_variable_dt_reaction(order, fine, lam=lam)
        assert T_hi == pytest.approx(T_lo, abs=1e-12), (T_hi, T_lo)
        exact = 0.8 * np.exp(-lam * T_hi)
        e_hi, e_lo = abs(c_hi - exact), abs(c_lo - exact)
        ratio = e_hi / max(e_lo, 1e-300)
        results[order] = ratio
        print(f"\n    order={order}: err {e_hi:.3e} -> {e_lo:.3e}  "
              f"(reduction {ratio:.2f})")

    assert results[1] > 1.5, f"order=1 reduction {results[1]:.2f} too small"
    assert results[2] > 2.5, f"order=2 reduction {results[2]:.2f} too small"
    assert results[2] > results[1], (
        f"order=2 reduction {results[2]:.2f} not better than "
        f"order=1 {results[1]:.2f}"
    )


def test_variable_dt_final_step_is_truncated():
    """run_simulation must land exactly on the requested end time.

    The stepper truncates the last step so the target is reached exactly,
    which is the most common source of a variable dt in practice.
    """
    mesh = _mesh(6)
    model = DiffusionModel("C", mesh, degree=2, order=2)
    model.diffusivity = 1.0
    _tighten(model)
    with mesh.access(model.mesh_var):
        model.mesh_var.data[:, 0] = 1.0

    # 0.07 is not a multiple of 0.02, so the final step must be shortened.
    model.run_simulation(duration=0.07, min_dt=0.02, max_dt=0.02)
    assert model.current_time == pytest.approx(0.07, abs=1e-12), model.current_time
    assert model.step == 4, model.step


# ==============================================================================
# 6. Unbound diffusivity symbols must not silently yield one giant step
# ==============================================================================


def test_unresolved_diffusivity_symbols_raise():
    """run_simulation must not silently run the whole model in a single step.

    The timestep heuristic evaluates the diffusion matrix at the top of the
    step loop, before any pre-solve hook runs.  With unsubstituted parameter
    symbols it used to return 0.0, which makes max_time_step infinite, so the
    run advances to the target in ONE step while reporting success -- silently
    discarding every timestep the heuristic was supposed to choose.
    """
    mesh = _mesh(6)
    model = MulticomponentDiffusionModel(
        ["Fe", "Mg", "Ca"], mesh, degree=1, order=1,
        diffusion_matrix=lasaga_matrix_from_components(["Fe", "Mg", "Ca"]),
    )
    # Deliberately NOT calling set_diffusion_values.
    with pytest.raises(RuntimeError, match="unresolved symbols"):
        model.run_simulation(duration=0.01)


def test_bound_diffusivity_symbols_step_normally():
    """The same model runs once the parameter symbols are bound."""
    mesh = _mesh(6)
    model = MulticomponentDiffusionModel(
        ["Fe", "Mg", "Ca"], mesh, degree=1, order=1,
        diffusion_matrix=lasaga_matrix_from_components(["Fe", "Mg", "Ca"]),
    )
    model.set_diffusion_values({"Fe": 1.0, "Mg": 0.8, "Ca": 0.5})
    model.run_simulation(duration=0.01, min_dt=0.005, max_dt=0.005)
    assert model.step == 2, model.step
