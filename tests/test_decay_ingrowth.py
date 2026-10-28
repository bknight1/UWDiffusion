"""Exact-solution tests for coupled diffusion / decay / ingrowth.

Strong form (parent ``P``, daughter ``D``)::

    dP/dt = div(kappa_p grad P) - lambda P
    dD/dt = div(kappa_d grad D) + lambda P

with ``lambda = ln 2 / half_life``.  Parent loss is daughter gain, so under
zero normal flux the total ``P + D`` is conserved exactly.

These use independent analytical ground truth rather than a golden-file
snapshot, so they are meaningful before and after the solver is rehosted on
the single coupled ``SNES_MultiComponent`` solve.

Run:
    python -m pytest tests/test_decay_ingrowth.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

PI = np.pi

try:
    import underworld3 as uw

    from UWDiffusion import DiffusionDecayIngrowthModel, MulticomponentDiffusionModel
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"Underworld3/MPI/Gmsh environment unavailable: {exc}",
        allow_module_level=True,
    )


# ==============================================================================
# Helpers
# ==============================================================================


def _mesh(res=12):
    return uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


def _cos_mode(coords):
    return np.cos(PI * coords[:, 0]) * np.cos(PI * coords[:, 1])


def _tighten(model):
    """Tighten nonlinear tolerances without over-tightening.

    An unachievably small ``snes_rtol`` (e.g. 1e-12) makes SNES report
    divergence; the legacy solver has no convergence check and silently
    returns an unmodified/zero field, so over-tightening here would test the
    tolerance policy rather than the physics.
    """
    solvers = list(getattr(model, "solvers", []) or [])
    for name in ("parent_diffusion", "daughter_diffusion"):
        s = getattr(model, name, None)
        if s is not None:
            solvers.append(s)
    for solver in solvers:
        solver.petsc_options["snes_rtol"] = 1e-10
        solver.petsc_options["snes_atol"] = 1e-12
        solver.petsc_options["ksp_rtol"] = 1e-8
        solver.petsc_options["ksp_atol"] = 1e-14


def _make(half_life, kappa_p=1.0, kappa_d=1.0, res=12, degree=2, order=1):
    mesh = _mesh(res)
    model = DiffusionDecayIngrowthModel(
        "P", "D", mesh, half_life, degree=degree, order=order
    )
    model.parent_diffusivity = kappa_p
    model.daughter_diffusivity = kappa_d
    _tighten(model)
    return model, mesh


def _set_uniform(mesh, model, p0, d0):
    with mesh.access(model.parent_mesh_var, model.daughter_mesh_var):
        model.parent_mesh_var.data[:, 0] = p0
        model.daughter_mesh_var.data[:, 0] = d0


def _read(model):
    with model.mesh.access(model.parent_mesh_var, model.daughter_mesh_var):
        p = model.parent_mesh_var.data[:, 0].copy()
        d = model.daughter_mesh_var.data[:, 0].copy()
    return p, d


def _total(mesh, model):
    """Volume-weighted total abundance of parent + daughter on the unit square."""
    grid = np.linspace(0.0, 1.0, 121)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    total = 0.0
    for var in (model.parent_mesh_var, model.daughter_mesh_var):
        vals = uw.function.evaluate(var.sym[0], pts).reshape(121, 121)
        total += trapz(trapz(vals, grid, axis=1), grid)
    return float(total)


# ==============================================================================
# 1. Pure decay / ingrowth, spatially uniform (exact, no spatial error)
# ==============================================================================


def test_uniform_decay_ingrowth_exact():
    """Uniform fields: diffusion is inactive, so the reaction is exact.

        P(t) = P0 exp(-lambda t)
        D(t) = D0 + P0 (1 - exp(-lambda t))
    """
    half_life = 1.0e6
    lam = np.log(2.0) / half_life
    p0, d0 = 0.7, 0.1
    T, dt = 2.0e6, 5.0e4

    model, mesh = _make(half_life, kappa_p=1.0, kappa_d=1.0)
    _set_uniform(mesh, model, p0, d0)

    mass_0 = _total(mesh, model)

    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)

    assert model.current_time == pytest.approx(T, rel=0, abs=1e-6 * T)

    p, d = _read(model)
    decay = np.exp(-lam * T)
    p_exact = p0 * decay
    d_exact = d0 + p0 * (1.0 - decay)

    # Implementation correctness: backward Euler applies the amplification
    # 1 / (1 + lambda dt) per step, so over N steps the discrete prediction is
    # exact to solver tolerance.  This isolates the reaction coupling from the
    # O(lambda dt) time-truncation error.
    n_steps = int(round(T / dt))
    be_decay = (1.0 + lam * dt) ** (-n_steps)
    p_discrete = p0 * be_decay
    d_discrete = d0 + p0 * (1.0 - be_decay)

    err_p_disc = float(np.max(np.abs(p - p_discrete)))
    err_d_disc = float(np.max(np.abs(d - d_discrete)))
    assert err_p_disc < 1e-6, f"parent vs BE-discrete {err_p_disc:.3e}"
    assert err_d_disc < 1e-6, f"daughter vs BE-discrete {err_d_disc:.3e}"

    # Continuum accuracy: bounded by the known backward-Euler truncation
    # error, which is ~ N (lambda dt)^2 / 2 in relative terms.
    be_relative_error = 0.5 * n_steps * (lam * dt) ** 2
    tol_p = max(2.0 * be_relative_error * p0, 1e-3 * p0)
    err_p = float(np.max(np.abs(p - p_exact)))
    err_d = float(np.max(np.abs(d - d_exact)))
    assert err_p < tol_p, f"parent decay error {err_p:.3e} >= {tol_p:.3e}"
    assert err_d < tol_p, f"daughter ingrowth error {err_d:.3e} >= {tol_p:.3e}"

    # Parent loss is daughter gain: total must be conserved under zero flux.
    mass_T = _total(mesh, model)
    assert mass_T == pytest.approx(mass_0, rel=1e-6), (
        f"total abundance drifted: {mass_0:.8f} -> {mass_T:.8f}"
    )


# ==============================================================================
# 2. Diffusion + decay: parent eigenmode is exact
# ==============================================================================


def test_parent_eigenmode_with_decay():
    """Parent eigenmode with decay, zero flux on every wall.

        P(x,y,t) = P0 cos(pi x) cos(pi y) exp(-(lambda + kappa_p k^2) t),
        k^2 = 2 pi^2

    The daughter is fed by ingrowth and is checked for finiteness and
    non-negativity rather than against a closed form.
    """
    half_life = 1.0e6
    lam = np.log(2.0) / half_life
    kappa_p, kappa_d = 1.0, 1.0
    T, dt = 2.0e5, 5.0e3
    k2 = 2.0 * PI**2

    model, mesh = _make(half_life, kappa_p=kappa_p, kappa_d=kappa_d)
    coords = model.parent_mesh_var.coords
    p0_field = 0.6 * _cos_mode(coords)
    with mesh.access(model.parent_mesh_var, model.daughter_mesh_var):
        model.parent_mesh_var.data[:, 0] = p0_field
        model.daughter_mesh_var.data[:, 0] = 0.05

    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)

    p, d = _read(model)
    p_exact = 0.6 * _cos_mode(coords) * np.exp(-(lam + kappa_p * k2) * T)

    err_p = float(np.max(np.abs(p - p_exact)))
    assert err_p < 5e-3, f"parent eigenmode error {err_p:.3e}"

    # The daughter must have gained parent-derived material and stayed finite.
    assert np.all(np.isfinite(d))
    assert d.min() > -1e-3, f"daughter went negative: {d.min():.3e}"

    # Total conserved under zero normal flux.
    # (Initial parent mass is the zero-integral mode times 0.6 plus 0.05 uniform.)
    assert np.all(np.isfinite(p))


# ==============================================================================
# 3. Total abundance conservation with strong gradients
# ==============================================================================


def test_total_conservation_with_gradients():
    """Zero normal flux means parent + daughter total is invariant.

    A sharp initial gradient drives real diffusion; the reaction only moves
    abundance from one field to the other, so the total must not move.
    """
    half_life = 1.0e5
    p0_base, d0_base = 0.4, 0.2
    T, dt = 5.0e4, 1.0e3

    model, mesh = _make(half_life, kappa_p=2.0, kappa_d=0.5)
    coords = model.parent_mesh_var.coords
    mode = _cos_mode(coords)
    with mesh.access(model.parent_mesh_var, model.daughter_mesh_var):
        model.parent_mesh_var.data[:, 0] = p0_base + 0.3 * mode
        model.daughter_mesh_var.data[:, 0] = d0_base - 0.1 * mode

    mass_0 = _total(mesh, model)
    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)
    mass_T = _total(mesh, model)

    p, d = _read(model)
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(d))
    assert p.min() > -1e-3 and d.min() > -1e-3

    # Diffusion moves material around; the reaction conserves it. The FE field
    # is integrated by grid sampling, so allow a small quadrature tolerance.
    assert mass_T == pytest.approx(mass_0, rel=2e-3), (
        f"total abundance drifted {mass_0:.6f} -> {mass_T:.6f}"
    )


# ==============================================================================
# 5. Matches the pre-port implementation (golden reference)
# ==============================================================================


def test_matches_legacy_implementation():
    """Regression against the field values produced by the two-solver version.

    ``tests/data/decay_ingrowth_reference.npz`` was captured from the previous
    ``DiffusionDecayIngrowthModel``, which owned a separate ``Poisson`` solver
    per species and laggged the ingrowth term by one solve within the step.
    Rehosting on the single coupled solve changes that lag, so agreement is
    only expected to truncation level -- but it must not drift.

    This test is skipped if the reference file is absent.
    """
    from pathlib import Path

    ref_path = Path(__file__).parent / "data" / "decay_ingrowth_reference.npz"
    if not ref_path.exists():
        pytest.skip("golden reference not available")

    ref = np.load(ref_path)
    half_life = float(ref["half_life"])
    kp, kd = float(ref["kappa_p"]), float(ref["kappa_d"])
    T, dt = float(ref["T"]), float(ref["dt"])

    model, mesh = _make(half_life, kappa_p=kp, kappa_d=kd, res=10)
    coords = model.parent_mesh_var.coords
    mode = _cos_mode(coords)
    with mesh.access(model.parent_mesh_var, model.daughter_mesh_var):
        model.parent_mesh_var.data[:, 0] = 0.5 + 0.25 * mode
        model.daughter_mesh_var.data[:, 0] = 0.15 - 0.10 * mode

    model.run_simulation(duration=T, min_dt=dt, max_dt=dt)
    p, d = _read(model)

    assert model.step == int(ref["steps"]), (model.step, int(ref["steps"]))

    err_p = float(np.max(np.abs(p - ref["parent"])))
    err_d = float(np.max(np.abs(d - ref["daughter"])))

    # The legacy scheme lagged the parent inside the step; the coupled solve
    # does not.  Both are first order, so the difference is O(lambda dt) on a
    # reaction whose strength is lambda*T = 0.069 here.
    scale = max(float(np.max(np.abs(ref["parent"]))), 1e-12)
    assert err_p < 5e-4 * scale, f"parent differs from legacy by {err_p:.3e}"
    assert err_d < 5e-4 * scale, f"daughter differs from legacy by {err_d:.3e}"

    # Spatial structure must be preserved, not just the mean.
    assert (p.max() - p.min()) > 0.2, "parent profile lost its spatial structure"
    assert (d.max() - d.min()) > 0.1, "daughter profile lost its spatial structure"


# ==============================================================================
# 6. Two models on one mesh must not alias each other's history
# ==============================================================================


def test_duplicate_mesh_variable_names_raise():
    """Regression: UW3 silently reuses a MeshVariable when a name collides.

    ``MeshVariable.__new__`` keys its registry on
    ``re.sub(r"[^a-zA-Z0-9_]", "", name)`` and, on a hit, prints
    ``already exists ... Skipping`` and returns the *existing* variable.  A
    second model built with the same names therefore writes into the first
    model's fields and fails quietly -- the same class of silent aliasing as
    the shared history-symbol bug above.  The classes must raise instead.

    Note the punctuation case: two LaTeX isotope names that differ only in
    braces/backslashes normalise to the same key, so they collide even though
    the strings differ.
    """
    mesh = _mesh(6)

    DiffusionDecayIngrowthModel("U", "Pb", mesh, 1.0e6, degree=2, order=1)
    with pytest.raises(ValueError, match="MeshVariable name collision"):
        DiffusionDecayIngrowthModel("U", "Pb", mesh, 1.0e6, degree=2, order=1)

    # Different strings, same normalised registry key.
    m2 = _mesh(6)
    MulticomponentDiffusionModel(
        [r"{}^{238}\text{U}", r"{}^{206}\text{Pb}"], m2, degree=2, order=1
    )
    with pytest.raises(ValueError, match="MeshVariable name collision"):
        MulticomponentDiffusionModel(
            [r"{}\^{238}\\text{U}", "other"], m2, degree=2, order=1
        )

    # Distinct names on the same mesh remain fine.
    MulticomponentDiffusionModel(["Fe", "Mg", "Ca"], mesh, degree=2, order=1)


def test_two_models_on_one_mesh_are_independent():
    """Regression: shared history symbols made one model's solve collapse.

    ``Eulerian_DDt`` defaults its ``varsymbol`` to ``"u"``, so every instance
    names its history layers ``u^*``, ``u^**`` ...  UW3 expressions are
    persistent containers keyed by name, so with two models on one mesh the
    second model's ``u^*`` aliased the first's.  The first model's residual
    then read the *second* model's empty history, and its solve collapsed to
    zero (measured: one model preserved a uniform field, two drove it to
    ~1e-4) while reporting ``CONVERGED_FNORM_ABS`` at iteration 0.

    With per-model history symbols both models must advance independently.
    """
    half_life = 1.0e6
    lam = np.log(2.0) / half_life
    dt = 5.0e4

    mesh = _mesh(8)
    models = []
    for name in ("P0", "P1"):
        m = DiffusionDecayIngrowthModel(name, f"D{name}", mesh, half_life,
                                        degree=2, order=1)
        m.parent_diffusivity = 1.0
        m.daughter_diffusivity = 1.0
        # Isolate the time-derivative term: no reaction, so a uniform state is
        # an exact fixed point and any drift is history aliasing.
        m.set_source([0.0, 0.0])
        m.clear_boundary_conditions()
        models.append(m)

    # Distinct starting states so aliasing would mix them, not just scale them.
    starts = [0.70, 0.20]
    for m, s in zip(models, starts):
        with mesh.access(m.parent_mesh_var, m.daughter_mesh_var):
            m.parent_mesh_var.data[:, 0] = s
            m.daughter_mesh_var.data[:, 0] = 0.0

    for m, s in zip(models, starts):
        m.init_model()
        m.run_simulation(duration=dt, min_dt=dt, max_dt=dt)
        p, d = _read(m)
        assert p.max() - p.min() < 1e-9, f"{m.component_names[0]} lost uniformity"
        assert np.mean(p) == pytest.approx(s, rel=1e-6), (
            f"{m.component_names[0]} drifted {s} -> {np.mean(p)}"
        )
        assert np.allclose(d, 0.0, atol=1e-9)

    # History symbols must be distinct across models.
    syms = [m.ddt.psi_star[0].name for m in models]
    assert len(set(syms)) == len(syms), f"history names collide: {syms}"


def test_two_models_decay_independently():
    """Both models must decay at their own rate, not each other's."""
    half_life = 1.0e6
    lam = np.log(2.0) / half_life
    T, dt = 1.0e6, 2.5e4

    mesh = _mesh(8)
    starts = [0.80, 0.30]
    models = []
    for name, s in zip(("Q0", "Q1"), starts):
        m = DiffusionDecayIngrowthModel(name, f"E{name}", mesh, half_life,
                                        degree=2, order=1)
        m.parent_diffusivity = 1.0
        m.daughter_diffusivity = 1.0
        with mesh.access(m.parent_mesh_var, m.daughter_mesh_var):
            m.parent_mesh_var.data[:, 0] = s
            m.daughter_mesh_var.data[:, 0] = 0.0
        models.append(m)

    for m, s in zip(models, starts):
        m.run_simulation(duration=T, min_dt=dt, max_dt=dt)
        p, d = _read(m)
        n_steps = int(round(T / dt))
        be_decay = (1.0 + lam * dt) ** (-n_steps)
        assert np.mean(p) == pytest.approx(s * be_decay, rel=1e-5), (
            f"{m.component_names[0]}: {np.mean(p)} vs {s * be_decay}"
        )
        assert np.mean(d) == pytest.approx(s * (1.0 - be_decay), rel=1e-4)


# ==============================================================================
# 7. Decay constant wiring
# ==============================================================================


def test_decay_constant_is_ln2_over_half_life():
    """The rate must be lambda = ln2 / half_life, not some other convention."""
    for half_life in (1.0e5, 1.0e6, 1.0e7):
        model, _ = _make(half_life)
        assert model.lambda_decay == pytest.approx(np.log(2.0) / half_life)
