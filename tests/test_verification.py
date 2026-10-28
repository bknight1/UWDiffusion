"""Strong numerical verification suite for ``MulticomponentDiffusion1D``.

These tests verify the solver against exact eigenmode solutions of

    dX_i/dt = 1/r^alpha * d/dr [ r^alpha sum_j D_ij dX_j/dr ],   sum_k X_k = 1

on the vertex-centered finite-volume discretization, in three geometries,
plus explicit spatial and temporal convergence studies.

Mathematical validity for this discretization
---------------------------------------------
The scheme is vertex-centered: unknowns live on vertices ``r_k``, control
volumes span ``[r_{k-1/2}, r_{k+1/2}]`` with faces at vertex midpoints (the
outer faces coincide with ``r[0]`` and ``r[-1]``), interface
transmissibilities are ``T = r_face^alpha / (r_{k+1} - r_k)``, and time
stepping is backward Euler.  Consequences used here:

- A Dirichlet condition *pins the boundary vertex* to the prescribed value
  (row replacement), so boundary nodes are exact and only interior/truncation
  error remains (second order for these smooth modes).
- Zero-flux Neumann conditions are the natural FV boundary (no flux term
  across the outer face); at ``r = 0`` the control volume ``[0, r_{1/2}]``
  enforces the symmetry condition without any singular division.  For the
  modes used here the semi-discrete flux balance at the origin node is exact
  to leading order, so no spurious first-order boundary error appears.
- The closure component is defined by ``X_dep = 1 - sum(independent)`` after
  every step, so closure holds to machine precision.
- Backward Euler is first-order in time; the temporal study below measures
  error against the *exact* solution on a grid fine enough that the spatial
  error (the semi-discrete "spatial floor", measured ~4e-5 on M = 161) is
  negligible compared with the temporal errors; the temporal test verifies
  this separation explicitly with a fine-dt reference run.

No existing benchmark tolerances are modified by this suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0, jn_zeros

from UWDiffusion.fd_1d import MulticomponentDiffusion1D

# ==============================================================================
# Helpers
# ==============================================================================


def _linf(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum norm of the difference."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _l2_volume(model: MulticomponentDiffusion1D, num: np.ndarray, exact: np.ndarray) -> float:
    """Volume-weighted (finite-volume natural) L2 norm of the error."""
    w = model.cell_volumes
    diff = np.asarray(num) - np.asarray(exact)
    return float(np.sqrt(np.sum(w * diff**2) / np.sum(w)))


def _observed_orders(errors) -> np.ndarray:
    """Convergence orders from successive halvings: log2(e_k / e_{k+1})."""
    e = np.asarray(errors, dtype=float)
    return np.log2(e[:-1] / e[1:])


def _make_spherical_mode_model(M: int, R: float = 1.0, D: float = 1.0):
    """Scalar (N=1) spherical model carrying the fundamental radial eigenmode.

    X(r, t) = sin(pi r / R) / r * exp(-D pi^2 t / R^2),  X(0, t) = (pi/R) exp(...)
    with the exact r->0 limit handled via the sinc representation
    sin(pi r / R)/r = (pi/R) * sinc(r/R).
    """
    r = np.linspace(0.0, R, M)
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D,
        geometry="spherical",
        components=1,
        bcs={"left": ("neumann", 0.0), "right": ("dirichlet", 0.0)},
    )
    model.set_initial_composition(lambda rr: (np.pi / R) * np.sinc(rr / R))
    return model, r


# ==============================================================================
# 1. Exact coupled planar eigenmode (non-symmetric positive-stable 2x2 matrix)
# ==============================================================================


def test_planar_coupled_nonsymmetric_eigenmode() -> None:
    """Exact planar eigenmode for a non-symmetric, positive-stable reduced matrix.

    Reduced diffusion matrix (P = 2 independent components):

        D = [[0.8, 0.3], [-0.1, 0.4]]

    has real, distinct, positive eigenvalues {0.7, 0.5} (positive stable) and
    is not symmetric.  The eigenvector for lambda = 0.7 is real: v = (3, -1).
    With the Neumann cosine mode phi(r) = cos(pi r / L) (zero gradient, hence
    zero flux, at r = 0 and r = L) the exact solution is

        X(r, t) = x_bar + a v cos(pi r / L) exp(-lambda pi^2 t / L^2),

    with the third (dependent) component given by closure; it is non-trivial
    because sum(v) != 0.
    """
    D = np.array([[0.8, 0.3], [-0.1, 0.4]])
    lam = 0.7
    v = np.array([3.0, -1.0])

    # Matrix properties required by the construction of this test.
    eigvals = np.linalg.eigvals(D)
    assert np.allclose(np.imag(eigvals), 0.0), f"expected real spectrum, got {eigvals}"
    assert np.all(np.real(eigvals) > 0.0), f"expected positive-stable matrix, got {eigvals}"
    assert not np.allclose(D, D.T), "matrix must be non-symmetric"
    np.testing.assert_allclose(D @ v, lam * v, rtol=1e-14)

    L = 1.0
    M = 101
    r = np.linspace(0.0, L, M)
    sigma = np.pi / L
    a = 0.03
    x0_bar, x1_bar = 0.30, 0.35

    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D,
        geometry="planar",
        components=["C0", "C1", "C2"],
        dependent_component="C2",
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    )
    # Independent components carry the eigenmode; C2 (closure) is filled in.
    model.set_initial_composition(
        {
            "C0": x0_bar + 3.0 * a * np.cos(sigma * r),
            "C1": x1_bar - a * np.cos(sigma * r),
        }
    )

    # Initial closure must hold exactly.
    X_init = model.get_composition()
    assert np.allclose(np.sum(X_init, axis=1), 1.0, atol=1e-14)

    result = model.run_simulation(dt=1e-4, output_times=[0.05])
    T = result.times[-1]
    X = result.compositions[-1]

    mode = np.cos(sigma * r) * np.exp(-lam * sigma**2 * T)
    exact0 = x0_bar + 3.0 * a * mode
    exact1 = x1_bar - a * mode
    exact2 = 1.0 - exact0 - exact1  # closure component, non-constant (sum(v) != 0)

    # Closure holds to machine precision at the final time.
    assert _linf(X[:, 2] - (1.0 - X[:, 0] - X[:, 1]), np.zeros(M)) < 1e-12

    err0 = _linf(X[:, 0], exact0)
    err1 = _linf(X[:, 1], exact1)
    err2 = _linf(X[:, 2], exact2)
    # Scale-aware tolerance: 4e-4 (0.04%) of the mode amplitude 3*a = 0.09,
    # i.e. ~3.8x the measured Linf errors (9.4e-6, 3.1e-6, 6.3e-6).  This is
    # tight enough to catch order-of-magnitude regressions while leaving a
    # reasonable safety factor for environment variation.
    tol = 4e-4 * (3.0 * a)
    assert err0 < tol, f"C0 eigenmode decay error {err0:.3e} >= {tol}"
    assert err1 < tol, f"C1 eigenmode decay error {err1:.3e} >= {tol}"
    assert err2 < tol, f"C2 (closure) error {err2:.3e} >= {tol}"


# ==============================================================================
# 2. Cylindrical Bessel (J0) eigenmode on a disk, binary with closure
# ==============================================================================


def test_cylindrical_bessel_j0_mode() -> None:
    """Binary J0 eigenmode on [0, R]: regular at r=0, Dirichlet at R.

    For the binary system the independent component Fe solves

        dFe/dt = D (1/r)(r Fe')',

    with exact solution

        Fe(r, t) = J0(zeta1 r / R) exp(-D zeta1^2 t / R^2),

    where zeta1 = jn_zeros(0, 1)[0] is the first zero of J0, so Fe is regular
    at r = 0 (J0'(0) = 0, matching the symmetry condition) and Fe(R, t) = 0
    (Dirichlet).  The dependent component Mg follows by closure,
    Mg = 1 - Fe, and must track 1 - Fe_exact.
    """
    R = 1.0
    D = 1.0
    zeta1 = float(jn_zeros(0, 1)[0])
    M = 161
    r = np.linspace(0.0, R, M)

    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D,
        geometry="cylindrical",
        components=["Fe", "Mg"],
        dependent_component="Mg",
        bcs={"left": ("neumann", 0.0), "right": ("dirichlet", 0.0)},
    )
    # Exact J0 profile for the independent component; closure fills Mg.
    model.set_initial_composition({"Fe": lambda rr: j0(zeta1 * rr / R)})

    # Initial condition: regular at the origin, zero at the rim, closed.
    fe_init = model.get_composition("Fe")
    assert fe_init[0] == pytest.approx(1.0, rel=1e-14)
    assert np.allclose(np.sum(model.get_composition(), axis=1), 1.0, atol=1e-14)

    result = model.run_simulation(dt=1e-4, output_times=[0.005, 0.01, 0.02])

    # Scale-aware tolerance: 1e-4 of the unit J0 amplitude, ~3x the measured
    # maximum error over all recorded times (3.2e-5 at t = 0.02).  The error
    # grows with elapsed time (decaying-mode phase error), so the last time
    # level dominates.
    tol = 1e-4
    for i, T in enumerate(result.times):
        if i == 0:
            continue
        fe_exact = j0(zeta1 * r / R) * np.exp(-D * zeta1**2 * T / R**2)
        fe_num = result.get_profile("Fe", time_idx=i)
        mg_num = result.get_profile("Mg", time_idx=i)

        err_fe = _linf(fe_num, fe_exact)
        err_mg = _linf(mg_num, 1.0 - fe_exact)
        assert err_fe < tol, f"Fe transient error at t={T}: {err_fe:.3e} >= {tol}"
        assert err_mg < tol, f"Mg (closure) transient error at t={T}: {err_mg:.3e} >= {tol}"

        # Closure and the Dirichlet pin at the rim.
        assert _linf(fe_num + mg_num, np.ones(M)) < 1e-12
        assert fe_num[-1] == pytest.approx(0.0, abs=1e-15)
        assert mg_num[-1] == pytest.approx(1.0, abs=1e-15)


# ==============================================================================
# 3. Spherical radial eigenmode, scalar single-component
# ==============================================================================


def test_spherical_radial_eigenmode() -> None:
    """Scalar spherical eigenmode on [0, R]: regular at origin, Dirichlet at R.

    Exact solution of dX/dt = D (1/r^2)(r^2 X')':

        X(r, t) = sin(pi r / R) / r * exp(-D pi^2 t / R^2),
        X(0, t) = (pi / R) * exp(-D pi^2 t / R^2)   (regular limit),

    which is regular at r = 0 (X'(0) = 0) and vanishes at r = R (Dirichlet).
    The sinc representation reproduces the analytic origin limit exactly.
    """
    R = 1.0
    D = 1.0
    M = 161

    model, r = _make_spherical_mode_model(M, R=R, D=D)

    # Initial condition carries the exact analytic origin limit.
    x_init = model.get_composition()[:, 0]
    assert x_init[0] == pytest.approx(np.pi / R, rel=1e-15)
    assert x_init[-1] == pytest.approx(0.0, abs=1e-15)

    result = model.run_simulation(dt=1e-4, output_times=[0.005, 0.01, 0.02])

    # Scale-aware tolerance: 1e-3 absolute is ~3e-4 (0.03%) of the peak mode
    # amplitude pi, ~3.7x the measured maximum over recorded times (2.7e-4 at
    # t = 0.02, growing with elapsed time as in the cylindrical case).
    tol = 1e-3
    for i, T in enumerate(result.times):
        if i == 0:
            continue
        x_exact = (np.pi / R) * np.sinc(r / R) * np.exp(-D * np.pi**2 * T / R**2)
        x_num = result.get_profile(0, time_idx=i)

        err = _linf(x_num, x_exact)
        assert err < tol, f"spherical eigenmode transient error at t={T}: {err:.3e} >= {tol}"
        # Dirichlet pin at the rim holds exactly.
        assert x_num[-1] == pytest.approx(0.0, abs=1e-15)


# ==============================================================================
# 4. Spatial convergence study (spherical eigenmode)
# ==============================================================================


def test_spatial_convergence_spherical_eigenmode() -> None:
    """Second-order spatial convergence on grids 21, 41, 81 (h halving).

    Backward Euler with dt = 2e-6 over T = 0.005 keeps the temporal error far
    below the spatial error on the finest grid, so the observed order reflects
    the FV discretization.  Measured volume-weighted L2 errors are
    1.43e-4 / 3.62e-5 / 9.48e-6 with orders ~1.98 / ~1.93 (Linf: ~1.96 / ~1.92).

    Regime separation is verified two independent ways:

    - the finest spatial error must materially exceed the *estimated*
      time-discretization contribution.  For a pure eigenmode of decay rate
      lam = D pi^2 / R^2, backward Euler damps the amplitude by
      (1 + lam*dt)^(-T/dt) instead of exp(-lam*T), giving a direct estimate
      of the temporal error scale; here ~5.4e-7 in L2 versus ~9.5e-6 measured
      spatial error (ratio ~18).
    - repeating the finest grid with dt refined 2x barely changes the error
      (measured ratio 0.97), demonstrating the observed order is spatial and
      not an artifact of the time discretization.
    """
    R = 1.0
    D = 1.0
    T = 0.005
    dt = 2e-6
    n_steps = int(round(T / dt))

    errs_l2: list[float] = []
    errs_inf: list[float] = []
    for M in (21, 41, 81):
        model, r = _make_spherical_mode_model(M, R=R, D=D)
        model.run_simulation(dt=dt, n_steps=n_steps)

        T_actual = model.time
        x_exact = (np.pi / R) * np.sinc(r / R) * np.exp(-D * np.pi**2 * T_actual / R**2)
        x_num = model.get_composition()[:, 0]

        errs_l2.append(_l2_volume(model, x_num, x_exact))
        errs_inf.append(_linf(x_num, x_exact))

    # Errors must decrease and converge at ~second order.
    assert errs_l2[0] > errs_l2[1] > errs_l2[2], f"non-monotone L2 errors: {errs_l2}"
    orders_l2 = _observed_orders(errs_l2)
    orders_inf = _observed_orders(errs_inf)

    assert np.all(orders_l2 > 1.7), (
        f"spatial L2 orders {orders_l2} below 1.7 (errors: {errs_l2})"
    )
    assert np.all(orders_inf > 1.7), (
        f"spatial Linf orders {orders_inf} below 1.7 (errors: {errs_inf})"
    )

    # Regime separation (1): finest spatial error >> estimated temporal
    # contribution, computed from the exact backward-Euler amplification
    # factor of the eigenmode and carried in the same volume-weighted L2 norm.
    lam = D * np.pi**2 / R**2
    rel_be = abs((1.0 + lam * dt) ** (-T / dt) - np.exp(-lam * T))
    x_exact_T = (np.pi / R) * np.sinc(r / R) * np.exp(-D * np.pi**2 * T / R**2)
    w = model.cell_volumes  # finest-grid (M = 81) volumes
    l2_mode = float(np.sqrt(np.sum(w * x_exact_T**2) / np.sum(w)))
    est_temporal_l2 = rel_be * l2_mode
    assert errs_l2[-1] > 4.0 * est_temporal_l2, (
        f"finest spatial error {errs_l2[-1]:.3e} not materially above the "
        f"estimated temporal contribution {est_temporal_l2:.3e}; the observed "
        f"order may be contaminated by time discretization"
    )

    # Regime separation (2): refining dt on the finest grid leaves the error
    # essentially unchanged, so the observed order is spatial.  With temporal
    # error rho x spatial error, halving dt rescales the total error by
    # (1 + rho/2)/(1 + rho); the 0.9 threshold only fails if rho > 0.25
    # (measured rho ~ 0.06).
    model_ref, r_ref = _make_spherical_mode_model(81, R=R, D=D)
    model_ref.run_simulation(dt=dt / 2.0, n_steps=2 * n_steps)
    x_exact_ref = (
        (np.pi / R)
        * np.sinc(r_ref / R)
        * np.exp(-D * np.pi**2 * model_ref.time / R**2)
    )
    err_l2_ref = _l2_volume(model_ref, model_ref.get_composition()[:, 0], x_exact_ref)
    assert err_l2_ref > 0.9 * errs_l2[-1], (
        f"halving dt changed the finest-grid L2 error from {errs_l2[-1]:.3e} "
        f"to {err_l2_ref:.3e}: temporal error contaminates the spatial study"
    )
    order_refined = float(np.log2(errs_l2[1] / err_l2_ref))
    assert order_refined > 1.7, (
        f"order recomputed with the dt-refined finest error is "
        f"{order_refined:.3f} < 1.7 (errors: {errs_l2[1]}, {err_l2_ref})"
    )


# ==============================================================================
# 5. Temporal convergence study (backward Euler, fixed fine grid)
# ==============================================================================


def test_temporal_convergence_backward_euler() -> None:
    """First-order temporal convergence under dt halving on a fixed fine grid.

    The spherical eigenmode is integrated on a fixed M = 161 grid to T = 0.1
    with dt = 0.01, 0.005, 0.0025, 0.00125.  Each run is compared with the
    *exact* solution at its own final time, so the measured errors are
    backward-Euler temporal errors plus a small semi-discrete spatial floor.
    Measured errors are 5.48e-2 / 2.80e-2 / 1.42e-2 / 7.14e-3 with orders
    ~0.97 / ~0.98 / ~0.99 (matching the pure-mode amplification-factor
    estimate to ~0.6%).

    Regime separation: the finest temporal error must remain well above the
    spatial floor of the M = 161 grid, otherwise the study would measure
    spatial rather than temporal error.  A reference run with dt refined 125x
    has its own temporal error only ~1/125 of the finest study error, so its
    total error bounds the dt-independent spatial floor from above; the
    finest temporal error must exceed that bound by >20x (measured ratio
    ~70, which pins the floor below ~5% of the finest temporal error).  The
    order staying near unity at every halving corroborates this: a
    floor-dominated error sequence would show the order collapsing to 0.
    """
    R = 1.0
    D = 1.0
    T = 0.1
    M = 161
    dts = (0.01, 0.005, 0.0025, 0.00125)

    errs: list[float] = []
    for dt in dts:
        model, r = _make_spherical_mode_model(M, R=R, D=D)
        model.run_simulation(dt=dt, n_steps=int(round(T / dt)))

        x_exact = (
            (np.pi / R)
            * np.sinc(r / R)
            * np.exp(-D * np.pi**2 * model.time / R**2)
        )
        errs.append(_linf(model.get_composition()[:, 0], x_exact))

    # Temporal errors must decrease monotonically under dt halving.
    assert errs[0] > errs[1] > errs[2] > errs[3], f"non-monotone temporal errors: {errs}"
    orders = _observed_orders(errs)

    # Backward Euler is first order: observed order in (0.8, 1.2).
    assert np.all(orders > 0.8), f"temporal orders {orders} below 0.8 (errors: {errs})"
    assert np.all(orders < 1.2), f"temporal orders {orders} above 1.2 (errors: {errs})"

    # Regime separation: finest temporal error >> spatial floor.  The
    # reference run below uses dt / 125 on the same grid; since its temporal
    # error is ~1/125 of errs[-1] (first-order scheme), its total error is an
    # upper bound for the spatial floor, and errs[-1] > 20 x (that bound)
    # guarantees the floor is a small fraction of the finest temporal error.
    dt_ref = dts[-1] / 125.0
    model_ref, r_ref = _make_spherical_mode_model(M, R=R, D=D)
    model_ref.run_simulation(dt=dt_ref, n_steps=int(round(T / dt_ref)))
    x_exact_ref = (
        (np.pi / R)
        * np.sinc(r_ref / R)
        * np.exp(-D * np.pi**2 * model_ref.time / R**2)
    )
    err_floor_bound = _linf(model_ref.get_composition()[:, 0], x_exact_ref)
    assert errs[-1] > 20.0 * err_floor_bound, (
        f"finest temporal error {errs[-1]:.3e} not above the spatial-floor "
        f"bound {err_floor_bound:.3e} (dt={dt_ref} reference run); the "
        f"temporal study is contaminated by spatial error"
    )
