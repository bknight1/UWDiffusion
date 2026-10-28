"""Cross-solver coupling tests: 1D cylindrical FV against the 2D UW3 FEM path.

``tests/test_2d_verification.py`` checks the 2D models against exact
solutions and ``tests/test_verification.py`` does the same for
``MulticomponentDiffusion1D``, but nothing checks the two solvers against
*each other*.  These tests do, on the smallest setup that still exercises the
coupling, so they run in seconds.

Also covered here: the two conventions in the 2D growth workflow that are easy
to get wrong and silent when you do -- the shell labelling contract of
``generate_garnet_annulus_mesh``, and ``sync_history_to_state()`` after a
manual field overwrite.
"""

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

import underworld3 as uw

from UWDiffusion import MulticomponentDiffusion1D, MulticomponentDiffusionModel
from UWDiffusion.fd_1d import lasaga_reduced_diffusivity
from UWDiffusion.meshing import generate_garnet_annulus_mesh


# ---------------------------------------------------------------------------
# 1. The two solvers agree with each other
# ---------------------------------------------------------------------------


def test_cylindrical_cross_diffusion_eigenmode_1d_matches_2d() -> None:
    """1D cylindrical FV and 2D annulus FEM agree on a coupled eigenmode.

    Same reduction as ``examples/compare_1d_2d_cylindrical_multicomponent.py``
    at a coarser setting: a radially symmetric Neumann eigenmode of the
    coupled 2x2 system

        D = [[1.0, 0.3], [0.2, 0.5]]

    with Ca by closure, evolved in both solvers with the same Backward Euler
    timestep.  Because both sides are first-order in time at the same ``dt``,
    the temporal errors largely cancel and the residual is the spatial
    discretisation difference.

    The tolerance allows that spatial difference while failing a wrong
    coupling, a transposed matrix, a stale closure, or a mismatched geometry
    weighting -- those move the result by ~1e-2, an order of magnitude more
    than the discretisation gap measured here.
    """
    r_inner, r_outer = 0.5, 1.0
    d_numeric = np.array([[1.0, 0.3], [0.2, 0.5]])
    components = ["Mg", "Fe", "Ca"]
    final_time, dt = 0.02, 0.002
    n_1d, cell_size = 25, 0.05

    def neumann_wavenumber() -> float:
        def det(k: float) -> float:
            return j1(k * r_inner) * y1(k * r_outer) - j1(k * r_outer) * y1(k * r_inner)

        return float(brentq(det, 6.0, 7.0, xtol=1e-14))

    def radial_mode(r: np.ndarray, k: float) -> np.ndarray:
        unscaled = y1(k * r_inner) * j0(k * r) - j1(k * r_inner) * y0(k * r)
        dense = np.linspace(r_inner, r_outer, 20001)
        ref = y1(k * r_inner) * j0(k * dense) - j1(k * r_inner) * y0(k * dense)
        return unscaled / np.max(np.abs(ref))

    k = neumann_wavenumber()
    eigvals, eigvecs = np.linalg.eig(d_numeric)
    idx = int(np.argmax(eigvals.real))
    eigvec = eigvecs[:, idx].real
    eigvec /= np.linalg.norm(eigvec)
    if eigvec[0] < 0:
        eigvec *= -1

    mean = np.array([0.5, 0.25])
    amplitude = 0.15

    def exact(r: np.ndarray, t: float) -> np.ndarray:
        decay = np.exp(-float(eigvals[idx].real) * k**2 * t)
        ind = mean[:, None] + eigvec[:, None] * (amplitude * radial_mode(r, k) * decay)
        return np.vstack((ind, 1.0 - ind.sum(axis=0)))

    # 1D: cylindrical weighting (alpha = 1) is the radial reduction of a
    # Cartesian annulus.
    grid_1d = np.linspace(r_inner, r_outer, n_1d)
    ind0 = mean[:, None] + eigvec[:, None] * (amplitude * radial_mode(grid_1d, k))
    model_1d = MulticomponentDiffusion1D(
        grid=grid_1d,
        diffusivity=d_numeric,
        geometry="cylindrical",
        components=components,
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
        initial_composition=np.vstack((ind0, 1.0 - ind0.sum(axis=0))).T,
    )
    result_1d = model_1d.run_simulation(dt=dt, total_time=final_time)

    # 2D: unconstrained rims give the natural zero-flux condition.
    mesh = uw.meshing.Annulus(
        radiusOuter=r_outer, radiusInner=r_inner, cellSize=cell_size, qdegree=2
    )
    d_sym = sp.Matrix(
        [[sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")],
         [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")]]
    )
    model_2d = MulticomponentDiffusionModel(
        components, mesh, diffusion_matrix=d_sym, degree=2, order=1
    )
    model_2d.set_diffusion_values(
        {"D_MgMg": 1.0, "D_MgFe": 0.3, "D_FeMg": 0.2, "D_FeFe": 0.5}
    )
    for solver in model_2d.solvers:
        solver.petsc_options["snes_rtol"] = 1e-12
        solver.petsc_options["snes_atol"] = 1e-8

    coords = model_2d.mesh_vars[0].coords
    r_nodes = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    init_2d = exact(r_nodes, 0.0)
    with mesh.access(*model_2d.mesh_vars):
        for i, var in enumerate(model_2d.mesh_vars):
            var.data[:, 0] = init_2d[i]

    model_2d.run_simulation(duration=final_time, min_dt=dt, max_dt=dt)

    # Circumferentially average the 2D field to remove polygonal-mesh
    # anisotropy, then compare on common radii.
    r_common = np.linspace(r_inner + 1e-10, r_outer - 1e-10, 61)
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    xy = np.column_stack(
        ((r_common[:, None] * np.cos(angles)).ravel(),
         (r_common[:, None] * np.sin(angles)).ravel())
    )
    prof_2d = np.stack(
        [
            uw.function.evaluate(v.sym[0], xy).reshape(len(r_common), len(angles))
            for v in model_2d.mesh_vars
        ]
    ).mean(axis=2)
    prof_1d = np.vstack(
        [np.interp(r_common, result_1d.r, result_1d.get_profile(n, -1)) for n in components]
    )

    diff = np.max(np.abs(prof_1d - prof_2d), axis=1)
    assert np.all(diff < 2e-3), dict(zip(components, diff))
    # Closure must hold exactly on both sides.
    assert np.max(np.abs(prof_1d.sum(axis=0) - 1.0)) < 1e-12
    assert np.max(np.abs(prof_2d.sum(axis=0) - 1.0)) < 1e-12
    # And both must be recognisably the eigenmode, not just each other.
    prof_exact = exact(r_common, final_time)
    assert np.max(np.abs(prof_1d - prof_exact)) < 5e-3
    assert np.max(np.abs(prof_2d - prof_exact)) < 5e-3


# ---------------------------------------------------------------------------
# 2. The shell labelling contract of the growth mesh
# ---------------------------------------------------------------------------


def test_garnet_annulus_mesh_labels_the_outermost_shells(tmp_path) -> None:
    """``shell_j`` constrains the circle at ``radii_nd[-G + j]``.

    The labelled shells are the ``growth_shell_count`` *outermost* entries of
    ``radii_nd``, with ``shell_0`` the innermost of those and ``shell_{G-1}``
    the outer rim.  Getting this wrong silently pins the wrong radii during a
    growth run -- the model still solves and the profiles still look
    plausible -- so the mapping is checked geometrically here: pin ``shell_0``
    alone to a distinctive value, take one tiny step, and confirm the
    constraint lands at the expected radius and nowhere else.
    """
    radii = np.linspace(0.0, 1.0, 21)  # dr = 0.05
    n_growth = 4
    expected_front = float(radii[-n_growth])  # 0.85 == shell_0

    mesh = generate_garnet_annulus_mesh(
        radii_nd=radii,
        mesh_degree=2,
        total_shell_count=len(radii),
        growth_shell_count=n_growth,
        output_path=tmp_path,
        mesh_name="shell_contract",
    )

    # The Mesh constructor also adds a synthetic "All_Boundaries" label.
    names = sorted(b.name for b in mesh.boundaries if b.name != "All_Boundaries")
    assert names == sorted(["Centre"] + [f"shell_{i}" for i in range(n_growth)])

    d_sym = sp.Matrix(
        [[sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")],
         [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")]]
    )
    model = MulticomponentDiffusionModel(
        ["Mg", "Fe", "Ca"], mesh, diffusion_matrix=d_sym, degree=2, order=1
    )
    model.set_diffusion_values(
        {"D_MgMg": 1.0, "D_MgFe": 0.3, "D_FeMg": 0.2, "D_FeFe": 0.5}
    )
    for solver in model.solvers:
        solver.petsc_options["snes_rtol"] = 1e-12
        solver.petsc_options["snes_atol"] = 1e-10

    vars_ = list(model.mesh_vars)
    start = np.array([0.20, 0.10, 0.70])
    pinned = np.array([0.90, 0.05])
    with mesh.access(*vars_):
        for k in range(3):
            vars_[k].data[:, 0] = start[k]

    # Pin ONLY shell_0.  Every other shell stays free.
    for solver in model.solvers:
        solver.essential_bcs = []
    for k in range(2):
        model.add_dirichlet_bc({"shell_0": float(pinned[k])}, component_index=k)
    model.sync_history_to_state()

    # One very short step: diffusion length sqrt(D dt) ~ 0.01, far below dr.
    model.run_simulation(duration=1e-4, min_dt=1e-4, max_dt=1e-4)

    probe = np.array([0.30, 0.60, expected_front, 0.90, 0.95, 1.00])
    xy = np.column_stack((probe, np.zeros_like(probe)))
    mg = uw.function.evaluate(model.mesh_vars[0].sym[0], xy)

    # shell_0 sits at the innermost growth radius, not at the rim.
    assert mg[2] == pytest.approx(pinned[0], abs=1e-6), (probe, mg)
    # Everything well inside is untouched.
    assert mg[0] == pytest.approx(start[0], abs=1e-3)
    assert mg[1] == pytest.approx(start[0], abs=1e-3)
    # The outer shells are NOT pinned: they have not been reached by the
    # one short step of diffusion either.
    assert np.all(mg[3:] < 0.5 * (start[0] + pinned[0])), (probe, mg)


def test_garnet_annulus_mesh_shell_count_matches_request(tmp_path) -> None:
    """The number of labelled shells is exactly ``growth_shell_count``."""
    radii = np.linspace(0.0, 1.0, 17)

    for n_growth in (1, 3):
        mesh = generate_garnet_annulus_mesh(
            radii_nd=radii,
            mesh_degree=2,
            total_shell_count=len(radii),
            growth_shell_count=n_growth,
            output_path=tmp_path,
            mesh_name=f"shell_count_{n_growth}",
        )
        shells = [b.name for b in mesh.boundaries if b.name.startswith("shell_")]
        assert len(shells) == n_growth
        assert sorted(shells) == sorted(f"shell_{i}" for i in range(n_growth))


# ---------------------------------------------------------------------------
# 3. History sync after a growth stamp
# ---------------------------------------------------------------------------


def test_growth_stamp_requires_history_sync(tmp_path) -> None:
    """A manual field overwrite must be followed by ``sync_history_to_state()``.

    Growth stamps new material straight into ``mesh_vars``.  The time
    derivative history still holds the *pre-stamp* state, so the next solve
    sees a spurious ``dC/dt`` and quietly moves the solution.  A growth driver
    that overwrites fields by hand must therefore call
    ``sync_history_to_state()`` afterwards.

    Setup: a uniform state, uniformly re-stamped to a different composition
    with matching Dirichlet rims.  With no gradients anywhere the exact
    discrete solution is "unchanged".  Synced, the model reproduces that;
    unsynced, it does not.
    """
    before = np.array([0.40, 0.25, 0.35])  # Mg, Fe, Ca
    after = np.array([0.30, 0.35, 0.35])

    def build(tag: str):
        radii = np.linspace(0.0, 1.0, 9)
        mesh = generate_garnet_annulus_mesh(
            radii_nd=radii,
            mesh_degree=2,
            total_shell_count=len(radii),
            growth_shell_count=3,
            output_path=tmp_path,
            mesh_name=f"history_sync_{tag}",
        )
        d_sym = sp.Matrix(
            [[sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")],
             [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")]]
        )
        model = MulticomponentDiffusionModel(
            ["Mg", "Fe", "Ca"], mesh, diffusion_matrix=d_sym, degree=2, order=1
        )
        model.set_diffusion_values(
            {"D_MgMg": 1.0, "D_MgFe": 0.3, "D_FeMg": 0.2, "D_FeFe": 0.5}
        )
        for solver in model.solvers:
            solver.petsc_options["snes_rtol"] = 1e-12
            solver.petsc_options["snes_atol"] = 1e-10
        return mesh, model, list(model.mesh_vars)

    def stamp(mesh, vars_, values) -> None:
        with mesh.access(*vars_):
            for k in range(3):
                vars_[k].data[:, 0] = values[k]

    def pin_rims(mesh, model, values) -> None:
        for solver in model.solvers:
            solver.essential_bcs = []
        for boundary in mesh.boundaries:
            if not boundary.name.startswith("shell_"):
                continue
            for k in range(2):  # independent components only
                model.add_dirichlet_bc(
                    {boundary.name: float(values[k])}, component_index=k
                )

    def rehearse(tag: str, sync: bool) -> np.ndarray:
        """Two steps: hold `before`, then re-stamp to `after` and step again."""
        mesh, model, vars_ = build(tag)
        stamp(mesh, vars_, before)
        pin_rims(mesh, model, before)
        if sync:
            model.sync_history_to_state()
        model.run_simulation(duration=0.01, min_dt=0.01, max_dt=0.01)

        stamp(mesh, vars_, after)
        pin_rims(mesh, model, after)
        if sync:
            model.sync_history_to_state()
        model.run_simulation(duration=0.02, min_dt=0.01, max_dt=0.01)

        with mesh.access(*vars_):
            return np.array([vars_[k].data[0, 0] for k in range(3)])

    held = rehearse("synced", sync=True)
    np.testing.assert_allclose(held, after, atol=1e-8)

    drifted = rehearse("unsynced", sync=False)
    assert not np.allclose(drifted, after, atol=1e-6), (
        "a stale time-derivative history should perturb the solution; if it "
        "does not, this test no longer demonstrates the failure mode"
    )
