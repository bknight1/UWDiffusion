"""Tests for 1D finite-volume multicomponent diffusion model."""

from __future__ import annotations

import numpy as np
import pytest

from UWDiffusion.fd_1d import (
    BoundaryCondition,
    MulticomponentDiffusion1D,
    SimulationResult,
    reduce_diffusion_matrix,
)

# ==============================================================================
# 1. Geometry and Cell Measures
# ==============================================================================


@pytest.mark.parametrize(
    "geometry,alpha", [("planar", 0), ("cylindrical", 1), ("spherical", 2)]
)
def test_measures_exact_domain_integral(geometry: str, alpha: int) -> None:
    """Verify sum of discrete cell volumes equals analytical domain integral."""
    # Test starting at origin r0 = 0
    r = np.linspace(0.0, 1.5, 151)
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=1e-4,
        geometry=geometry,
        components=2,
    )
    p = alpha + 1
    expected_vol = (r[-1] ** p - r[0] ** p) / float(p)
    assert np.isclose(np.sum(model.cell_volumes), expected_vol, rtol=1e-14)

    # Test non-zero inner radius (annulus / shell / offset slab)
    r_offset = np.linspace(0.5, 2.0, 101)
    model_offset = MulticomponentDiffusion1D(
        grid=r_offset,
        diffusivity=1e-4,
        geometry=geometry,
        components=2,
    )
    expected_vol_offset = (r_offset[-1] ** p - r_offset[0] ** p) / float(p)
    assert np.isclose(
        np.sum(model_offset.cell_volumes), expected_vol_offset, rtol=1e-14
    )


def test_non_uniform_grid_measures() -> None:
    """Verify cell measures on a geometrically clustered non-uniform grid."""
    xi = np.linspace(0, 1, 80)
    r = 2.0 * (xi**1.8)  # non-uniform grid clustered near origin
    for geom, alpha in [("planar", 0), ("cylindrical", 1), ("spherical", 2)]:
        model = MulticomponentDiffusion1D(grid=r, diffusivity=1e-4, geometry=geom)
        expected = (r[-1] ** (alpha + 1) - r[0] ** (alpha + 1)) / (alpha + 1)
        assert np.isclose(np.sum(model.cell_volumes), expected, rtol=1e-14)


def test_origin_regularity_no_division_by_zero() -> None:
    """Check that at r=0, all discrete measures and transmissibilities are finite and positive."""
    r = np.linspace(0.0, 1.0, 51)
    for geom in ["planar", "cylindrical", "spherical"]:
        model = MulticomponentDiffusion1D(grid=r, diffusivity=1e-5, geometry=geom)
        assert np.all(np.isfinite(model.cell_volumes))
        assert np.all(model.cell_volumes > 0.0)
        assert np.all(np.isfinite(model.T_faces))
        assert np.all(model.T_faces > 0.0)


# ==============================================================================
# 2. Strict Mass / Species Conservation
# ==============================================================================


@pytest.mark.parametrize("geometry", ["planar", "cylindrical", "spherical"])
def test_mass_conservation_closed_boundary(geometry: str) -> None:
    """Verify that total mass is conserved to machine precision under Neumann zero-flux BCs."""
    r = np.linspace(0.0, 1.0, 81)
    D = 1e-4
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D,
        geometry=geometry,
        components=["A", "B"],
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    )

    # Step function initial profile
    init_A = np.where(r <= 0.4, 0.75, 0.25)
    model.set_initial_composition({"A": init_A})

    initial_mass_A = model.total_mass("A")
    initial_mass_B = model.total_mass("B")

    # Run for 20 steps
    dt = 5.0
    for _ in range(20):
        model.step(dt)

    final_mass_A = model.total_mass("A")
    final_mass_B = model.total_mass("B")

    assert np.isclose(final_mass_A, initial_mass_A, rtol=1e-12, atol=1e-14)
    assert np.isclose(final_mass_B, initial_mass_B, rtol=1e-12, atol=1e-14)


# ==============================================================================
# 3. Analytical Benchmarks
# ==============================================================================


def test_spherical_diffusion_crank_benchmark() -> None:
    """Benchmark spherical diffusion against Crank (1975) Eq. 6.18 exact series solution."""
    R = 1.0
    D = 1e-4
    t_final = 1000.0
    X0 = 0.8
    XR = 0.2

    M = 201
    r = np.linspace(0.0, R, M)

    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D,
        geometry="spherical",
        components=["C", "Matrix"],
        bcs={
            "left": ("neumann", 0.0),
            "right": ("dirichlet", {"C": XR, "Matrix": 1.0 - XR}),
        },
        initial_composition={"C": X0},
    )

    result = model.run_simulation(dt=1.0, total_time=t_final)
    C_num = result.get_profile("C", time_idx=-1)

    # Exact Crank (1975) series solution:
    # C(r, t) = C_R + 2 R (C_0 - C_R) / (pi r) * sum_{n=1}^inf (-1)^{n+1}/n sin(n pi r / R) exp(-D n^2 pi^2 t / R^2)
    # At r=0: C(0, t) = C_R + 2 (C_0 - C_R) * sum_{n=1}^inf (-1)^{n+1} exp(-D n^2 pi^2 t / R^2)
    tau = D * t_final / (R**2)
    n = np.arange(1, 200)
    exp_terms = np.exp(-(n**2) * (np.pi**2) * tau)

    C_exact = np.empty_like(r)
    # For r > 0
    for i in range(1, len(r)):
        terms = ((-1.0) ** (n + 1) / n) * np.sin(n * np.pi * r[i] / R) * exp_terms
        C_exact[i] = XR + 2.0 * R * (X0 - XR) / (np.pi * r[i]) * np.sum(terms)

    # For r = 0
    C_exact[0] = XR + 2.0 * (X0 - XR) * np.sum(((-1.0) ** (n + 1)) * exp_terms)

    # Max difference between numerical and analytical should be < 5e-4
    max_err = np.max(np.abs(C_num - C_exact))
    assert max_err < 5e-4


def test_planar_fourier_mode_decay() -> None:
    """Benchmark planar 1D diffusion against decaying fundamental sine mode."""
    L = 2.0
    D = 2.5e-4
    M = 101
    x = np.linspace(0.0, L, M)

    A0 = 0.3
    base = 0.5
    decay_rate = D * (np.pi / L) ** 2

    model = MulticomponentDiffusion1D(
        grid=x,
        diffusivity=D,
        geometry="planar",
        components=["C", "Bg"],
        bcs={
            "left": ("dirichlet", {"C": base, "Bg": 1.0 - base}),
            "right": ("dirichlet", {"C": base, "Bg": 1.0 - base}),
        },
        initial_composition={"C": base + A0 * np.sin(np.pi * x / L)},
    )

    t_eval = 400.0
    result = model.run_simulation(dt=1.0, total_time=t_eval)
    c_num = result.get_profile("C", -1)

    c_exact = base + A0 * np.sin(np.pi * x / L) * np.exp(-decay_rate * t_eval)
    assert np.max(np.abs(c_num - c_exact)) < 2e-4


def test_steady_state_curved_geometries() -> None:
    """Test steady-state profile in cylindrical (log) and spherical (1/r) shells."""
    r_in, r_out = 1.0, 3.0
    r = np.linspace(r_in, r_out, 61)
    c_in, c_out = 0.8, 0.2
    D = 1e-3

    # 1. Cylindrical: C(r) = C_in + (C_out - C_in) * ln(r / r_in) / ln(r_out / r_in)
    model_cyl = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D,
        geometry="cylindrical",
        components=2,
        bcs={"left": ("dirichlet", c_in), "right": ("dirichlet", c_out)},
        initial_composition=0.5,
    )
    # Run to long time (steady state)
    res_cyl = model_cyl.run_simulation(dt=20.0, total_time=10000.0)
    c_cyl_num = res_cyl.get_profile("C0", -1)
    c_cyl_exact = c_in + (c_out - c_in) * np.log(r / r_in) / np.log(r_out / r_in)
    assert np.max(np.abs(c_cyl_num - c_cyl_exact)) < 1e-3

    # 2. Spherical: C(r) = C_in + (C_out - C_in) * (1/r_in - 1/r) / (1/r_in - 1/r_out)
    model_sph = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D,
        geometry="spherical",
        components=2,
        bcs={"left": ("dirichlet", c_in), "right": ("dirichlet", c_out)},
        initial_composition=0.5,
    )
    res_sph = model_sph.run_simulation(dt=20.0, total_time=10000.0)
    c_sph_num = res_sph.get_profile("C0", -1)
    c_sph_exact = c_in + (c_out - c_in) * (1.0 / r_in - 1.0 / r) / (
        1.0 / r_in - 1.0 / r_out
    )
    assert np.max(np.abs(c_sph_num - c_sph_exact)) < 1e-3


# ==============================================================================
# 4. Multicomponent Diffusion Matrix & Closure Relation
# ==============================================================================


def test_reduce_diffusion_matrix() -> None:
    """Verify matrix reduction on full 4x4 and 3x3 matrices."""
    # 4-component system (e.g. Alm, Prp, Grs, Sps)
    D_full = np.array(
        [
            [4.5e-19, -0.4e-19, 0.1e-19, 0.05e-19],
            [-4.7e-19, 0.66e-19, 0.05e-19, 0.02e-19],
            [0.2e-19, 0.1e-19, 1.2e-19, 0.01e-19],
            [0.1e-19, 0.05e-19, 0.02e-19, 2.0e-19],
        ]
    )
    D_red = reduce_diffusion_matrix(D_full, dependent_idx=-1)
    assert D_red.shape == (3, 3)

    # Check flux identity: sum_{k=0}^3 D_full[i, k] grad(X_k) == sum_{j=0}^2 D_red[i, j] grad(X_j)
    grad_X = np.array([0.02, -0.05, 0.01, 0.02])  # sums to 0
    assert np.isclose(np.sum(grad_X), 0.0)

    flux_full = D_full @ grad_X
    flux_red = D_red @ grad_X[:3]
    assert np.allclose(flux_full[:3], flux_red, rtol=1e-14)


def test_garnet_multicomponent_closure_and_cross_diffusion() -> None:
    """Verify multicomponent coupling in ternary garnet (Mg, Fe, Ca) with closure."""
    # Based on Vielzeuf & Saul (2011) / Gaides et al. (2008)
    D_MgMg = 0.66e-19
    D_MgFe = -4.74e-19
    D_FeMg = -0.40e-19
    D_FeFe = 4.50e-19

    D_matrix = np.array(
        [
            [D_MgMg, D_MgFe],
            [D_FeMg, D_FeFe],
        ]
    )

    r = np.linspace(0.0, 10.0, 101)  # 0 to 10 micrometers
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D_matrix,
        geometry="spherical",
        components=["Mg", "Fe", "Ca"],
        dependent_component="Ca",
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    )

    # Step function in Mg and Fe
    Mg_init = np.where(r <= 5.0, 0.461, 0.457)
    Fe_init = np.where(r <= 5.0, 0.519, 0.510)
    model.set_initial_composition({"Mg": Mg_init, "Fe": Fe_init})

    # Check initial closure
    X_init = model.get_composition()
    assert np.allclose(np.sum(X_init, axis=1), 1.0, atol=1e-14)

    # Advance time
    dt = 1e6
    result = model.run_simulation(dt=dt, n_steps=20, save_interval=5)

    # Check closure at all recorded times
    for comp_step in result.compositions:
        sum_row = np.sum(comp_step, axis=1)
        assert np.allclose(sum_row, 1.0, atol=1e-12)

    # Check result dictionary access
    assert "Mg" in result.profiles
    assert "Fe" in result.profiles
    assert "Ca" in result.profiles
    assert result["Mg"].shape == (5, 101)  # 5 saved times, 101 points


def test_quaternary_garnet_system() -> None:
    """Test 4-component garnet diffusion (Alm, Prp, Grs, Sps)."""
    # 4x4 matrix with diagonal diffusivities
    D_full = np.diag([4.0e-19, 1.0e-19, 0.2e-19, 2.0e-19])
    r = np.linspace(0.0, 500.0, 51)  # 500 um radius garnet

    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D_full,
        geometry="spherical",
        components=["Alm", "Prp", "Grs", "Sps"],
        dependent_component="Sps",
        bcs={
            "left": ("neumann", 0.0),
            "right": (
                "dirichlet",
                {"Alm": 0.65, "Prp": 0.20, "Grs": 0.10, "Sps": 0.05},
            ),
        },
        initial_composition={
            "Alm": 0.70,
            "Prp": 0.15,
            "Grs": 0.08,
            "Sps": 0.07,
        },
    )

    result = model.run_simulation(dt=1e10, n_steps=10)
    assert result.compositions.shape == (2, 51, 4)
    # Check closure
    assert np.allclose(np.sum(result.compositions[-1], axis=1), 1.0, atol=1e-12)


# ==============================================================================
# 5. Composition-dependent / Callable Diffusivity
# ==============================================================================


def test_callable_diffusivity_picard() -> None:
    """Test nonlinear diffusion with composition-dependent D(X) via Picard iteration."""
    r = np.linspace(0.0, 1.0, 61)

    # D increases with concentration: D(X) = 1e-4 * (1.0 + 3.0 * X['C0'])
    def D_func(comp_dict: dict[str, float]) -> float:
        return 1e-4 * (1.0 + 3.0 * comp_dict["C0"])

    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D_func,
        geometry="spherical",
        components=["C0", "C1"],
        bcs={
            "left": ("neumann", 0.0),
            "right": ("dirichlet", {"C0": 0.2, "C1": 0.8}),
        },
        initial_composition={"C0": 0.8, "C1": 0.2},
        picard_max_iter=30,
        picard_tol=1e-8,
    )

    result = model.run_simulation(dt=5.0, total_time=100.0)
    c0 = result.get_profile("C0", -1)
    c1 = result.get_profile("C1", -1)

    # Profile should have diffused inward from right
    assert c0[-1] == pytest.approx(0.2)
    assert c0[0] < 0.8  # diffused
    assert np.allclose(c0 + c1, 1.0)


# ==============================================================================
# 6. Boundary Conditions, Output Options, and Error Handling
# ==============================================================================


def test_output_times_option() -> None:
    """Verify run_simulation with arbitrary specified output times."""
    r = np.linspace(0, 1, 51)
    model = MulticomponentDiffusion1D(grid=r, diffusivity=1e-4, components=2)

    requested_times = [10.0, 25.5, 50.0, 100.0]
    result = model.run_simulation(dt=4.0, output_times=requested_times)

    np.testing.assert_allclose(result.times, [0.0, 10.0, 25.5, 50.0, 100.0])
    assert len(result.compositions) == 5


def test_single_component_tracer() -> None:
    """Verify single-component (N=1) tracer diffusion works without closure requirement."""
    r = np.linspace(0, 1, 51)
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=1e-3,
        components=1,
        bcs={"left": ("neumann", 0.0), "right": ("dirichlet", 0.0)},
        initial_composition=1.0,
    )
    result = model.run_simulation(dt=1.0, total_time=20.0)
    assert result.compositions.shape[-1] == 1
    assert result.get_profile(0, -1)[-1] == 0.0


def test_invalid_inputs_raise_errors() -> None:
    """Verify appropriate validation errors for incorrect grid or parameters."""
    # Decreasing or non-strictly increasing grid
    with pytest.raises(ValueError, match="strictly increasing"):
        MulticomponentDiffusion1D(grid=[0.0, 1.0, 0.5], diffusivity=1e-4)

    # Negative coordinates
    with pytest.raises(ValueError, match="non-negative"):
        MulticomponentDiffusion1D(grid=[-1.0, 0.0, 1.0], diffusivity=1e-4)

    # Unknown geometry
    with pytest.raises(ValueError, match="Unknown geometry"):
        MulticomponentDiffusion1D(
            grid=[0.0, 1.0], diffusivity=1e-4, geometry="hyperbolic"
        )

    # Incompatible D matrix shape
    with pytest.raises(ValueError, match="Invalid diffusivity shape"):
        MulticomponentDiffusion1D(grid=[0.0, 1.0], diffusivity=np.eye(5), components=3)

    # Negative time step
    model = MulticomponentDiffusion1D(grid=[0.0, 1.0], diffusivity=1e-4)
    with pytest.raises(ValueError, match="positive"):
        model.step(-1.0)


def test_boundary_condition_class() -> None:
    """Verify BoundaryCondition object creation, normalization, and error checks."""
    bc_d = BoundaryCondition(bc_type="DIRICHLET", value={"C0": 0.5})
    assert bc_d.bc_type == "dirichlet"

    bc_n = BoundaryCondition(bc_type="zero_flux", value=0.0)
    assert bc_n.bc_type == "neumann"

    with pytest.raises(ValueError, match="Unknown bc_type"):
        BoundaryCondition(bc_type="robin", value=1.0)


# ==============================================================================
# 7. Internal point Dirichlet constraints (pins)
# ==============================================================================


def test_internal_pin_holds_vertex_and_forces_neighbours() -> None:
    """An internal point Dirichlet pin is honoured; clearing it re-enables diffusion."""
    r = np.linspace(0.0, 1.0, 11)
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=1e-3,
        geometry="cylindrical",
        components=["A", "B"],
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
        initial_composition={"A": np.linspace(0.9, 0.1, 11)},
    )
    X0 = model.get_composition("A").copy()

    # Pin the vertex nearest r = 0.6 to A = 0.9 (a step above its neighbours).
    model.set_internal_bcs(0.6, {"A": 0.9})
    assert X0[6] != pytest.approx(0.9)  # the gradient makes the test meaningful
    model.step(1.0)
    X1 = model.get_composition("A")

    # The pinned vertex is a Dirichlet constraint: unchanged, exactly 0.9.
    assert X1[6] == pytest.approx(0.9, abs=1e-14)
    # Material flows in from both sides toward the new pin value.
    assert X1[8] > X0[8]  # right side (below 0.9) gains A
    assert X1[4] > X0[4]  # left side (below 0.9) gains A

    # Clearing the pin returns the vertex to the ordinary solve.
    model.clear_internal_bcs()
    model.step(1.0)
    X2 = model.get_composition("A")
    assert X2[6] < 0.9  # released vertex relaxes back toward its surroundings

    # Closure is still enforced everywhere, including the pinned vertex.
    X_full = model.get_composition()
    assert np.allclose(np.sum(X_full, axis=1), 1.0, atol=1e-12)


def test_internal_pin_position_forms() -> None:
    """Pins can be specified by index, radius, radii array, or boolean mask."""
    r = np.linspace(0.0, 1.0, 6)
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=1e-3,
        geometry="planar",
        components=["A", "B"],
        initial_composition={"A": 0.5},
    )
    # Index form (vertex 3, r = 0.6).
    model.set_internal_bcs(3, {"A": 0.9})
    assert model._internal_bcs == {3: {0: 0.9}}
    # Radii (array) form — matched to nearest vertices.
    model.set_internal_bcs([0.2, 0.6], {"A": 0.8})
    assert set(model._internal_bcs) == {1, 3}
    # Boolean mask form.
    model.set_internal_bcs(np.array([False, True, False, True, False, False]), {"A": 0.7})
    assert set(model._internal_bcs) == {1, 3}
    # Replace (not accumulate), then clear.
    model.set_internal_bcs([], {"A": 0.6})
    assert model._internal_bcs == {}
    model.set_internal_bcs([0, 5], {"A": 0.5})
    model.clear_internal_bcs()
    assert model._internal_bcs == {}


def test_internal_pin_all_independent_components_and_scalar_value() -> None:
    """Pins cover every independent component; the dependent one follows closure."""
    r = np.linspace(0.0, 1.0, 6)
    D = np.diag([4.0e-3, 1.0e-3, 0.5e-3, 2.0e-3])  # full 4x4, reduced on Ca
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D,
        geometry="spherical",
        components=["Fe", "Mg", "Mn", "Ca"],
        dependent_component="Ca",
        initial_composition={"Fe": 0.6, "Mg": 0.15, "Mn": 0.1},
    )
    # Dict value: only the independent components are pinned; Ca is closure.
    model.set_internal_bcs(2, {"Fe": 0.5, "Mg": 0.2, "Mn": 0.05})
    model.step(1.0)
    X = model.get_composition()
    assert X[2, 0] == pytest.approx(0.5)
    assert X[2, 1] == pytest.approx(0.2)
    assert X[2, 2] == pytest.approx(0.05)
    assert X[2, 3] == pytest.approx(1.0 - 0.5 - 0.2 - 0.05)

    # Scalar value: applied to every independent component.
    model.set_internal_bcs(4, 0.25)
    model.step(1.0)
    X = model.get_composition()
    assert X[4, 0] == pytest.approx(0.25)
    assert X[4, 1] == pytest.approx(0.25)
    assert X[4, 2] == pytest.approx(0.25)
    assert X[4, 3] == pytest.approx(0.25)


def test_internal_pin_validation_errors() -> None:
    """Out-of-range indices/radii and bad masks are rejected."""
    r = np.linspace(0.0, 1.0, 6)
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=1e-3,
        geometry="planar",
        components=["A", "B"],
        initial_composition={"A": 0.5},
    )
    with pytest.raises(ValueError, match="indices"):
        model.set_internal_bcs([0, 6], {"A": 0.5})
    with pytest.raises(ValueError, match="within the grid"):
        model.set_internal_bcs(1.5, {"A": 0.5})
    with pytest.raises(ValueError, match="mask"):
        model.set_internal_bcs(np.array([True, False]), {"A": 0.5})
    with pytest.raises(TypeError, match="value"):
        model.set_internal_bcs([1], None)
    # Missing independent component in the dict is rejected like set_bc values.
    with pytest.raises(KeyError, match="A"):
        model.set_internal_bcs([1], {"B": 0.5})


def test_simulation_result_plot() -> None:
    """Verify SimulationResult.plot runs without error."""
    import matplotlib

    matplotlib.use("Agg")

    r = np.linspace(0.0, 1.0, 31)
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=1e-4,
        components=["A", "B"],
        bcs={
            "left": BoundaryCondition("neumann", 0.0),
            "right": ("dirichlet", {"A": 0.3, "B": 0.7}),
        },
        initial_composition={"A": 0.8, "B": 0.2},
    )
    result = model.run_simulation(dt=2.0, n_steps=5, save_interval=2)
    assert isinstance(result, SimulationResult)
    ax = result.plot(show=False)
    assert ax is not None
