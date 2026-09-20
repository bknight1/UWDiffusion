"""End-to-end benchmarks for the 1D finite-volume diffusion model.

Reproduces the two 2D Underworld3 reference benchmarks in 1D:

1. MC (garnet) coupled diffusion -- Tutorials/Diffusion/MC_diffusion-garnet_example.ipynb
2. CZGM garnet growth + diffusion -- Tutorials/Diffusion/CZGM_garnet_model_annulus.ipynb

and unit-tests the supporting API (time-dependent diffusivities, growth
steps, the Darken/CZGM reduced-matrix helper, run_to_time).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy import interpolate

from UWDiffusion import MulticomponentDiffusion1D, darken_reduced_diffusivity
from UWDiffusion.fd_1d import reduce_diffusion_matrix

CZGM_DATA = (
    Path(__file__).resolve().parent.parent
    / "Tutorials"
    / "Diffusion"
    / "CZGM_data"
)

MYR_SEC = 365.25 * 86400.0 * 1e6  # seconds per million years

# ----------------------------------------------------------------------
# Reference data extracted from MC_diffusion-garnet_example.ipynb
# (x in micrometers, values in percent; divide by 100 for mole fractions)
# ----------------------------------------------------------------------
MC_MG_PAPER = np.array(
    [
        (3.0554956807897984, 46.12225405069445),
        (4.017667626491156, 46.12552537491616),
        (4.739296585767175, 46.157829701605515),
        (5.329720279720281, 46.27859608745685),
        (5.635865898807076, 46.37469123646949),
        (5.902649938296998, 46.14635280579436),
        (6.148191220544162, 45.51413824448076),
        (6.483233237350886, 45.49124627699181),
        (6.794845742492803, 45.58861928702855),
        (7.155660222130812, 45.65745340086029),
        (7.5164747017688205, 45.68812206543879),
        (7.9975606746195, 45.699844310566576),
        (8.478646647470178, 45.702843024436476),
        (8.958732620320857, 45.702843024436476),
    ]
)

MC_FE_PAPER = np.array(
    [
        (3.0882969971205263, 51.91246621815881),
        (3.809925956396545, 51.91246621815881),
        (4.411283422459894, 51.90223380954621),
        (4.892369395310572, 51.86186768495721),
        (5.242250102838339, 51.79231882391275),
        (5.592130810366106, 51.678479016239926),
        (5.923907856849034, 51.49709799608993),
        (6.204422048539697, 51.305445920303605),
        (6.570703414232827, 51.161014317750556),
        (6.9041834635952295, 51.07834303009393),
        (7.256493898258605, 51.030536228828446),
        (7.614878650761005, 51.008199800314685),
        (8.095964623611684, 51.00308205393594),
        (8.577050596462364, 51.00053475935829),
    ]
)


# ==========================================================================
# 1. MC (ternary garnet) coupled-diffusion benchmark
# ==========================================================================


def test_mc_garnet_planar_benchmark() -> None:
    """Reproduce the MC_diffusion-garnet_example benchmarking step at 618400 s.

    Planar slab x in [0, 12] um, components Mg/Fe with Ca implicit (closure),
    constant reduced diffusion matrix (m2/s):
        [[ 0.66e-19, -4.74e-19],
         [-0.40e-19,  4.50e-19]]
    Initial Heaviside step at x = 6 um, zero-flux boundaries, integrated in
    two segments: 1e5 s then 518400 s (total 618400 s).
    """
    D_matrix = np.array(
        [
            [0.66e-19, -4.74e-19],
            [-0.40e-19, 4.50e-19],
        ]
    )
    x = np.linspace(0.0, 12e-6, 801)
    left, right = 6e-6, 6e-6
    Mg_init = np.where(x <= right, 0.461185, 0.4569965)
    Fe_init = np.where(x <= left, 0.5190967, 0.50997849)

    model = MulticomponentDiffusion1D(
        grid=x,
        diffusivity=D_matrix,
        geometry="planar",
        components=["Mg", "Fe", "Ca"],
        dependent_component="Ca",
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    )
    model.set_initial_composition({"Mg": Mg_init, "Fe": Fe_init})

    # Two-stage benchmark: 1e5 s then an extra 518400 s (5.184e5).
    r1 = model.run_simulation(dt=1e4, total_time=1e5)
    r2 = model.run_simulation(dt=1e4, total_time=5.184e5)
    assert r1.times[-1] == pytest.approx(1e5)
    assert r2.times[-1] == pytest.approx(618400.0)

    X = r2.compositions[-1]

    # Closure is exact at every point
    assert np.allclose(np.sum(X, axis=1), 1.0, atol=1e-12)

    # Compare with the paper/UW reference profiles
    x_um = x * 1e6
    Mg_num = np.interp(MC_MG_PAPER[:, 0], x_um, X[:, 0])
    Fe_num = np.interp(MC_FE_PAPER[:, 0], x_um, X[:, 1])
    Mg_err = np.max(np.abs(Mg_num - MC_MG_PAPER[:, 1] / 100.0))
    Fe_err = np.max(np.abs(Fe_num - MC_FE_PAPER[:, 1] / 100.0))
    assert Mg_err < 1e-3, f"Mg mismatch vs benchmark: {Mg_err:.3e}"
    assert Fe_err < 6e-4, f"Fe mismatch vs benchmark: {Fe_err:.3e}"


# ==========================================================================
# 2. CZGM garnet growth + multicomponent diffusion benchmark (spherical)
# ==========================================================================


def _make_czgm_diffusivity(ptt: np.ndarray) -> callable:
    """Build a time- and composition-dependent CZGM diffusivity callable.

    Uses the Arrhenius tracer diffusivities (Chakraborty & Ganguly 1992)
    interpolated along the PTt path, and the Darken-reduced 3x3 matrix with
    Ca as the dependent component:
        D_ij = D_i*delta_ij - (D_i*X_i/sum_k D_k*X_k)*(D_j - D_Ca)

    The PTt-path temperatures stored in ``PTt_path.csv`` are Kelvin and are
    used directly in the Arrhenius expression (no degC->K conversion).
    """
    Tt = interpolate.interp1d(
        ptt[:, 2], ptt[:, 1], fill_value="extrapolate"
    )
    Pt = interpolate.interp1d(
        ptt[:, 2], ptt[:, 0], fill_value="extrapolate"
    )

    specs = {
        "Fe": (6.36e-4, 65824.0, 5.63),
        "Mg": (1.11e-3, 67997.0, 5.27),
        "Mn": (5.15e-4, 60569.0, 6.04),
    }
    Rgas = 1.9872

    def d_tracer(t_myr: float) -> np.ndarray:
        temp = float(Tt(t_myr))
        pressure = float(Pt(t_myr))
        out = [
            D0 * np.exp(-(Ea + (pressure - 1.0) * Ev / 41.84) / (Rgas * temp))
            for D0, Ea, Ev in specs.values()
        ]
        out.append(0.05 * out[2])  # D_Ca = 0.05 * D_Mn
        return np.array(out) / 1e4  # cm2/s -> m2/s

    def d_czgm(t_sec: float, X: dict[str, float]) -> np.ndarray:
        Di = d_tracer(t_sec / MYR_SEC)
        Xv = np.array([X["Fe"], X["Mg"], X["Mn"], X["Ca"]])
        return darken_reduced_diffusivity(Di, Xv, dependent_component=3)

    return d_czgm


@pytest.fixture(scope="module")
def czgm_data() -> dict:
    """Load the CZGM input arrays (skip cleanly if the data are absent)."""
    combined = CZGM_DATA / "combined_array.csv"
    ptt_path = CZGM_DATA / "PTt_path.csv"
    final = CZGM_DATA / "combined_C_final.csv"
    if not all(p.exists() for p in (combined, ptt_path, final)):
        pytest.skip("CZGM data files not available")
    return {
        "data": np.loadtxt(combined, delimiter=","),
        "ptt": np.loadtxt(ptt_path, delimiter=","),
        "final": np.loadtxt(final, delimiter=","),
    }


def test_czgm_spherical_growth_benchmark(czgm_data: dict) -> None:
    """Reproduce the CZGM garnet benchmark in spherical 1D.

    Four components [Fe, Mg, Mn, Ca], Ca closure, spherical radial domain
    0..76.122 um, growing rim set by combined_array.csv, and time-dependent
    Arrhenius diffusivities along the PTt path.  End state is compared with
    the 'combined_C_final.csv' spherical reference.
    """
    data = czgm_data["data"]
    ptt = czgm_data["ptt"]
    final = czgm_data["final"]

    Rr, tr = data[:, 0], data[:, 1]
    Fer, Mgr, Mnr = data[:, 4], data[:, 5], data[:, 6]

    growth_R = Rr[tr > tr.min()]
    growth_t = tr[tr > tr.min()]
    gFe = Fer[tr > tr.min()]
    gMg = Mgr[tr > tr.min()]
    gMn = Mnr[tr > tr.min()]

    r = np.linspace(0.0, Rr[-1] * 1e-6, 161)
    init = {"Fe": Fer[0], "Mg": Mgr[0], "Mn": Mnr[0]}

    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=_make_czgm_diffusivity(ptt),
        geometry="spherical",
        components=["Fe", "Mg", "Mn", "Ca"],
        dependent_component="Ca",
        bcs={"left": ("neumann", 0.0), "right": ("dirichlet", init)},
        initial_composition=init,
        picard_max_iter=50,
        picard_tol=1e-12,
    )

    # --- prograde growth phase ---
    model.time = growth_t[0] * MYR_SEC
    for i in range(len(growth_t) - 1):
        model.apply_growth(
            growth_R[i] * 1e-6,
            {"Fe": gFe[i], "Mg": gMg[i], "Mn": gMn[i]},
        )
        model.run_to_time(
            growth_t[i + 1] * MYR_SEC,
            dt=(growth_t[i + 1] - growth_t[i]) * MYR_SEC,
        )
    # final rim reset at full radius (mirrors the notebook's last growth step)
    model.apply_growth(
        growth_R[-1] * 1e-6,
        {"Fe": gFe[-1], "Mg": gMg[-1], "Mn": gMn[-1]},
    )
    assert model.time == pytest.approx(growth_t[-1] * MYR_SEC)

    # --- retrograde diffusion phase along the PTt path ---
    retro = ptt[:, 2][ptt[:, 2] >= model.time / MYR_SEC - 1e-3]
    for i in range(len(retro) - 1):
        model.run_to_time(
            retro[i + 1] * MYR_SEC,
            dt=(retro[i + 1] - retro[i]) * MYR_SEC,
        )
    assert model.time == pytest.approx(retro[-1] * MYR_SEC)

    X = model.get_composition()
    assert np.max(np.abs(np.sum(X, axis=1) - 1.0)) < 1e-10
    assert np.all(X >= 0.0) and np.all(X <= 1.0)

    # The rim must hold the final growth composition exactly (Dirichlet pin)
    assert X[-1, 0] == pytest.approx(gFe[-1], abs=1e-12)
    assert X[-1, 1] == pytest.approx(gMg[-1], abs=1e-12)
    assert X[-1, 2] == pytest.approx(gMn[-1], abs=1e-12)

    # Compare with the CZGM spherical reference solution
    r_ref = final[:, 0]
    r_um = r * 1e6
    for k, name in enumerate(["Fe", "Mg", "Mn", "Ca"]):
        err = np.max(np.abs(np.interp(r_ref, r_um, X[:, k]) - final[:, k + 1]))
        assert err < 0.012, (
            f"CZGM {name} profile deviates from the reference by {err:.4f}"
        )


def test_czgm_arrhenius_uses_stored_kelvin(czgm_data: dict) -> None:
    """The CZGM data store T in Kelvin, used directly in the Arrhenius term.

    The original MATLAB source subtracts 273 only for a Celsius-facing output
    routine and adds it back inside the Arrhenius term, so the stored T must
    enter the diffusivity expression unchanged; adding 273.15 would
    double-convert the stored Kelvin values.
    """
    ptt = czgm_data["ptt"]

    # Stored values are thermodynamic temperature: 719-883 K (446-610 degC),
    # so a degC-labeled reading of the file must be rejected.
    assert ptt[:, 1].min() > 700.0 and ptt[:, 1].max() < 1000.0

    # Tracer diffusivities at the first path point, from T exactly as stored.
    P0, T0 = ptt[0, 0], ptt[0, 1]
    specs = {
        "Fe": (6.36e-4, 65824.0, 5.63),
        "Mg": (1.11e-3, 67997.0, 5.27),
        "Mn": (5.15e-4, 60569.0, 6.04),
    }
    tracer = np.array(
        [
            D0 * np.exp(-(Ea + (P0 - 1.0) * Ev / 41.84) / (1.9872 * T0))
            for D0, Ea, Ev in specs.values()
        ]
    )
    tracer = np.append(tracer, 0.05 * tracer[-1]) / 1e4  # cm2/s -> m2/s

    X = {"Fe": 0.4, "Mg": 0.05, "Mn": 0.2, "Ca": 0.35}
    d_model = _make_czgm_diffusivity(ptt)(ptt[0, 2] * MYR_SEC, X)
    d_expected = darken_reduced_diffusivity(
        tracer,
        np.array([X["Fe"], X["Mg"], X["Mn"], X["Ca"]]),
        dependent_component=3,
    )
    np.testing.assert_allclose(d_model, d_expected, rtol=1e-12)


# ==========================================================================
# 3. Supporting API: time-dependent diffusivities
# ==========================================================================


def test_time_named_diffusivity_uses_model_time() -> None:
    """A callable whose first parameter is 't' must see the model's time."""
    r = np.linspace(0.0, 1e-5, 51)
    seen: dict[str, float] = {}

    def D_time(t: float, X: dict[str, float]) -> float:
        seen["t"] = float(t)
        return 2.0e-4 * (1.0 + 0.1 * float(t))

    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D_time,
        geometry="planar",
        components=["A", "B"],
        bcs={"left": ("neumann", 0.0), "right": ("dirichlet", {"A": 0.2})},
        initial_composition={"A": 0.8},
    )
    model.time = 123.0
    model.step(1.0)
    assert seen["t"] == pytest.approx(123.0)


def test_radius_named_diffusivity_uses_radius() -> None:
    """A callable whose first parameter is 'r' keeps radius-first semantics."""
    r = np.linspace(0.0, 1e-5, 51)
    seen: dict[str, float] = {}

    def D_radius(rr: float, X: dict[str, float]) -> float:
        seen["r"] = float(rr)
        return 2.0e-4

    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=D_radius,
        geometry="planar",
        components=["A", "B"],
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
        initial_composition={"A": 0.8},
    )
    model.step(1.0)
    assert 0.0 < seen["r"] < r[-1]


def test_darken_reduced_diffusivity_matches_full_matrix_reduction() -> None:
    """Cross-check darken_reduced_diffusivity against reduce_diffusion_matrix."""
    D_tracers = np.array([1.0e-4, 3.0e-4, 5.0e-5, 2.0e-5])
    X = np.array([0.50, 0.20, 0.20, 0.10])

    D_red = darken_reduced_diffusivity(D_tracers, X, dependent_component=3)
    assert D_red.shape == (3, 3)

    # Full Darken matrix with the same formula, reduced via the generic path
    S = float(D_tracers @ X)
    F = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            F[i, j] = D_tracers[i] * (1.0 if i == j else 0.0) - (
                D_tracers[i] * X[i] / S
            ) * (D_tracers[j] - D_tracers[3])
    np.testing.assert_allclose(
        D_red, reduce_diffusion_matrix(F, dependent_idx=3), rtol=1e-14
    )


# ==========================================================================
# 4. Supporting API: growth steps and run_to_time
# ==========================================================================


def test_apply_growth_and_run_to_time() -> None:
    """apply_growth stamps the rim, pins a Dirichlet BC, and run_to_time lands exactly."""
    r = np.linspace(0.0, 1e-5, 101)  # 0..10 um spherical
    model = MulticomponentDiffusion1D(
        grid=r,
        diffusivity=np.diag([1e-4, 1e-4]),
        geometry="spherical",
        components=["A", "B", "C"],
        dependent_component="C",
        bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    )
    model.set_initial_composition({"A": 0.2, "B": 0.3})
    mass_before = model.total_mass("A")

    # Grow a shell 2..10 um with a new rim composition
    model.apply_growth(2e-6, {"A": 0.6, "B": 0.3})
    X = model.get_composition()
    # rim node equals the new rim composition with exact closure
    np.testing.assert_allclose(X[-1], [0.6, 0.3, 0.1], atol=1e-12)
    # material added: A mass increases
    assert model.total_mass("A") > mass_before
    # rim pinned by a Dirichlet BC for subsequent diffusion
    assert model._bcs["right"].bc_type == "dirichlet"

    # run_to_time advances the clock exactly and returns history
    res = model.run_to_time(5.0, dt=1.0)
    assert model.time == pytest.approx(5.0)
    assert res.times[0] == 0.0
    assert res.times[-1] == pytest.approx(5.0)
    assert np.max(np.abs(np.sum(res.compositions, axis=2) - 1.0)) < 1e-10

    # invalid growth radii are rejected
    with pytest.raises(ValueError, match="exceeds"):
        model.apply_growth(2e-5, {"A": 0.6, "B": 0.3})
    with pytest.raises(ValueError, match="greater than"):
        model.apply_growth(0.0, {"A": 0.6, "B": 0.3})
    with pytest.raises(ValueError, match="target_time"):
        model.run_to_time(4.0, dt=1.0)