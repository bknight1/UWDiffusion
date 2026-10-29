"""Time-integrated diffusion length and BC-regime screening.

Two properties are checked:

1. **The integral.**  On a constant T/P path the trapezoidal integral is
   analytic: ``x = sqrt(D * t)``, so the function must reproduce
   independently measured values from the CG92 registry to round-off (with
   headroom for transcription).  Input units are geological (Myr, kbar,
   degC) and conversion is checked end-to-end against values measured in
   the registry's own seconds/K/GPa convention.  The path-handling tests
   pin the discretisation contract (trapezoid in D over the path nodes).

2. **The regime classification.**  The eta thresholds map onto the three
   regime strings, and the boundary conditions they imply are the reason
   the function exists -- an off-by-one in the comparison would silently
   recommend the wrong reservoir for a growth model.
"""

import numpy as np
import pytest

from UWDiffusion.tracer_diffusivities import CG92, cg92_ca
from UWDiffusion.utilities import (
    BC_REGIME_EQUILIBRATED_ETA,
    BC_REGIME_ISOLATED_ETA,
    time_integrated_diffusion_length,
)

# Reference diffusion lengths measured directly from the package CG92
# registry at 10 kbar for a 1 Myr constant-T path.  Headroom is ~1%: far
# larger than float noise, far smaller than any change to the Arrhenius
# parameters, the unit conversions, or the quadrature would produce.
_X_700_UM = {"Fe": 40.8, "Mg": 31.1, "Mn": 138.4, "Ca": 31.0}
_X_900_UM = {"Fe": 788.5, "Mg": 658.7, "Mn": 2128.6, "Ca": 476.0}


def _constant_path(t_end_myr, T_C, P_kbar=10.0, n_nodes=2):
    """A constant T/P path from 0 to t_end_myr (geological units)."""
    t = np.linspace(0.0, t_end_myr, n_nodes)
    T = np.full(n_nodes, T_C)
    P = np.full(n_nodes, P_kbar)
    return t, T, P


def test_constant_path_matches_sqrt_Dt() -> None:
    """x must equal sqrt(D * t) on a constant path, to measured headroom.

    The trapezoid of a constant integrand is exact, so this checks the
    diffusivity lookup and the Myr/kbar/degC -> s/GPa/K conversions
    end-to-end against values measured independently from the CG92
    registry (10 kbar, 1 Myr).
    """
    t, T, P = _constant_path(1.0, T_C=700.0)
    result = time_integrated_diffusion_length(t, T, P, radius_m=1.0e-3)

    for component, x_um in _X_700_UM.items():
        got_um = result[component]["diffusion_length_m"] * 1e6
        assert got_um == pytest.approx(x_um, rel=0.01), (component, got_um)


def test_constant_path_900C_matches_sqrt_Dt() -> None:
    """Same check at 900 degC / 10 kbar, where lengths are ~20x larger."""
    t, T, P = _constant_path(1.0, T_C=900.0)
    result = time_integrated_diffusion_length(t, T, P, radius_m=1.0)

    for component, x_um in _X_900_UM.items():
        got_um = result[component]["diffusion_length_m"] * 1e6
        assert got_um == pytest.approx(x_um, rel=0.01), (component, got_um)


def test_regimes_at_700C_1mm() -> None:
    """At 700 degC / 10 kbar / 1 Myr every component is isolated for R = 1 mm.

    Measured etas: Fe 0.041, Mg 0.031, Ca 0.031, Mn 0.138 -- all below
    BC_REGIME_ISOLATED_ETA (0.3).  Mn is the nearest to the threshold, so
    the assertion also guards against a threshold regression moving above
    ~0.14.
    """
    t, T, P = _constant_path(1.0, T_C=700.0)
    result = time_integrated_diffusion_length(t, T, P, radius_m=1.0e-3)

    for component, values in result.items():
        assert values["regime"] == "isolated", (component, values)
        assert values["eta"] < BC_REGIME_ISOLATED_ETA


def test_regimes_at_900C_1mm() -> None:
    """900 degC / 10 kbar / 1 Myr straddles regimes: Mn equilibrated, Fe/Mg/Ca not.

    Measured etas: Mn 2.13 (> 1, equilibrated), Fe 0.789, Mg 0.659,
    Ca 0.476 -- all in the transitional band (0.3, 1.0).  This is the
    screening result the two-loop BC plan hinges on, so the regime strings
    are asserted component by component.
    """
    t, T, P = _constant_path(1.0, T_C=900.0)
    result = time_integrated_diffusion_length(t, T, P, radius_m=1.0e-3)

    assert result["Mn"]["regime"] == "equilibrated"
    for component in ("Fe", "Mg", "Ca"):
        assert result[component]["regime"] == "transitional", component
        assert BC_REGIME_ISOLATED_ETA < result[component]["eta"] < BC_REGIME_EQUILIBRATED_ETA


def test_threshold_edges() -> None:
    """Isolated is eta < 0.3, equilibrated is eta > 1.0, else transitional."""
    t, T, P = _constant_path(1.0, T_C=700.0)
    x700_mn = time_integrated_diffusion_length(t, T, P, radius_m=1.0)["Mn"][
        "diffusion_length_m"
    ]
    # Choose radii that put Mn exactly on each side of the thresholds:
    # eta = x / R  ->  R = x / eta.
    r_at_eta = lambda eta: x700_mn / eta  # noqa: E731

    result = time_integrated_diffusion_length(t, T, P, radius_m=r_at_eta(0.29))
    assert result["Mn"]["regime"] == "isolated"
    result = time_integrated_diffusion_length(t, T, P, radius_m=r_at_eta(0.31))
    assert result["Mn"]["regime"] == "transitional"
    result = time_integrated_diffusion_length(t, T, P, radius_m=r_at_eta(0.99))
    assert result["Mn"]["regime"] == "transitional"
    result = time_integrated_diffusion_length(t, T, P, radius_m=r_at_eta(1.01))
    assert result["Mn"]["regime"] == "equilibrated"


def test_path_splitting_is_invariant() -> None:
    """Splitting a constant-T path into many segments must not change x.

    The trapezoid of a constant integrand is exact regardless of node
    placement, so any difference is a discretisation bug.  Tolerance is
    float noise only.
    """
    t2, T2, P2 = _constant_path(1.0, T_C=700.0, n_nodes=2)
    t9, T9, P9 = _constant_path(1.0, T_C=700.0, n_nodes=9)

    r2 = time_integrated_diffusion_length(t2, T2, P2, radius_m=1.0)
    r9 = time_integrated_diffusion_length(t9, T9, P9, radius_m=1.0)
    for component in r2:
        assert r2[component]["diffusion_length_m"] == pytest.approx(
            r9[component]["diffusion_length_m"], rel=1e-12
        ), component


def test_two_temperature_path_matches_explicit_trapezoid() -> None:
    """A changing-T path must match an explicit trapezoid over the nodes.

    Pins the discretisation contract: x^2 = sum_k 0.5 (D_k + D_{k+1}) dt_k,
    with D evaluated from the CG92 registry at each node (converted here
    back to the registry's K/GPa convention by hand).
    """
    t_myr = np.array([0.0, 2.0, 5.0])          # nodes at 0, 2, 5 Myr
    T_C = np.array([700.0, 700.0, 900.0])      # hold, then up to 900
    P_kbar = np.full(3, 10.0)

    result = time_integrated_diffusion_length(t_myr, T_C, P_kbar, radius_m=1.0)
    t_s = t_myr * 365.25 * 24 * 3600 * 1e6     # Myr -> s by definition
    T_K = T_C + 273.15
    P_GPa = P_kbar * 0.1                       # 1 kbar = 0.1 GPa exactly
    for component, values in result.items():
        fn = cg92_ca if component == "Ca" else CG92[component].at
        D = np.array([fn(Tk, Pk) for Tk, Pk in zip(T_K, P_GPa)])
        x2 = np.sum(0.5 * (D[:-1] + D[1:]) * np.diff(t_s))
        assert values["diffusion_length_m"] == pytest.approx(
            np.sqrt(x2), rel=1e-12
        ), component


def test_ramp_convergence() -> None:
    """Refining nodes along a linear T ramp must converge (trapezoid O(h^2)).

    Measured relative changes: 11 -> 101 nodes ~1.1e-2 (coarse trapezoid
    is ~1% off), 101 -> 1001 nodes ~1.1e-4 -- a ~100x reduction per 10x
    refinement, as O(h^2) requires.  Headroom is 2x the measured values.
    """
    def ramp(n_nodes):
        t = np.linspace(0.0, 10.0, n_nodes)      # 0-10 Myr
        T = np.linspace(700.0, 900.0, n_nodes)   # degC
        P = np.full(n_nodes, 10.0)               # kbar
        return time_integrated_diffusion_length(t, T, P, radius_m=1.0)

    coarse, fine, finer = ramp(11), ramp(101), ramp(1001)
    for component in coarse:
        xc = coarse[component]["diffusion_length_m"]
        xf = fine[component]["diffusion_length_m"]
        xff = finer[component]["diffusion_length_m"]
        assert xc == pytest.approx(xf, rel=2e-2), component
        assert xf == pytest.approx(xff, rel=2e-3), component


def test_input_validation() -> None:
    """Invalid inputs raise ValueError with a clear message."""
    t, T, P = _constant_path(1.0, T_C=700.0)

    with pytest.raises(ValueError, match="same length"):
        time_integrated_diffusion_length(t[:-1], T, P, radius_m=1.0)
    with pytest.raises(ValueError, match="at least two nodes"):
        time_integrated_diffusion_length(t[:1], T[:1], P[:1], radius_m=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        time_integrated_diffusion_length(t[::-1], T, P, radius_m=1.0)
    with pytest.raises(ValueError, match="positive"):
        time_integrated_diffusion_length(t, T, P, radius_m=0.0)
    with pytest.raises(ValueError, match="unknown component"):
        time_integrated_diffusion_length(t, T, P, radius_m=1.0, components=("Zn",))
