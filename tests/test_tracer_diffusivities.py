"""Tracer diffusivity registry: unit convention and transcription.

Two distinct properties are checked, because they fail differently:

1. **The unit convention.**  The SI Arrhenius helper must reproduce the
   legacy CG92 formula (D0 in cm^2/s, Ea in cal/mol, dV in cm^3/mol, P in bar)
   *exactly* when given the same physical parameters expressed in the new
   units.  This is the one that can silently break -- a wrong conversion
   factor gives plausible-looking numbers.

2. **The transcription.**  The registry's values come from Table 1 of
   Acta Geochimica (2018) 37(1):19-31, which rounds the published CG92
   parameters.  They should land within a few per cent of the unrounded
   values used by the CZGM implementation, and no further.
"""

import numpy as np
import pytest

from UWDiffusion.tracer_diffusivities import (
    CG92,
    CG92_CA_D_MN_RATIO,
    R_KJ_PER_MOL_K,
    TracerDiffusivity,
    cg92_ca,
)


# Legacy CG92 convention, exactly as the CZGM implementation states it.
_LEGACY_D0_CM2S = {"Mn": 5.15e-4, "Mg": 1.11e-3, "Fe": 6.36e-4}
_LEGACY_EA_CAL = {"Mn": 60569.0, "Mg": 67997.0, "Fe": 65824.0}
_LEGACY_EV_CM3 = {"Mn": 6.04, "Mg": 5.27, "Fe": 5.63}
_R_CAL_PER_MOL_K = 1.9872
_CAL_PER_BAR_CM3 = 41.84


def _legacy_um2s(species: str, T_K: float, P_bar: float) -> float:
    """CG92 tracer diffusivity in um^2/s, legacy units."""
    d_cm2s = _LEGACY_D0_CM2S[species] * np.exp(
        -(_LEGACY_EA_CAL[species] + (P_bar - 1.0) * _LEGACY_EV_CM3[species] / _CAL_PER_BAR_CM3)
        / (_R_CAL_PER_MOL_K * T_K)
    )
    return float(d_cm2s * 1e8)  # cm^2/s -> um^2/s


# Same physical parameters expressed in the module's SI convention.
#   cm^2/s -> m^2/s : x 1e-4          cal/mol -> kJ/mol : x 4.184 / 1000
#   cm^3/mol is already the module unit.
_LEGACY_AS_SI = {
    "Mn": TracerDiffusivity(log_D0=np.log10(5.15e-4 * 1e-4), dE=60569.0 * 4.184 / 1000.0, dV=6.04),
    "Mg": TracerDiffusivity(log_D0=np.log10(1.11e-3 * 1e-4), dE=67997.0 * 4.184 / 1000.0, dV=5.27),
    "Fe": TracerDiffusivity(log_D0=np.log10(6.36e-4 * 1e-4), dE=65824.0 * 4.184 / 1000.0, dV=5.63),
}


def test_si_units_reproduce_legacy_cg92_exactly() -> None:
    """The new convention must agree with the old one to round-off.

    This is the regression that matters: if the GPa*cm^3/mol = kJ/mol
    identity were wrong, or the reference pressure mishandled, every
    diffusivity would shift by a factor while still looking plausible.
    """
    for species in ("Fe", "Mg", "Mn"):
        for T_K in (900.0, 1100.0, 1573.0, 1753.0):
            for P_bar in (1.0, 29000.0, 43000.0):  # 1 bar, 2.9 and 4.3 GPa
                legacy = _legacy_um2s(species, T_K, P_bar)
                # 1 bar is the CG92 reference pressure; in GPa that is 1e-4.
                # 1 GPa = 1e4 bar.
                si = _LEGACY_AS_SI[species].at(T_K, P_GPa=P_bar * 1e-4) * 1e12  # m^2/s -> um^2/s
                assert si == pytest.approx(legacy, rel=1e-3), (species, T_K, P_bar)


def test_registry_agrees_with_the_czgm_implementation() -> None:
    """Paper values must reproduce the CZGM code's diffusivities closely.

    The tolerance is 2%: the paper publishes Q to five figures (so it matches
    the code exactly) but D0 and dV to two, where the code carries three.
    That residual is transcription precision, not a physics difference.
    """
    for species in ("Fe", "Mg", "Mn"):
        for T_K in (1200.0, 1600.0):
            for P_GPa in (0.0, 3.0):
                paper = CG92[species].at(T_K, P_GPa) * 1e12
                legacy = _legacy_um2s(species, T_K, P_GPa * 1e4)
                assert paper == pytest.approx(legacy, rel=0.02), (species, T_K, P_GPa)


def test_acta_2018_review_misreports_mg_activation_volume() -> None:
    """The 2018 review says Mg dV = 5.0; the CG92 paper says 5.3 +/- 3.0.

    Recorded so the registry follows the primary source and the discrepancy
    is visible rather than silently resolved.  The paper also gives the
    uncertainty as 3.0, not the review's 2.9.  Every other CG92 value in that
    review agrees with the paper.
    """
    assert CG92["Mg"].dV == 5.3
    assert CG92["Mg"].dV_err == 3.0
    # Fe and Mn match the review exactly.
    assert CG92["Fe"].dV == 5.6 and CG92["Mn"].dV == 6.0


def test_ca_is_mn_over_twenty_not_measured() -> None:
    """CG92 publish no Ca tracer diffusivity; D_Ca = D_Mn / 20 is a convention."""
    assert CG92_CA_D_MN_RATIO == pytest.approx(0.05)
    T_K, P_GPa = 1500.0, 2.0
    assert cg92_ca(T_K, P_GPa) == pytest.approx(
        CG92["Mn"].at(T_K, P_GPa) / 20.0
    )
    # And Ca must not masquerade as a measured entry in the registry.
    assert "Ca" not in CG92


def test_registry_entries_are_complete_and_cited() -> None:
    """Every registry row must have log_D0, dE, dV and a source.

    Incomplete rows are deliberately excluded from the module -- filling in a
    missing activation volume would be invention, not transcription.
    ``dV == 0`` is allowed and meaningful: Loomis's fits are pressure
    independent by construction.
    """
    from UWDiffusion import tracer_diffusivities as td

    for group_name, group in td.BY_SOURCE.items():
        for species, entry in group.items():
            assert np.isfinite(entry.log_D0), (group_name, species)
            assert np.isfinite(entry.dE) and entry.dE > 0, (group_name, species)
            assert np.isfinite(entry.dV) and entry.dV >= 0, (group_name, species)
            assert entry.source, (group_name, species)


def test_reference_pressure_travels_with_the_calibration() -> None:
    """P_ref is a property of the fit, not of an evaluation call.

    Every call site used to have to remember to pass it; now the entry
    carries it and ``at()`` needs only T and P.
    """
    import inspect

    from UWDiffusion import tracer_diffusivities as td

    params = inspect.signature(td.TracerDiffusivity.at).parameters
    assert set(params) == {"self", "T_K", "P_GPa"}, params
    assert td.CG92["Fe"].P_ref_GPa == pytest.approx(1e-4)  # 1 bar
    # Re-referencing is a dataclasses.replace, not a new formula.
    from dataclasses import replace

    alt = replace(td.CG92["Fe"], P_ref_GPa=0.0)
    assert alt.at(1500.0, 2.0) != td.CG92["Fe"].at(1500.0, 2.0)


def test_gas_constant_is_consistent_with_units() -> None:
    """R in kJ/mol/K; multiplied by T in K it must match dE in kJ/mol."""
    assert R_KJ_PER_MOL_K == pytest.approx(8.314462618e-3)
    # Sanity: 1 GPa*cm^3/mol == 1 kJ/mol, the identity the convention rests on.
    assert 1.0e9 * 1.0e-6 == pytest.approx(1.0e3)


def test_monotonic_in_temperature_and_pressure() -> None:
    """D must increase with T and, for positive dV, decrease with P."""
    for species in ("Fe", "Mg", "Mn"):
        e = CG92[species]
        assert e.at(1500.0) > e.at(1000.0)
        assert e.at(1500.0, P_GPa=3.0) < e.at(1500.0, P_GPa=0.0)


def test_pint_provides_the_gas_constant_and_enforces_the_pressure_identity() -> None:
    """R comes from pint's registry; the pressure unit identity is checked at runtime.

    Neither guard catches a wrongly transcribed *value* -- pint cannot know
    that 5.6 should have been 5.63.  What they do catch is convention drift:
    if the dV unit ever stops being compatible with dE, or R is hand-edited to
    a value inconsistent with the unit registry, these fail loudly instead of
    shifting every diffusivity by a few per cent and still looking plausible.
    """
    from UWDiffusion import tracer_diffusivities as td

    # Gas constant is pint's, not a transcribed literal.
    assert td.MOLAR_GAS_CONSTANT.units == td.UREG.Unit("molar_gas_constant")
    assert td.R_KJ_PER_MOL_K == pytest.approx(8.314462618e-3)

    # The identity the whole convention rests on.
    one = td.UREG.Quantity(1.0, "GPa") * td.UREG.Quantity(1.0, "cm**3/mol")
    assert one.to("kJ/mol").magnitude == pytest.approx(1.0)

    # And it is genuinely used: 1 GPa shifts the exponent by exactly dV/(R T),
    # which is only true if GPa*cm^3/mol really is kJ/mol here.
    e = CG92["Fe"]
    T = 1500.0
    expected_ratio = float(np.exp(-e.dV / (td.R_KJ_PER_MOL_K * T)))
    assert e.at(T, P_GPa=1.0) / e.at(T, P_GPa=0.0) == pytest.approx(
        expected_ratio, rel=1e-12
    )


def test_loomis_is_pressure_independent() -> None:
    """Loomis Eqs. (4.1)-(4.3) have no pressure term, so dV = 0 exactly.

    That is the published form, not an assumption.  It follows that D must not
    move with pressure and that P_ref is moot.
    """
    from UWDiffusion import tracer_diffusivities as td

    for species, entry in td.LOOMIS_1985.items():
        assert entry.dV == 0.0, species
        assert entry.P_ref_GPa == 0.0, species
        for P in (0.0, 1.0, 5.0, 20.0):
            assert entry.at(1500.0, P) == pytest.approx(
                entry.at(1500.0, 0.0), rel=1e-15
            ), (species, P)


def test_loomis_matches_acta_2018_sps_alm_rows() -> None:
    """The 2018 review's "Sps-Alm" rows are Loomis Eqs. (4.1)-(4.3), converted.

    Independent check of the transcription: the review quotes logD0 in m^2/s
    and dE in kJ/mol, computed here from Loomis's cm^2/s and kcal/mol.
    """
    from UWDiffusion import tracer_diffusivities as td

    # (review logD0 [m^2/s], review dE [kJ/mol])
    acta = {"Fe": (-8.51, 257.0), "Mg": (-8.70, 251.0), "Mn": (-8.66, 201.0)}
    for species, (log_D0_acta, dE_acta) in acta.items():
        entry = td.LOOMIS_1985[species]
        assert entry.log_D0 == pytest.approx(log_D0_acta, abs=5e-3), species
        assert entry.dE == pytest.approx(dE_acta, rel=2e-2), species


def test_acta_2018_alm_prp_rows_are_run17_point_values() -> None:
    """The review's "Alm and Prp" Loomis rows are NOT Arrhenius parameters.

    They are single-run tracer diffusivities from Run 17 (Pyr-Alm(1), 40 kbar,
    1440 C) printed in a column headed logD0.  Identified here so nobody
    transcribes them as pre-exponentials -- that would be wrong by tens of
    orders of magnitude once fed through an Arrhenius law.
    """
    from UWDiffusion import tracer_diffusivities as td

    run17 = td.LOOMIS_1985_RUN17
    assert run17["T_K"] == pytest.approx(1440.0 + 273.0)   # 1713 K
    assert run17["P_GPa"] == pytest.approx(40.0 * 0.1)     # 40 kbar = 4 GPa

    # log10(D*) of Run 17 reproduces the review's logD0 column exactly.
    for species, review_logD0 in (("Fe", -17.52), ("Mg", -16.40), ("Ca", -17.82)):
        assert float(np.log10(run17["D_M2S"][species])) == pytest.approx(
            review_logD0, abs=5e-3
        ), species

    # And they are point values, not usable as pre-exponentials: Loomis publish
    # no Ca Arrhenius fit at all.
    assert "Ca" not in td.LOOMIS_1985
