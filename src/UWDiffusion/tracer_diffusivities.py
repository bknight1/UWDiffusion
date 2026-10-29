"""Published tracer (self) diffusivity parameters for garnet.

A registry of the Arrhenius parameters used by this package, in **one
consistent unit convention** so that no conversion factors are needed at the
point of use:

==================  ==========================================
quantity            units
==================  ==========================================
``log_D0``          log10 of D0, with D0 in m^2/s
``dE``              activation energy, kJ/mol
``dV``              activation volume, cm^3/mol
temperature         K
pressure            GPa
==================  ==========================================

The pressure term of the Arrhenius law is ``(P - P_ref) * dV``.  Since

    1 GPa * cm^3 = 1e9 Pa * 1e-6 m^3 = 1e3 J = 1 kJ

this is directly addable to ``dE`` in kJ/mol.  That identity is why this
convention is used: in the CG92 convention (D0 in cm^2/s, Ea in cal/mol,
dV in cm^3/mol, P in bar) the same term needs a factor of 41.84 to convert
bar*cm^3/mol to cal/mol.  With consistent units the factor is exactly 1.

:mod:`pint` is used for the gas constant and for that conversion, so the
identity is enforced at runtime rather than merely documented.  Getting it
wrong is otherwise silent -- the wrong factor still returns a plausible
number.

Source
------
All values transcribed from Table 1 of

    *Summary of diffusion data in Garnet*, Acta Geochimica (2018) 37(1):19-31,
    DOI 10.1007/s11631-017-0187-x

which tabulates published garnet diffusion studies in these units.  Only rows
with a complete set (log_D0, dE and dV all given) are reproduced here; studies
tabulated without an activation volume are deliberately absent, since filling
one in would be invention rather than transcription.  Per-entry ``source``
fields record the original study.

Composition dependence of the tracer diffusivities themselves is not modelled;
these are constants per species.  The composition dependence of the
*interdiffusion* matrix comes from the Lasaga mean-field reduction
(:mod:`UWDiffusion.diffusion_matrix`), not from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pint

UREG = pint.UnitRegistry()

# Molar gas constant, taken from pint's registry rather than transcribed here.
# CG92 quote R = 1.9872 cal/(mol K), i.e. 8.31443 J/(mol K) with 1 cal = 4.184 J;
# pint's value differs by 3e-6, which is visible at the 1e-3 level in an
# Arrhenius exponent of ~25.  Tests match the legacy convention explicitly.
MOLAR_GAS_CONSTANT = UREG.Quantity("molar_gas_constant")
R_KJ_PER_MOL_K = float(MOLAR_GAS_CONSTANT.to("kilojoule/(mol*kelvin)").magnitude)

# Published parameters arrive in CG92/Loomis units; convert once here.
_CM2S_TO_M2S = 1.0e-4
_KCAL_TO_KJ = 4.184


# Reference pressure for the Arrhenius fits.  CG92 write the pressure term as
# (P - 1) with P in bar, i.e. a 1 bar reference.  That is 1e-4 GPa --
# numerically negligible against geological pressures, but kept explicit so the
# convention is recorded once rather than re-derived at each call site.
P_REF_1BAR_GPA = 1.0e-4


@dataclass(frozen=True)
class TracerDiffusivity:
    """Arrhenius parameters for one tracer (self) diffusivity.

    ``D(T, P) = 10**log_D0 * exp(-(dE + (P - P_ref_GPa) * dV) / (R * T))``

    ``P_ref_GPa`` is a property of the calibration, not of a particular
    evaluation, so it travels with the parameters.  To re-reference a fit,
    use ``dataclasses.replace(entry, P_ref_GPa=...)``.
    """

    log_D0: float
    dE: float
    dV: float
    P_ref_GPa: float = P_REF_1BAR_GPA
    source: str = ""
    composition: str = ""
    T_range: Optional[tuple[float, float]] = None
    P_range: Optional[tuple[float, float]] = None
    method: str = ""
    dE_err: Optional[float] = None
    dV_err: Optional[float] = None

    @property
    def D0(self) -> float:
        """Pre-exponential factor in m^2/s."""
        return 10.0 ** self.log_D0

    def at(self, T_K: float, P_GPa: float = 0.0) -> float:
        """Tracer diffusivity in m^2/s at temperature ``T_K`` and pressure ``P_GPa``.

        The pressure is measured against the calibration's own reference
        (``P_ref_GPa``, 1 bar by convention).  The arithmetic runs through
        :mod:`pint`, so the identity ``1 GPa * cm^3/mol == 1 kJ/mol`` is
        *enforced* rather than assumed.  If the unit conventions ever stop
        being compatible, this raises instead of silently returning a
        plausible-looking number.
        """
        dE = UREG.Quantity(self.dE, "kJ/mol")
        dV = UREG.Quantity(self.dV, "cm**3/mol")
        dP = UREG.Quantity(float(P_GPa) - self.P_ref_GPa, "GPa")
        T = UREG.Quantity(float(T_K), "K")

        numerator = (dE + dP * dV).to("kJ/mol")  # enforces GPa*cm^3/mol == kJ/mol
        exponent = float((-(numerator / (MOLAR_GAS_CONSTANT * T))).to("dimensionless").magnitude)
        return float(self.D0 * np.exp(exponent))


# ---------------------------------------------------------------------------
# Chakraborty & Ganguly (1992), Alm80Prp20-Sps94Alm6, 1573-1753 K,
# 2.9-4.3 GPa, EPMA.  Values are transcribed from the paper's own text, not
# from a secondary tabulation:
#
#     Fe: D0 = 6.4(-4) cm^2/s; Q = 65,824 (+/- 8,721) cal/mol; dV = 5.6 (+/- 2.9) cm^3/mol
#     Mg: D0 = 1.1(-3) cm^2/s; Q = 67,997 (+/- 8,973) cal/mol; dV = 5.3 (+/- 3.0) cm^3/mol
#     Mn: D0 = 5.1(-4) cm^2/s; Q = 60,569 (+/- 8,889) cal/mol; dV = 6.0 (+/- 2.9) cm^3/mol
#
# Two things the paper settles that secondary sources get wrong:
#
# * Mg dV is 5.3, NOT the 5.0 tabulated by the 2018 Acta Geochimica review.
#   That review also misreports the uncertainty (2.9 vs the paper's 3.0).
# * Q is published to five figures while D0 and dV carry two, so D0 and dV
#   are the precision limit here.  Tabulating Q rounded to three figures
#   (275 kJ/mol rather than 275.4) moves D by several per cent because the
#   Arrhenius exponent amplifies it -- which is what happens if you take
#   these values from the 2018 table instead.
#
# CG92 do not publish a Ca tracer diffusivity.  The package convention is
# D_Ca = D_Mn / 20, which is what the CZGM implementation uses, so CA is
# derived from MN below rather than independently measured.
# ---------------------------------------------------------------------------

CG92_FE = TracerDiffusivity(
    log_D0=float(np.log10(6.4e-4 * _CM2S_TO_M2S)),
    dE=65824.0 * _KCAL_TO_KJ / 1000.0,
    dV=5.6,
    source="Chakraborty & Ganguly (1992)",
    composition="Alm80Prp20-Sps94Alm6",
    T_range=(1573.0, 1753.0),
    P_range=(2.9, 4.3),
    method="EPMA",
    dE_err=8721.0 * _KCAL_TO_KJ / 1000.0,
    dV_err=2.9,
)

CG92_MG = TracerDiffusivity(
    log_D0=float(np.log10(1.1e-3 * _CM2S_TO_M2S)),
    dE=67997.0 * _KCAL_TO_KJ / 1000.0,
    dV=5.3,
    source="Chakraborty & Ganguly (1992)",
    composition="Alm80Prp20-Sps94Alm6",
    T_range=(1573.0, 1753.0),
    P_range=(2.9, 4.3),
    method="EPMA",
    dE_err=8973.0 * _KCAL_TO_KJ / 1000.0,
    dV_err=3.0,
)

CG92_MN = TracerDiffusivity(
    log_D0=float(np.log10(5.1e-4 * _CM2S_TO_M2S)),
    dE=60569.0 * _KCAL_TO_KJ / 1000.0,
    dV=6.0,
    source="Chakraborty & Ganguly (1992)",
    composition="Alm80Prp20-Sps94Alm6",
    T_range=(1573.0, 1753.0),
    P_range=(2.9, 4.3),
    method="EPMA",
    dE_err=8889.0 * _KCAL_TO_KJ / 1000.0,
    dV_err=2.9,
)

# D_Ca = D_Mn / 20: not measured, a package convention inherited from the
# CZGM implementation.  Kept as a ratio so it is obvious which part is
# measured and which is assumed.
CG92_CA_D_MN_RATIO = 1.0 / 20.0


def cg92_ca(T_K: float, P_GPa: float = 0.0) -> float:
    """Ca tracer diffusivity under the package convention D_Ca = D_Mn / 20."""
    return CG92_CA_D_MN_RATIO * CG92_MN.at(T_K, P_GPa)


CG92 = {
    "Fe": CG92_FE,
    "Mg": CG92_MG,
    "Mn": CG92_MN,
}


# ---------------------------------------------------------------------------
# Loomis et al. (1985), "Experimental determination of cation diffusivities in
# aluminosilicate garnets; II. Multicomponent simulation and tracer diffusion
# coefficient".  Their Eqs. (4.1)-(4.3) are pressure independent:
#
#     D*_i = D0_i * exp(-Q_i / (R T))          [D0 in cm^2/s, Q in kcal/mol]
#
# There is no pressure term, so dV = 0 exactly -- that is the published form,
# not an assumption.  P_ref is therefore moot and set to 0.
#
# Two caveats worth carrying with the numbers:
#
# * dV = 0 is a *limitation of the fit*, not a physical claim.  The underlying
#   experiments span only 29-43 kbar, over which a real dV of ~5.6 cm^3/mol
#   would move D by a factor ~1.8 -- absorbed into the quoted 13-16%
#   uncertainty on Q.
# * The values are tracer diffusivities *derived* via the Manning-Lasaga
#   model applied to diffusion-couple experiments, not directly measured
#   tracer D.  Loomis tabulate a Darken/Hartley-Crank alternative for Run 32
#   T2 that differs.
#
# NOTE on the 2018 Acta Geochimica review (DOI 10.1007/s11631-017-0187-x):
# its "Sps-Alm" rows under Loomis are these Eqs. (4.1)-(4.3), correctly
# converted.  Its "Alm and Prp" rows are NOT Arrhenius parameters at all --
# they are single-run diffusivities from Run 17 (Pyr-Alm(1), 40 kbar, 1440 C)
# printed in the logD0 column.  Do not transcribe those as pre-exponentials.
# ---------------------------------------------------------------------------

LOOMIS_1985 = {
    # D0 = 3.1e-5 cm^2/s, Q = 61.3 +/- 8.5 kcal/mol
    "Fe": TracerDiffusivity(
        log_D0=float(np.log10(3.1e-5 * _CM2S_TO_M2S)),
        dE=61.3 * _KCAL_TO_KJ,
        dV=0.0,
        P_ref_GPa=0.0,
        source="Loomis et al. (1985), Eq. 4.1",
        composition="Spess-Alm(2), Pyr-Alm(1)",
        T_range=(1573.0, 1753.0),
        P_range=(2.9, 4.33),
        method="EPMA (D* derived via Manning-Lasaga)",
        dE_err=8.5 * _KCAL_TO_KJ,
    ),
    # D0 = 2.2e-5 cm^2/s, Q = 48.1 +/- 7.9 kcal/mol
    "Mn": TracerDiffusivity(
        log_D0=float(np.log10(2.2e-5 * _CM2S_TO_M2S)),
        dE=48.1 * _KCAL_TO_KJ,
        dV=0.0,
        P_ref_GPa=0.0,
        source="Loomis et al. (1985), Eq. 4.2",
        composition="Spess-Alm(2), Pyr-Alm(1)",
        T_range=(1573.0, 1753.0),
        P_range=(2.9, 4.33),
        method="EPMA (D* derived via Manning-Lasaga)",
        dE_err=7.9 * _KCAL_TO_KJ,
    ),
    # D0 = 2.0e-5 cm^2/s, Q = 59.9 +/- 7.9 kcal/mol
    "Mg": TracerDiffusivity(
        log_D0=float(np.log10(2.0e-5 * _CM2S_TO_M2S)),
        dE=59.9 * _KCAL_TO_KJ,
        dV=0.0,
        P_ref_GPa=0.0,
        source="Loomis et al. (1985), Eq. 4.3",
        composition="Spess-Alm(2), Pyr-Alm(1)",
        T_range=(1573.0, 1753.0),
        P_range=(2.9, 4.33),
        method="EPMA (D* derived via Manning-Lasaga)",
        dE_err=7.9 * _KCAL_TO_KJ,
    ),
}

# Loomis publish no Ca Arrhenius fit.  Their only Ca datum is a single-run
# tracer diffusivity, Run 17 (Pyr-Alm(1), 40 kbar, 1440 C = 1713 K):
#
#     D*_Ca = 0.15e-13 cm^2/s = 1.5e-18 m^2/s     ->  log10 = -17.824
#
# That is a point value, NOT a pre-exponential -- the 2018 review prints it in
# a column headed logD0, which is how the confusion arises.  Recorded here as
# data rather than as a parameterisation so it is not lost and not misused.
LOOMIS_1985_RUN17_CA_D_M2S = 1.5e-18
LOOMIS_1985_RUN17 = {
    "T_K": 1713.0,
    "P_GPa": 4.0,
    "D_M2S": {"Fe": 3.0e-18, "Mg": 4.0e-17, "Ca": LOOMIS_1985_RUN17_CA_D_M2S},
    "source": "Loomis et al. (1985), Table 1, Run 17 (Pyr-Alm(1))",
}


# ---------------------------------------------------------------------------
# Only CG92 and Loomis are retained here.  Binary/interdiffusion
# parameterisations (Ganguly et al. 1995, Ganguly & Cheng 1998, Elphick et
# al. 1985, Freer & Edwards 1999) are not inputs to the multicomponent
# matrix -- the Lasaga mean field needs per-species *tracer* diffusivities,
# and an Fe-Mg or Ca-(Mg,Fe) pair coefficient is a different quantity.
# ---------------------------------------------------------------------------

BY_SOURCE = {
    "Chakraborty & Ganguly (1992)": CG92,
    "Loomis et al. (1985)": LOOMIS_1985,
}
