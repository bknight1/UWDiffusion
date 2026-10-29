# %%
"""BC-regime screening for two garnet P-T loops.

Evaluates the time-integrated diffusion length eta = x/R for a prograde ->
peak-hold -> retrograde loop at two peak temperatures and asks which
boundary-condition source each element implies:

* Loop 1, peak 700 degC -- expect interior isolation (fractionated chain:
  effective-bulk BCs during growth, final-effective-bulk retrograde BCs).
* Loop 2, peak 900 degC -- expect interior communication for at least some
  elements (original-bulk equilibrium BC, prograde zoning erased).

Assumptions, all deliberate simplifications:

* Path shape: 400 -> peak over 10 Myr (5 -> 6 kbar), isothermal hold at
  peak (1 or 10 Myr), cool back to 400 degC over 10 Myr (6 -> 4 kbar).
  The Arrhenius integral is dominated by the hold, so the limb durations
  barely matter; they are there so the path is a loop.
* Radius R = 1 mm (typical garnet); eta scales linearly with 1/R.
* CG92 tracer diffusivities with D_Ca = D_Mn/20 (package convention).
  CG92 is calibrated at 1573-1753 K, 2.9-4.3 GPa, so 700-900 degC / 5-6
  kbar is a strong extrapolation -- read the regimes as indicators, not
  precise thresholds.

Run from the repository root in an environment with the UW3/PETSc/MPI stack
available (see AGENTS.md):

    python examples/diffusion_length_bc_regime.py
"""

# %%
import numpy as np

from UWDiffusion.utilities import time_integrated_diffusion_length

# %%
# Loop parameters.  Both loops share the path shape; only the peak differs.

RADIUS_M = 1.0e-3  # 1 mm garnet
LIMB_MYR = 10.0    # prograde and retrograde limb durations
HOLDS_MYR = (1.0, 10.0)
PEAKS_C = (700.0, 900.0)
COMPONENTS = ("Fe", "Mg", "Mn", "Ca")

BC_PRESCRIPTION = {
    "isolated": "effective-bulk BC chain (fractionating growth, final-bulk retrograde)",
    "transitional": "neither end-member -- run both BC sources as bounds",
    "equilibrated": "original-bulk equilibrium BC; interior zoning erased",
}


def loop_path(peak_T_C, hold_Myr, n_limb=20):
    """Prograde 400 -> peak (5 -> 6 kbar), hold, retrograde -> 400 (6 -> 4 kbar).

    Returns t_Myr, T_C, P_kbar node arrays (strictly increasing in time).
    """
    t_up = np.linspace(0.0, LIMB_MYR, n_limb)
    T_up = np.linspace(400.0, peak_T_C, n_limb)
    P_up = np.linspace(5.0, 6.0, n_limb)

    t_hold = np.linspace(LIMB_MYR, LIMB_MYR + hold_Myr, 5)[1:]
    T_hold = np.full(t_hold.size, peak_T_C)
    P_hold = np.full(t_hold.size, 6.0)

    t0 = LIMB_MYR + hold_Myr
    t_dn = np.linspace(t0, t0 + LIMB_MYR, n_limb)[1:]
    T_dn = np.linspace(peak_T_C, 400.0, n_limb)[1:]
    P_dn = np.linspace(6.0, 4.0, n_limb)[1:]

    return (
        np.concatenate([t_up, t_hold, t_dn]),
        np.concatenate([T_up, T_hold, T_dn]),
        np.concatenate([P_up, P_hold, P_dn]),
    )


# %%
# Screen every loop configuration.

for loop_no, peak_T_C in enumerate(PEAKS_C, start=1):
    for hold_Myr in HOLDS_MYR:
        t_Myr, T_C, P_kbar = loop_path(peak_T_C, hold_Myr)
        result = time_integrated_diffusion_length(
            t_Myr, T_C, P_kbar, radius_m=RADIUS_M, components=COMPONENTS
        )
        print(
            f"Loop {loop_no} (peak {peak_T_C:.0f} degC, hold {hold_Myr:g} Myr, "
            f"R = {RADIUS_M * 1e3:g} mm):"
        )
        print(f"  {'element':8s} {'x (m)':>10s} {'eta':>7s}  {'regime':<13s} BC source")
        for component in COMPONENTS:
            v = result[component]
            print(
                f"  {component:8s} {v['diffusion_length_m']:10.3e} "
                f"{v['eta']:7.3f}  {v['regime']:<13s} "
                f"{BC_PRESCRIPTION[v['regime']]}"
            )
        print()

# %%
# Reading the table:
# * eta << 1 -- the interior cannot exchange with the rim on this path, so
#   components sequestered during growth stay sequestered: fractionate the
#   bulk (Loop 1 plan).
# * eta >~ 1 -- interior, rim and matrix stay connected for that element:
#   original-bulk equilibrium BCs are defensible and prograde zoning in
#   that element cannot survive (Loop 2 plan).
# * elements straddling the thresholds (typically Mn at 700 degC over long
#   holds, Ca at 900 degC over short holds) are the ones worth a BC-source
#   sensitivity run.
