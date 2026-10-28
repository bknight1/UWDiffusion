# %%
"""Is MATLAB's planar spatial operator the source of the 1.5x gap?

Two discretisations of ``div(D(X) grad X)`` are compared under *identical*
driving and identical explicit time integration, so the only variable is the
spatial operator:

``matlab``  -- node-``D`` with a forward difference for ``grad D``::

    A(i,:) = C(i,:) + QQ*( (D(i+1)-D(i)) @ (C(i+1)-C(i))
                         +  D(i)      @ (C(i+1)-2C(i)+C(i-1)) )
    QQ = dt/dx^2 ;  dt = 0.3*dx^2/De  ;  De = max|eig(D)| at centre and rim

``fv``      -- conservative finite volume with face-averaged ``D``::

    A(i,:) = C(i,:) + QQ*( D(i+1/2) @ (C(i+1)-C(i))
                         - D(i-1/2) @ (C(i)-C(i-1)) )

For constant ``D`` the two are algebraically identical; they differ only where
``D`` depends on composition.  Both are advanced with forward Euler, MATLAB's
``dt = 0.3 dx^2 / De`` rule, MATLAB's centre condition (``C(1)=C(2)``, node 1
frozen) and MATLAB's growing-domain prograde driving.

Run:  python development/multicomponent_solver/test_matlab_operator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

SEC_MY = 1.0e6 * 365.25 * 24.0 * 3600.0
DATA = Path(__file__).resolve().parents[1] / "Tutorials" / "Diffusion" / "CZGM_data"


# %% MATLAB com_D(p, T, ...) -- pressure FIRST, temperature in degrees C


def tracer_um2s(temp_C, pressure_bar):
    D0 = {"Mn": 5.15e-4, "Mg": 1.11e-3, "Fe": 6.36e-4}   # cm^2/s
    Ea = {"Mn": 60569.0, "Mg": 67997.0, "Fe": 65824.0}
    Ev = {"Mn": 6.04, "Mg": 5.27, "Fe": 5.63}
    R = 1.9872

    def _D(e):
        return D0[e] * 1e8 * np.exp(
            -(Ea[e] + (pressure_bar - 1.0) * Ev[e] / 41.84) / (R * (temp_C + 273.0))
        )
    DMn, DMg, DFe = _D("Mn"), _D("Mg"), _D("Fe")
    return DMn, DMg, DFe, 0.05 * DMn


def com_D(pressure_bar, temp_C, cMn, cMg, cFe, cCa):
    """Reduced 3x3 matrix, MATLAB component order (Mn, Mg, Fe), Ca dependent."""
    DD1, DD2, DD3, DD4 = tracer_um2s(temp_C, pressure_bar)
    Q = DD1 * cMn + DD2 * cMg + DD3 * cFe + DD4 * cCa
    DDv = (DD1, DD2, DD3)
    cv = (cMn, cMg, cFe)
    DD = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            DD[i, j] = DDv[i] * (1.0 if i == j else 0.0) - (DDv[i] * cv[i] / Q) * (
                DDv[j] - DD4
            )
    return DD, float(np.max(np.abs(np.linalg.eigvals(DD))))


# %% one explicit step, either operator


def step_operator(C, CC, dx, QQ, Di, operator, centre="matlab"):
    """Advance one explicit step.  ``Di`` is D at every node, shape (nc,3,3)."""
    nc = C.shape[0]
    A = np.empty_like(C)
    if centre == "matlab":
        A[0, :] = C[0, :]      # MATLAB: A(1,:) = C(1,:)  (node 1 frozen)
    else:
        # true zero-flux at the left face: node 0 evolves from its right
        # neighbour only, with the half control volume of a boundary vertex
        D_half = 0.5 * (Di[0] + Di[1])
        A[0, :] = C[0, :] + 2.0 * QQ * (D_half @ (C[1, :] - C[0, :]))
    A[-1, :] = C[-1, :]        # rim handled by the caller
    if operator == "matlab":
        for i in range(1, nc - 1):
            D = Di[i]
            dD = Di[i + 1] - D
            A[i, :] = C[i, :] + QQ * (
                dD @ (C[i + 1, :] - C[i, :])
                + D @ (C[i + 1, :] - 2.0 * C[i, :] + C[i - 1, :])
            )
    else:  # conservative FV, face-averaged D
        for i in range(1, nc - 1):
            Dp = 0.5 * (Di[i] + Di[i + 1])
            Dm = 0.5 * (Di[i - 1] + Di[i])
            A[i, :] = C[i, :] + QQ * (
                Dp @ (C[i + 1, :] - C[i, :]) - Dm @ (C[i, :] - C[i - 1, :])
            )
    return A


def interval(C, CC, r, t0, t1, T_of_t, P_of_t, rim_fun, operator, cfl=0.3, centre="matlab"):
    """Advance one interval with MATLAB's explicit rule."""
    nc = len(r)
    dx = float(r[1] - r[0])
    tim = float(t0)
    n = 0
    while t1 - tim > 1.0e-6 * max(1.0, abs(t1)) and n < 200000:
        Pa = float(P_of_t(tim / SEC_MY))
        Ta = float(T_of_t(tim / SEC_MY)) - 273.0
        _, De1 = com_D(Pa, Ta, C[0, 0], C[0, 1], C[0, 2], CC[0])
        _, De3 = com_D(Pa, Ta, C[-1, 0], C[-1, 1], C[-1, 2], CC[-1])
        De = max(De1, De3)
        if not np.isfinite(De) or De <= 0.0:
            raise RuntimeError(f"bad De={De}")
        dt = min(cfl * dx * dx / De, t1 - tim)
        QQ = dt / (dx * dx)

        C = C.copy()
        CC = CC.copy()
        if centre == "matlab":
            # MATLAB: zero-gradient mirror, node 0 frozen at node 1's value
            C[0, :] = C[1, :]
        # rim
        C[-1, :] = rim_fun((tim + dt) / SEC_MY)

        Di = np.empty((nc, 3, 3))
        for i in range(nc):
            Di[i], _ = com_D(Pa, Ta, C[i, 0], C[i, 1], C[i, 2], CC[i])

        A = step_operator(C, CC, dx, QQ, Di, operator, centre)
        if centre == "matlab":
            A[0, :] = C[0, :]
        A[-1, :] = C[-1, :]
        C = A
        CC[:] = 1.0 - C[:, 0] - C[:, 1] - C[:, 2]
        tim += dt
        n += 1
    return C, CC, n


# %% full history: prograde (growing domain) + retrograde


def run(operator, i0=10, drive="grow", cfl=0.3, centre="matlab"):
    cz = pd.read_csv(DATA / "combined_array.csv", header=None).to_numpy(float)
    ptt = pd.read_csv(DATA / "PTt_path.csv", header=None).to_numpy(float)
    ref = pd.read_csv(DATA / "combined_C_final_cart.csv", header=None).to_numpy(float)

    Rr, tr, Pr, Tr, Fer, Mgr, Mnr, Car = [cz[:, i] for i in range(8)]
    T_of_t = interpolate.interp1d(ptt[:, 2], ptt[:, 1], fill_value="extrapolate")
    P_of_t = interpolate.interp1d(ptt[:, 2], ptt[:, 0], fill_value="extrapolate")
    # MATLAB BCond_creation=3 interpolates the path end-members at the current
    # time; the shell compositions trace the same curve, so interpolate those.
    Fe_of_t = interpolate.interp1d(tr, Fer, fill_value="extrapolate")
    Mg_of_t = interpolate.interp1d(tr, Mgr, fill_value="extrapolate")
    Mn_of_t = interpolate.interp1d(tr, Mnr, fill_value="extrapolate")

    def rim_fun(t_myr):
        return np.array(
            [float(Mn_of_t(t_myr)), float(Mg_of_t(t_myr)), float(Fe_of_t(t_myr))]
        )

    nr = len(Rr)
    # state arrays in MATLAB's (Mn, Mg, Fe) order
    cMn, cMg, cFe, cCa = Mnr.copy(), Mgr.copy(), Fer.copy(), Car.copy()

    # ---- prograde -------------------------------------------------------
    # 'grow'  : MATLAB ind=[1:i] -- the domain expands, the new node carries
    #           its zonation value and the interior is untouched otherwise.
    # 'stamp' : fixed full domain; everything beyond the current growth front
    #           is overwritten with the current rim composition (what
    #           fd_1d.apply_growth does).
    growth = tr > tr.min()
    Rg, tg = Rr[growth], tr[growth]
    rim_zonation = np.column_stack([Mnr[growth], Mgr[growth], Fer[growth]])

    nsteps = 0
    if drive == "grow":
        for i in range(i0, nr + 1):          # i = number of nodes in the domain
            r = Rr[:i]
            C = np.column_stack([cMn[:i], cMg[:i], cFe[:i]]).astype(float)
            CC = cCa[:i].astype(float).copy()
            if i == i0:
                t0, t1 = tr[0] * SEC_MY, tr[i0 - 1] * SEC_MY
            else:
                t0, t1 = tr[i - 2] * SEC_MY, tr[i - 1] * SEC_MY
            if t1 <= t0:
                continue
            C, CC, ns = interval(C, CC, r, t0, t1, T_of_t, P_of_t, rim_fun, operator, cfl, centre)
            cMn[:i], cMg[:i], cFe[:i], cCa[:i] = C[:, 0], C[:, 1], C[:, 2], CC
            nsteps += ns
    else:  # 'stamp'
        r = Rr
        for k in range(len(Rg)):
            # stamp everything beyond the current front with the rim value
            mask = r > Rg[k]
            cMn[mask] = rim_zonation[k, 0]
            cMg[mask] = rim_zonation[k, 1]
            cFe[mask] = rim_zonation[k, 2]
            cCa[mask] = 1.0 - cMn[mask] - cMg[mask] - cFe[mask]
            t0 = tg[k - 1] * SEC_MY if k > 0 else tr[0] * SEC_MY
            t1 = tg[k] * SEC_MY
            if t1 <= t0:
                continue
            C = np.column_stack([cMn, cMg, cFe]).astype(float)
            CC = cCa.astype(float).copy()
            C, CC, ns = interval(C, CC, r, t0, t1, T_of_t, P_of_t, rim_fun, operator, cfl, centre)
            cMn, cMg, cFe, cCa = C[:, 0].copy(), C[:, 1].copy(), C[:, 2].copy(), CC.copy()
            nsteps += ns

    # ---- retrograde: full domain, rim frozen at the last growth value --
    r = Rr
    C = np.column_stack([cMn, cMg, cFe]).astype(float)
    CC = cCa.astype(float).copy()
    retro = ptt[ptt[:, 2] >= tr.max() - 1.0e-2, 2]
    rim_last = np.array([Mnr[-1], Mgr[-1], Fer[-1]])

    def rim_fixed(_t):
        return rim_last

    for j in range(len(retro) - 1):
        C, CC, ns = interval(C, CC, r, retro[j] * SEC_MY, retro[j + 1] * SEC_MY,
                             T_of_t, P_of_t, rim_fixed, operator, cfl, centre)
        nsteps += ns

    return r, C, CC, nsteps, ref


# %% compare both operators


def main():
    print("=" * 74)
    print("Spatial operator x growth driving -- which one causes the gap?")
    print("identical explicit timestepping in every run")
    print("=" * 74)

    out = {}
    for drive in ("grow", "stamp"):
        for op in ("matlab", "fv"):
            r, C, CC, ns, ref = run(op, drive=drive)
            out[(op, drive)] = (C, CC)
            print(f"\n[{op:6s} | {drive:5s}]  {ns} explicit steps")
            print(f"   centre  Mn={C[0,0]:.5f} Mg={C[0,1]:.5f} Fe={C[0,2]:.5f} Ca={CC[0]:.5f}")

    print("\n" + "-" * 74)
    cz = pd.read_csv(DATA / "combined_array.csv", header=None).to_numpy(float)
    Fer, Mnr = cz[:, 4], cz[:, 6]
    ref = pd.read_csv(DATA / "combined_C_final_cart.csv", header=None).to_numpy(float)
    print("relaxation at the centre (final - zonation)")
    print(f"  zonation                 Fe={Fer[0]:.5f}  Mn={Mnr[0]:.5f}")
    for key in (("matlab", "grow"), ("fv", "grow"), ("matlab", "stamp"), ("fv", "stamp")):
        C, _ = out[key]
        print(f"  {key[0]:6s} | {key[1]:5s}     Fe={C[0,2]:.5f} (Δ{C[0,2]-Fer[0]:+.5f})   "
              f"Mn={C[0,0]:.5f} (Δ{C[0,0]-Mnr[0]:+.5f})")
    print(f"  CZGM reference           Fe={ref[0,1]:.5f} (Δ{ref[0,1]-Fer[0]:+.5f})   "
          f"Mn={ref[0,3]:.5f} (Δ{ref[0,3]-Mnr[0]:+.5f})")

    print("\nFe profile, deviation from the CZGM reference:")
    r = cz[:, 0]
    hdr = f"{'x (um)':>9}" + "".join(f"{k[0][:5]+'/'+k[1][:5]:>13}" for k in out)
    print(hdr)
    for k in range(0, len(r), 4):
        rf = float(np.interp(r[k], ref[:, 0], ref[:, 1]))
        row = f"{r[k]:9.3f}"
        for key in out:
            row += f"{out[key][0][k,2]-rf:13.2e}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
