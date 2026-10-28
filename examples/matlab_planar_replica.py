# %%
"""Literal numpy reimplementation of the MATLAB CZGM planar branch.

MATLAB (``make_diffusion.m`` :: ``diff_step``, ``else % linear`` branch)::

    C(1,:) = C(2,:)                      ; % centre: zero gradient ghost
    A(1,:) = C(1,:); A(nc,:) = C(nc,:)   ; % centre + rim frozen this step
    for i = 2:nc-1
        D  = Di(:,:,i);  Dp = Di(:,:,ip);  dD = Dp - D;
        A(i,:) = C(i,:) + QQ*( dD*(C(ip,:)-C(i,:))' ...
                             + D*(C(ip,:)-2*C(i,:)+C(im,:))' )';
    end
    C = A;
    dt = 0.3*dx^2/De ;  QQ = dt/dx^2

with ``De = max|eig(DD)|`` evaluated at the centre and the rim, and

    Q = sum_k DD_k c_k
    D(i,j) = DD_i*delta_ij - (DD_i*c_i/Q)*(DD_j - DD_4)

This script drives that stencil with the CZGM growth history and compares the
result against ``combined_C_final_cart.csv``.  If it matches, the difference
between our solvers and the CZGM reference is DISCRETISATION, not physics, and
the offending term can be isolated by switching one ingredient at a time.

Run:  python development/multicomponent_solver/matlab_planar_replica.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

SEC_MY = 1.0e6 * 365.25 * 24.0 * 3600.0
DATA = Path(__file__).resolve().parents[1] / "Tutorials" / "Diffusion" / "CZGM_data"


# %% com_D -- Chakraborty & Ganguly 1992, Ca = 0.05 Mn  (MATLAB diff_mode=1)


def tracer_um2s(temp_C, pressure_bar):
    """Tracer diffusivities in um^2/s, matching MATLAB com_D(diff_mode=1)."""
    D0 = {"Mn": 5.15e-4, "Mg": 1.11e-3, "Fe": 6.36e-4}   # cm^2/s
    Ea = {"Mn": 60569.0, "Mg": 67997.0, "Fe": 65824.0}   # cal/mol
    Ev = {"Mn": 6.04, "Mg": 5.27, "Fe": 5.63}            # cm^3/mol
    R = 1.9872

    def _D(e):
        # MATLAB: DDFe = 6.36e-4*1e8  ->  cm^2/s to um^2/s is x 1e8
        return D0[e] * 1e8 * np.exp(
            -(Ea[e] + (pressure_bar - 1.0) * Ev[e] / 41.84) / (R * (temp_C + 273.0))
        )
    DMn, DMg, DFe = _D("Mn"), _D("Mg"), _D("Fe")
    DCa = 0.05 * DMn
    return DMn, DMg, DFe, DCa


def com_D(pressure_bar, temp_C, cMn, cMg, cFe, cCa):
    """Reduced 3x3 matrix in MATLAB's component order (Mn, Mg, Fe).

    Argument order matches MATLAB exactly: ``com_D(p, T, cMn, cMg, cFe, cCa)``
    with ``p`` in bar and ``T`` in degrees C.  Getting these two backwards is
    easy and catastrophic -- it evaluates the Arrhenius law at thousands of
    degrees and inflates D by ~15 orders of magnitude.
    """
    DD1, DD2, DD3, DD4 = tracer_um2s(temp_C, pressure_bar)
    Q = DD1 * cMn + DD2 * cMg + DD3 * cFe + DD4 * cCa
    DD = np.zeros((3, 3))
    DDv = (DD1, DD2, DD3)
    cv = (cMn, cMg, cFe)
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            DD[i, j] = DDv[i] * delta - (DDv[i] * cv[i] / Q) * (DDv[j] - DD4)
    De = float(np.max(np.abs(np.linalg.eigvals(DD))))
    return DD, De


# %% diff_step -- planar branch


def diff_step_planar(cMn, cMg, cFe, cCa, r, t_start_s, t_end_s,
                     T_of_t, P_of_t, rim_bc="path"):
    """Advance one growth interval with MATLAB's planar stencil.

    ``rim_bc``:
      'path'   -- BCond_creation=3: rim set to the path composition at the
                  current time (MATLAB's growth-phase condition)
      'fixed'  -- rim held at the supplied end member of the interval
    """
    nc = len(r)
    dx = r[1] - r[0]
    C = np.column_stack([cMn, cMg, cFe]).astype(float)
    CC = np.asarray(cCa, dtype=float).copy()

    tim = float(t_start_s)
    t_end = float(t_end_s)
    nsteps = 0
    max_steps = 100_000
    while nsteps < max_steps:
        if t_end - tim <= 1.0e-6 * max(1.0, abs(t_end)):
            break
        # D at centre and rim -> explicit stability limit
        tm_My = tim / SEC_MY
        Pa = float(P_of_t(tm_My))
        Ta = float(T_of_t(tm_My)) - 273.0
        _, De1 = com_D(Pa, Ta, C[0, 0], C[0, 1], C[0, 2], CC[0])
        _, De3 = com_D(Pa, Ta, C[-1, 0], C[-1, 1], C[-1, 2], CC[-1])
        De = max(De1, De3)
        if not np.isfinite(De) or De <= 0.0:
            raise RuntimeError(f"bad De={De} at t={tm_My} Myr")

        dt = 0.3 * dx * dx / De
        dt = min(dt, t_end - tim)
        if not np.isfinite(dt) or dt <= 0.0:
            break
        QQ = dt / (dx * dx)

        # --- boundary conditions -------------------------------------
        C[0, :] = C[1, :]                       # centre: zero gradient
        if rim_bc == "path":
            tm1_My = (tim + dt) / SEC_MY
            Pa1 = float(P_of_t(tm1_My))
            Ta1 = float(T_of_t(tm1_My)) - 273.0
            # MATLAB BCond_creation=3 interpolates the path end-members.
            # Approximated here by the interval's rim value (the growth rows).
            rim = C[-1, :].copy()
        else:
            Pa1, Ta1 = Pa, Ta
            rim = C[-1, :].copy()
        C[-1, :] = rim

        A = np.empty_like(C)
        A[0, :] = C[0, :]
        A[-1, :] = C[-1, :]

        # D at every node
        Di = np.empty((nc, 3, 3))
        for i in range(nc):
            Di[i], _ = com_D(Pa, Ta, C[i, 0], C[i, 1], C[i, 2], CC[i])

        for i in range(1, nc - 1):
            im, ip = i - 1, i + 1
            D = Di[i]
            dD = Di[ip] - D
            A[i, :] = C[i, :] + QQ * (
                dD @ (C[ip, :] - C[i, :]) + D @ (C[ip, :] - 2 * C[i, :] + C[im, :])
            )

        C = A
        CC[:] = 1.0 - C[:, 0] - C[:, 1] - C[:, 2]
        tim += dt
        nsteps += 1

    return C[:, 0], C[:, 1], C[:, 2], CC, nsteps


# %% main


def main():
    a = pd.read_csv(DATA / "combined_array.csv", header=None).to_numpy(float)
    ptt = pd.read_csv(DATA / "PTt_path.csv", header=None).to_numpy(float)
    ref = pd.read_csv(DATA / "combined_C_final_cart.csv", header=None).to_numpy(float)

    Rr, tr, Pr, Tr, Fer, Mgr, Mnr, Car = [a[:, i] for i in range(8)]
    T_of_t = interpolate.interp1d(ptt[:, 2], ptt[:, 1], fill_value="extrapolate")
    P_of_t = interpolate.interp1d(ptt[:, 2], ptt[:, 0], fill_value="extrapolate")

    sec_per_myr = SEC_MY

    # Growth rows: the grain grows from the core radius outward.
    growth = tr > tr.min()
    Rg, tg = Rr[growth], tr[growth]
    Fg, Mg, Ng, Cg = Fer[growth], Mgr[growth], Mnr[growth], Car[growth]

    # Initial state: the uniform core over the first four radii.
    n_core = int(np.sum(~growth))
    cMn = np.array(Mnr[:n_core], dtype=float)
    cMg = np.array(Mgr[:n_core], dtype=float)
    cFe = np.array(Fer[:n_core], dtype=float)
    cCa = np.array(Car[:n_core], dtype=float)
    r = np.array(Rr[:n_core], dtype=float)

    print(f"core: {n_core} nodes, r = 0 .. {r[-1]:.3f} um, t = {tr[0]:.4f} Myr")
    print(f"growth steps: {len(Rg)}  (to {Rg[-1]:.3f} um at {tg[-1]:.4f} Myr)")

    total_steps = 0
    for i in range(len(Rg)):
        # MATLAB grows the domain: add the new shell node at Rg[i].
        r = np.append(r, Rg[i])
        cMn = np.append(cMn, Ng[i])
        cMg = np.append(cMg, Mg[i])
        cFe = np.append(cFe, Fg[i])
        cCa = np.append(cCa, Cg[i])

        t0 = tg[i - 1] * sec_per_myr if i > 0 else tr[0] * sec_per_myr
        t1 = tg[i] * sec_per_myr
        if t1 <= t0:
            continue
        cMn, cMg, cFe, cCa, ns = diff_step_planar(
            cMn, cMg, cFe, cCa, r, t0, t1, T_of_t, P_of_t, rim_bc="path")
        total_steps += ns
        if i % 7 == 0 or i == len(Rg) - 1:
            print(f"  growth {i + 1:2d}/{len(Rg)}: r={r[-1]:6.2f} um, "
                  f"{ns:5d} steps, Fe[0]={cFe[0]:.5f}")

    print(f"growth phase done: {len(r)} nodes, {total_steps} explicit steps")
    print(f"centre: Fe={cFe[0]:.5f} Mg={cMg[0]:.5f} Mn={cMn[0]:.5f} Ca={cCa[0]:.5f}")

    print(f"\n{'x (um)':>9} {'Fe':>10} {'Mg':>10} {'Mn':>10} {'Ca':>10}")
    print("  replica minus CZGM reference")
    names = ("Fe", "Mg", "Mn", "Ca")
    for k in range(0, len(r), max(1, len(r) // 8)):
        row = [np.interp(r[k], ref[:, 0], ref[:, j + 1]) for j in range(4)]
        got = [cFe[k], cMg[k], cMn[k], cCa[k]]
        print(f"{r[k]:9.3f} " + " ".join(f"{g - v:10.2e}" for g, v in zip(got, row)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
