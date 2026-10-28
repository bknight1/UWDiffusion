# %%
"""Approximation-theory floor for the p-vs-h question.

Does raising the element degree let you use a coarser mesh? The cleanest
answer does not need a solver run at all.

If ``u_h`` is the L2 projection of the exact solution onto piecewise
degree-``p`` polynomials on a mesh of spacing ``h``, then ``u_h`` is the best
approximation available at that ``(h, p)`` in the L2 norm. No finite-element
solve can beat it. Measuring ``||u - u_h||`` therefore isolates the spatial
approximation question from time error, nonlinear tolerance, and mesh
generation.

Two functions are compared:

``smooth``
    the first Neumann radial eigenfunction of the annulus ``0.5 <= r <= 1``,
    i.e. the exact solution shape the UWDiffusion convergence study uses.
``front``
    ``tanh((r - r0) / eps)`` with small ``eps`` -- a smooth surrogate for a
    compositional step (growth shell, Heaviside initial condition), where
    theory predicts ``p``-refinement stops helping.

Reporting both norms matters: theory gives ``O(h^(p+1))`` in L2 but only
``O(h^p)`` in max-norm, so a max-norm can hide the very superconvergence that
makes high degree attractive.

Run from the repository root (numpy + scipy only; no Underworld3 needed):

    python examples/p_vs_h_projection_study.py
"""

# %%
import json
from pathlib import Path

import numpy as np
from scipy.special import j0, j1, y0, y1
from scipy.optimize import brentq

# %%
# Exact target functions on the annulus 0.5 <= r <= 1.0.

R_INNER, R_OUTER = 0.5, 1.0
FRONT_R0, FRONT_EPS = 0.75, 0.01


def first_neumann_wavenumber(r_inner: float, r_outer: float) -> float:
    """First positive annulus radial eigenvalue satisfying u'=0 at both rims."""

    def determinant(k: float) -> float:
        return j1(k * r_inner) * y1(k * r_outer) - j1(k * r_outer) * y1(k * r_inner)

    return float(brentq(determinant, 6.0, 7.0, xtol=1e-14))


def radial_mode(r: np.ndarray, k: float, r_inner: float, r_outer: float) -> np.ndarray:
    """Neumann radial Laplacian eigenfunction, normalised to max(abs)=1.

    Evaluated directly from the Bessel combination -- no grid interpolation,
    which would cap the achievable accuracy near 1e-9 and pollute the p=3 rates.
    """
    u = j0(k * r) * y1(k * r_outer) - y0(k * r) * j1(k * r_outer)
    dense_r = np.linspace(r_inner, r_outer, 20001)
    dense_u = j0(k * dense_r) * y1(k * r_outer) - y0(k * dense_r) * j1(k * r_outer)
    return u / np.max(np.abs(dense_u))


def front(r: np.ndarray) -> np.ndarray:
    """Smooth surrogate for a compositional step."""
    return np.tanh((r - FRONT_R0) / FRONT_EPS)


# %%
# L2 projection of f onto piecewise degree-p polynomials on an equispaced mesh.


def l2_project(f, edges: np.ndarray, p: int, n_quad: int = 40):
    """Project f onto piecewise polynomials of degree p.

    Uses a Gauss-Legendre rule per element and a least-squares fit of degree p,
    which is exactly the element-wise L2 projection.
    """
    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    coeffs = []
    for a, b in zip(edges[:-1], edges[1:]):
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        x = mid + half * xq
        # Legendre basis on [-1,1] is well conditioned for this fit.
        basis = np.polynomial.legendre.legvander(xq, p)
        rhs = (basis.T * wq) @ f(x)
        mass = basis.T @ (wq[:, None] * basis)
        coeffs.append(np.linalg.solve(mass, rhs))
    return edges, coeffs


def evaluate(edges, coeffs, x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for a, b, c in zip(edges[:-1], edges[1:], coeffs):
        m = (x >= a) & (x <= b if b == edges[-1] else x < b)
        if not m.any():
            continue
        t = 2.0 * (x[m] - 0.5 * (b + a)) / (b - a)
        out[m] = np.polynomial.legendre.legval(t, c)
    return out


def measure(f, edges, p: int, n_samples: int = 20001):
    """Return (L2 error, max-abs error) of the piecewise degree-p projection."""
    edges, coeffs = l2_project(f, edges, p)
    x = np.linspace(edges[0], edges[-1], n_samples)
    err = np.abs(f(x) - evaluate(edges, coeffs, x))
    # Trapezoidal L2 norm over the physical coordinate.
    l2 = float(np.sqrt(np.trapezoid(err**2, x)))
    return l2, float(np.max(err))


# %%
# Sweep A: geometric mesh ladder per degree -> fitted rates.
# Sweep B: matched DOF budgets -> the actual p-vs-h trade-off.

# For equal cost across degrees on the element-space dimension n*(p+1),
# choose n1 = 6k, n2 = 4k, n3 = 3k so all three give 12k DOF.


def run_rates(name, f, n_levels=6):
    rows = []
    for p in (1, 2, 3):
        for level in range(n_levels):
            n = 10 * (2**level)
            edges = np.linspace(R_INNER, R_OUTER, n + 1)
            h = (R_OUTER - R_INNER) / n
            l2, mx = measure(f, edges, p)
            rows.append({"function": name, "p": p, "n_elements": n, "h": h,
                         "dof": n * (p + 1), "l2": l2, "max": mx})
    return rows


def run_matched_dof(name, f, ks=(4, 8, 16, 32)):
    rows = []
    for k in ks:
        for p, mult in ((1, 6), (2, 4), (3, 3)):
            n = mult * k
            edges = np.linspace(R_INNER, R_OUTER, n + 1)
            h = (R_OUTER - R_INNER) / n
            l2, mx = measure(f, edges, p)
            rows.append({"function": name, "p": p, "n_elements": n, "h": h,
                         "dof": n * (p + 1), "l2": l2, "max": mx})
    return rows


def fit_rate(rows, p, key, n_finest=3):
    sel = sorted([r for r in rows if r["p"] == p], key=lambda r: -r["h"])[:n_finest]
    hs = np.array([r["h"] for r in sel])
    es = np.array([r[key] for r in sel])
    return float(np.polyfit(np.log(hs), np.log(es), 1)[0])


# %%
k = first_neumann_wavenumber(R_INNER, R_OUTER)
smooth = lambda r: radial_mode(r, k, R_INNER, R_OUTER)

rate_rows = run_rates("smooth", smooth) + run_rates("front", front)
dof_rows = run_matched_dof("smooth", smooth) + run_matched_dof("front", front)

# %%
# Overshoot: a composition must stay in [0, 1]. The L2 projection of a
# monotone step is NOT monotone, and the violation grows with degree -- this
# is the practical reason to avoid high p at a growth interface, independent
# of any error norm. Uses a monotone step in [0,1] so overshoot is well posed.


def step(r: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((r - FRONT_R0) / FRONT_EPS))


def overshoot(edges, p, f, n_samples=20001):
    e, c = l2_project(f, edges, p)
    x = np.linspace(e[0], e[-1], n_samples)
    uh = evaluate(e, c, x)
    lo = float(max(0.0, -uh.min()))
    hi = float(max(0.0, uh.max() - 1.0))
    return lo, hi, float(max(lo, hi))


print("### Overshoot outside the physical range [0, 1] at a monotone step")
print()
print("| h | p=1 | p=2 | p=3 | worst offender |")
print("|---|---|---|---|---|")
for n in (20, 40, 80, 160):
    edges = np.linspace(R_INNER, R_OUTER, n + 1)
    h = (R_OUTER - R_INNER) / n
    vals = {p: overshoot(edges, p, step)[2] for p in (1, 2, 3)}
    worst = max(vals, key=vals.get)
    print("| {:.4g} | {:.3e} | {:.3e} | {:.3e} | p={} |".format(
        h, vals[1], vals[2], vals[3], worst))
print()

# %%
print(f"first Neumann wavenumber k = {k:.12g}")
print()
print("### Fitted spatial rates (approximation-theory floor)")
print()
print("| function | p | rate L2 | rate max | L2 err (finest) | max err (finest) |")
print("|---|---|---|---|---|---|")
for name in ("smooth", "front"):
    sub = [r for r in rate_rows if r["function"] == name]
    finest = min(r["h"] for r in sub)
    for p in (1, 2, 3):
        best = [r for r in sub if r["p"] == p and r["h"] == finest][0]
        print("| {} | {} | {:.2f} | {:.2f} | {:.3e} | {:.3e} |".format(
            name, p, fit_rate(sub, p, "l2"), fit_rate(sub, p, "max"),
            best["l2"], best["max"]))
print()

# %%
print("### Equal-DOF comparison (same total DOF per row)")
print()
for name in ("smooth", "front"):
    sub = [r for r in dof_rows if r["function"] == name]
    print(f"**{name}**")
    print("| dof | p=1 max | p=2 max | p=3 max | best | p3/p1 | p3/p2 |")
    print("|---|---|---|---|---|---|---|")
    for d in sorted({r["dof"] for r in sub}):
        cell = {r["p"]: r for r in sub if r["dof"] == d}
        winner = min(cell, key=lambda pp: cell[pp]["max"])
        print("| {} | {:.2e} | {:.2e} | {:.2e} | p={} | {:.3f} | {:.3f} |".format(
            d, cell[1]["max"], cell[2]["max"], cell[3]["max"], winner,
            cell[3]["max"] / cell[1]["max"], cell[3]["max"] / cell[2]["max"]))
    print()

# %%
out = Path(__file__).resolve().parent / "output"
out.mkdir(exist_ok=True)
(out / "p_vs_h_projection.json").write_text(json.dumps({"rates": rate_rows, "matched_dof": dof_rows}, indent=2))
print(f"wrote {out / 'p_vs_h_projection.json'}")
