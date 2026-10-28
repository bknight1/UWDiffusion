"""Multicomponent interdiffusion matrices.

Single home for the Lasaga (1979) mean-field reduction, in both of the forms
this package needs:

* :func:`lasaga_reduced_diffusivity` -- numeric ``(N-1)x(N-1)`` matrix, used by
  :class:`UWDiffusion.fd_1d.MulticomponentDiffusion1D`.
* :func:`lasaga_matrix` -- the same expression as a SymPy matrix, used by
  :class:`UWDiffusion.Models.MulticomponentDiffusionModel` and by the
  examples/tutorials, which build their matrices symbolically.
* :func:`reduce_diffusion_matrix` -- generic closure reduction of an already
  assembled N x N matrix.

Keeping the two representations in one module is deliberate: the formula is
otherwise hand-rolled in several examples and tutorials, and each restatement
is a fresh chance to change a factor.  The two forms are asserted to agree in
``tests/test_1d_2d_coupling.py``.

Theory
------
Lasaga (1979), mean-field approximation for multicomponent
**ideal ionic solutions**::

    D_ij = D_i^* * delta_ij
           - (D_i^* * z_i * z_j * X_i / sum_k z_k^2 * X_k * D_k^*) * (D_j^* - D_n^*)

For an isovalent series -- Fe, Mg, Mn and Ca on the garnet dodecahedral site,
all divalent -- the charge factors cancel identically and this reduces to the
form evaluated here::

    D_ij = D_i * delta_ij
           - (D_i * X_i / sum_k D_k X_k) * (D_j - D_dep)

That reduction is exact for equal charges and is **not valid for
mixed-valence systems** (e.g. Fe2+/Fe3+), where the charge factors do not
cancel.

This is *not* the classical Darken binary coefficient
``D = X_B D_A^* + X_A D_B^*``.  Both close the unknown off-diagonal Onsager
coefficient, but Darken partitions the flux-conserving drift by mole fraction
``X_i`` where Lasaga's mean field partitions it by ``D_i X_i / sum_k X_k D_k``;
for a binary with ``D = [1, 2]``, ``X = [0.5, 0.5]`` they give 1.5 and 1.333.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def lasaga_reduced_diffusivity(
    tracer_diffusivities: Sequence[float] | np.ndarray,
    composition: Sequence[float] | np.ndarray,
    dependent_component: int | str = -1,
) -> np.ndarray:
    """Build the reduced (N-1) x (N-1) interdiffusion matrix from tracer diffusivities.

    Implements the mean-field approximation for multicomponent **ideal ionic
    solutions** of Lasaga (1979), which closes the Onsager cross
    terms that tracer experiments do not provide:

        D_ij = D_i^* * delta_ij
               - (D_i^* * z_i * z_j * X_i / sum_k z_k^2 * X_k * D_k^*)
                 * (D_j^* - D_n^*)

    where ``D_i^*`` are the tracer (self) diffusivities of the N components,
    ``X_k`` the mole fractions, ``z_k`` the ionic charges, and ``D_n^*`` the
    tracer diffusivity of the dependent (closure) component.  The result is
    ``(N-1) x (N-1)`` because closure (``X_dep = 1 - sum_i X_i``) removes one
    degree of freedom.

    This is the **isovalent** form: the charge factors cancel exactly when all
    species carry the same charge (Fe, Mg, Mn and Ca on the garnet dodecahedral
    site are all divalent), so no further approximation is involved.  It is
    **not valid for mixed-valence systems** (e.g. Fe2+/Fe3+), and it is **not**
    the classical Darken binary coefficient.  See the module docstring for the
    charged form and for the difference from Darken.

    Parameters
    ----------
    tracer_diffusivities : array_like
        1D array of tracer diffusivities of length N, one per component.
    composition : array_like
        Mole fractions of all N components at the evaluation point
        (sum must be > 0).
    dependent_component : int or str, optional
        Index (0-based) of the dependent component. A string is only allowed
        when ``composition`` is a mapping; for arrays an index is expected.
        Default is -1 (the last component, i.e. Ca in CZGM).

    Returns
    -------
    np.ndarray
        Reduced (N-1) x (N-1) interdiffusion matrix.

    Raises
    ------
    ValueError
        If inputs are inconsistent or the denominator sum(D_k X_k) <= 0.
    """
    D = np.asarray(tracer_diffusivities, dtype=float)
    X = np.asarray(composition, dtype=float)

    if isinstance(dependent_component, str):
        raise TypeError(
            "dependent_component for lasaga_reduced_diffusivity must be an "
            "integer index (a string name is not resolvable from arrays)."
        )

    if D.ndim != 1 or X.ndim != 1:
        raise ValueError(
            f"tracer_diffusivities and composition must be 1D arrays, got "
            f"{D.shape} and {X.shape}."
        )
    if D.shape[0] != X.shape[0]:
        raise ValueError(
            f"tracer_diffusivities ({D.shape[0]}) and composition ({X.shape[0]}) "
            "must have the same length N."
        )
    N = D.shape[0]
    if N < 2:
        raise ValueError(f"Need at least 2 components, got N={N}.")
    if np.any(D < 0.0):
        raise ValueError("Tracer diffusivities must be non-negative.")
    if np.any(X < 0.0):
        raise ValueError("Mole fractions must be non-negative.")

    dep = int(dependent_component) % N
    denom = float(np.sum(D * X))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(
            f"Invalid mean-field denominator sum(D_k X_k) = {denom}; it must be "
            "positive and finite."
        )

    indep = [k for k in range(N) if k != dep]
    P = N - 1
    D_red = np.zeros((P, P), dtype=float)
    for i_new, i in enumerate(indep):
        for j_new, j in enumerate(indep):
            D_red[i_new, j_new] = D[i] * (1.0 if i == j else 0.0) - (
                D[i] * X[i] / denom
            ) * (D[j] - D[dep])
    return D_red


def reduce_diffusion_matrix(
    D_full: np.ndarray,
    dependent_idx: int = -1,
) -> np.ndarray:
    """Reduce an N x N multicomponent diffusion matrix to an (N-1) x (N-1) interdiffusion matrix.

    Given N chemical components whose mole fractions sum to unity (sum_k X_k = 1),
    the concentration gradient of the dependent component d satisfies:
        grad(X_d) = - sum_{j != d} grad(X_j).

    Substituting this into the flux relation:
        J_i = - sum_k D_ik grad(X_k)
    yields for all independent components i != d:
        J_i = - sum_{j != d} (D_ij - D_id) grad(X_j).

    Therefore, the reduced matrix entries are:
        D_red[i, j] = D[i, j] - D[i, d].

    Parameters
    ----------
    D_full : np.ndarray
        Full N x N diffusion matrix.
    dependent_idx : int, optional
        Index of the dependent component. Default is -1 (the last component).

    Returns
    -------
    np.ndarray
        Reduced (N-1) x (N-1) interdiffusion matrix.

    Raises
    ------
    ValueError
        If D_full is not square or has dimension < 2.
    """
    D_arr = np.asarray(D_full, dtype=float)
    if D_arr.ndim != 2 or D_arr.shape[0] != D_arr.shape[1]:
        raise ValueError(f"D_full must be a 2D square matrix, got shape {D_arr.shape}.")
    N = D_arr.shape[0]
    if N < 2:
        raise ValueError(f"D_full must have dimension at least 2, got {N}.")

    dep = dependent_idx % N
    indep = [k for k in range(N) if k != dep]

    D_red = np.zeros((N - 1, N - 1), dtype=float)
    for i_new, i in enumerate(indep):
        for j_new, j in enumerate(indep):
            D_red[i_new, j_new] = D_arr[i, j] - D_arr[i, dep]

    return D_red


def lasaga_matrix(d_symbols: Sequence[Any], x_symbols: Sequence[Any]) -> Any:
    """Build the symbolic (n-1)x(n-1) Lasaga mean-field matrix.

    The SymPy counterpart of :func:`lasaga_reduced_diffusivity`, for
    :class:`UWDiffusion.Models.MulticomponentDiffusionModel`, which takes a
    symbolic diffusion matrix and resolves its composition symbols at solve
    time.  Keeping the two in one module is what stops them drifting apart.

    Parameters
    ----------
    d_symbols : sequence of sympy expressions
        Tracer diffusivities, one per component, in component order -- e.g.
        ``[D_Fe, D_Mg, D_Mn, D_Ca]``.  These names are what you later pass to
        ``set_diffusion_values``.
    x_symbols : sequence of sympy expressions
        Mole fractions in the *same* order.  These **must** be named
        ``xi_1 .. xi_n``.  ``MulticomponentDiffusionModel`` discovers them by
        regex on the symbol name and substitutes the current unknowns, so any
        other naming is silently left unsubstituted.  The last one is the
        closure component and resolves to ``1 - sum(x_1..x_{n-1})``.

    Returns
    -------
    sympy.Matrix
        ``(n-1) x (n-1)`` matrix; closure has already eliminated the last row
        and column.

    Raises
    ------
    ValueError
        If the two symbol sequences differ in length, or fewer than 2
        components are given.

    Examples
    --------
    >>> import sympy as sp
    >>> D_Fe, D_Mg, D_Mn, D_Ca = sp.symbols("D_Fe D_Mg D_Mn D_Ca")
    >>> xi = sp.symbols("xi_1:5")
    >>> lasaga_matrix([D_Fe, D_Mg, D_Mn, D_Ca], list(xi)).shape
    (3, 3)
    """
    import sympy  # local: the numeric path above must not require SymPy

    if len(d_symbols) != len(x_symbols):
        raise ValueError(
            f"d_symbols and x_symbols must be the same length, got "
            f"{len(d_symbols)} and {len(x_symbols)}."
        )
    n = len(d_symbols)
    if n < 2:
        raise ValueError(f"Need at least 2 components, got {n}.")

    n_ind = n - 1
    denom = sum(d_symbols[k] * x_symbols[k] for k in range(n))
    matrix = sympy.zeros(n_ind, n_ind)
    for i in range(n_ind):
        for j in range(n_ind):
            matrix[i, j] = (
                d_symbols[i] * (1 if i == j else 0)
                - (d_symbols[i] * x_symbols[i]) / denom * (d_symbols[j] - d_symbols[-1])
            )
    return matrix
