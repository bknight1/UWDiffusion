"""1D Finite-Difference / Finite-Volume Multicomponent Diffusion Model.

Provides conservative, vertex-centered finite-volume discretization for multicomponent
diffusion in planar (alpha=0), cylindrical (alpha=1), and spherical (alpha=2) geometries,
designed for geochemical and petrological applications such as garnet diffusion zoning.
"""

from __future__ import annotations

import copy
import inspect
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

GeometryType = Literal["planar", "cylindrical", "spherical"]


def _resolve_vertex_positions(
    positions: int | float | Sequence[float | int] | np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Resolve pin positions (vertex indices, radii, or a boolean mask) to vertex indices.

    Positions may be:
    - integer vertex indices (validated against the grid),
    - float radii (matched to the nearest grid vertex),
    - an array-like of indices and/or radii, or
    - a boolean mask of length ``len(r)``.

    Parameters
    ----------
    positions : int, float, array_like, or np.ndarray
        Vertex index(s), radius/radii, or boolean mask selecting grid vertices.
    r : np.ndarray
        The model's grid coordinate array.

    Returns
    -------
    np.ndarray
        Sorted, de-duplicated array of vertex indices.

    Raises
    ------
    ValueError
        If an index is out of range, a boolean mask has the wrong length, or a
        radius lies outside the grid.
    """
    arr = np.asarray(positions)
    n_points = r.shape[0]
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.dtype == bool:
        if arr.shape[0] != n_points:
            raise ValueError(
                f"Boolean pin mask must have length n_points={n_points}, "
                f"got length {arr.shape[0]}."
            )
        idx = np.flatnonzero(arr)
    elif np.issubdtype(arr.dtype, np.integer):
        idx = arr.astype(int, copy=False)
        if np.any(idx < 0) or np.any(idx >= n_points):
            bad = sorted(set(int(v) for v in idx if v < 0 or v >= n_points))
            raise ValueError(
                f"Pin vertex indices must be in [0, {n_points - 1}], got {bad}."
            )
    else:
        # Float radii: match to the nearest grid vertex.
        radii = arr.astype(float, copy=False)
        if np.any(radii < r[0]) or np.any(radii > r[-1]):
            raise ValueError(
                f"Pin radii must lie within the grid [{r[0]:.6g}, {r[-1]:.6g}]."
            )
        idx = np.array(
            [int(np.argmin(np.abs(r - float(v)))) for v in radii], dtype=int
        )

    return np.unique(idx)


# Parameter-name hints used to disambiguate callable diffusivity signatures
# between radius-first D(r, X), time-first D(t, X) and time-only D(t).
_TIME_PARAM_NAMES = frozenset(
    {
        "t",
        "time",
        "tau",
        "tm",
        "tsec",
        "t_sec",
        "ts",
        "tmyr",
        "t_myr",
        "time_myr",
        "time_yr",
        "age",
        "year",
        "myr",
        "yr",
    }
)
_RADIUS_PARAM_NAMES = frozenset(
    {
        "r",
        "rr",
        "x",
        "rad",
        "radius",
        "dist",
        "distance",
        "position",
        "pos",
        "coord",
        "coordinate",
    }
)


from .diffusion_matrix import (
    lasaga_reduced_diffusivity,
    reduce_diffusion_matrix,
)


@dataclass
class BoundaryCondition:
    """Specification of a boundary condition on one side of the 1D domain.

    Attributes
    ----------
    bc_type : {'dirichlet', 'neumann'}
        Type of boundary condition:
        - 'dirichlet': fixed concentration at the boundary vertex.
        - 'neumann': prescribed gradient dX/dr at the boundary (default 0 for zero flux).
    value : float or dict or np.ndarray
        Prescribed value(s). Can be a scalar (applied to all components),
        a dict mapping component names to floats, or an array for independent components.
    """

    bc_type: Literal["dirichlet", "neumann"]
    value: float | dict[str, float] | np.ndarray = 0.0

    def __post_init__(self) -> None:
        norm_type = self.bc_type.strip().lower()
        if norm_type in ("dirichlet", "value", "fixed"):
            self.bc_type = "dirichlet"
        elif norm_type in ("neumann", "flux", "gradient", "zero_flux"):
            self.bc_type = "neumann"
        else:
            raise ValueError(
                f"Unknown bc_type '{self.bc_type}'. Expected 'dirichlet' or 'neumann'."
            )


@dataclass
class SimulationResult:
    """Container for time-series output of a 1D multicomponent diffusion simulation.

    Attributes
    ----------
    r : np.ndarray
        1D coordinates of the grid vertices.
    times : np.ndarray
        1D array of output times.
    compositions : np.ndarray
        3D array of shape (n_times, n_points, n_components) containing mole fractions.
    components : list of str
        List of component names.
    geometry : str
        Geometry name ('planar', 'cylindrical', 'spherical').
    alpha : int
        Geometry factor (0, 1, or 2).
    """

    r: np.ndarray
    times: np.ndarray
    compositions: np.ndarray
    components: list[str]
    geometry: str
    alpha: int

    @property
    def profiles(self) -> dict[str, np.ndarray]:
        """Dictionary mapping each component name to a 2D array of shape (n_times, n_points)."""
        return {
            name: self.compositions[:, :, i] for i, name in enumerate(self.components)
        }

    @property
    def final_composition(self) -> dict[str, np.ndarray]:
        """Dictionary mapping each component name to its 1D profile at the final time."""
        return {
            name: self.compositions[-1, :, i] for i, name in enumerate(self.components)
        }

    def __getitem__(self, key: str | int | tuple[Any, ...]) -> np.ndarray:
        """Allow subscripting by component name or array slicing."""
        if isinstance(key, str):
            if key in self.components:
                idx = self.components.index(key)
                return self.compositions[:, :, idx]
            raise KeyError(
                f"Component '{key}' not recognized. Available: {self.components}"
            )
        return self.compositions[key]

    def get_profile(
        self,
        component: str | int,
        time_idx: int = -1,
    ) -> np.ndarray:
        """Return 1D array of concentration profile for a component at a specific time index.

        Parameters
        ----------
        component : str or int
            Component name or index.
        time_idx : int, optional
            Index of the time step. Default is -1 (final step).

        Returns
        -------
        np.ndarray
            1D array of concentration values along r.
        """
        if isinstance(component, str):
            if component not in self.components:
                raise KeyError(f"Component '{component}' not in {self.components}")
            idx = self.components.index(component)
        else:
            idx = int(component)
        return self.compositions[time_idx, :, idx]

    def plot(
        self,
        components: list[str] | None = None,
        times_to_plot: int | Sequence[int] | None = None,
        ax: Any = None,
        figsize: tuple[float, float] = (8, 5),
        show: bool = True,
    ) -> Any:
        """Convenience method to plot concentration profiles using matplotlib.

        Parameters
        ----------
        components : list of str, optional
            Components to plot. Defaults to all components.
        times_to_plot : int, sequence of int, optional
            Indices of times to plot. If an int, plots that many evenly spaced times.
            Defaults to initial and final times.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        figsize : tuple of float, optional
            Figure size if creating a new figure.
        show : bool, optional
            Whether to call plt.show().

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib is required for plotting.") from err

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if components is None:
            comps = self.components
        else:
            comps = [c for c in components if c in self.components]

        n_t = len(self.times)
        if times_to_plot is None:
            t_indices = [0, n_t - 1] if n_t > 1 else [0]
        elif isinstance(times_to_plot, int):
            t_indices = np.linspace(0, n_t - 1, times_to_plot, dtype=int).tolist()
        else:
            t_indices = list(times_to_plot)

        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in t_indices:
            if idx not in seen and 0 <= idx < n_t:
                seen.add(idx)
                unique_indices.append(idx)

        for comp in comps:
            c_idx = self.components.index(comp)
            for idx in unique_indices:
                t_val = self.times[idx]
                label = (
                    f"{comp} (t={t_val:.3g})"
                    if len(unique_indices) <= 6
                    else (comp if idx == unique_indices[0] else None)
                )
                alpha_val = 0.4 + 0.6 * (idx / max(1, n_t - 1))
                ax.plot(
                    self.r,
                    self.compositions[idx, :, c_idx],
                    label=label,
                    alpha=alpha_val,
                )

        ax.set_xlabel("r")
        ax.set_ylabel("Mole Fraction X")
        ax.set_title(f"1D Diffusion Profiles ({self.geometry.capitalize()})")
        if len(unique_indices) <= 6:
            ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()
        return ax


class MulticomponentDiffusion1D:
    """1D conservative vertex-centered finite-volume multicomponent diffusion model.

    Solves the coupled multicomponent diffusion equation for N-1 independent components:
        dX_i/dt = 1/r^alpha * d/dr [ r^alpha sum_j D_ij dX_j/dr ]
    where:
        - alpha = 0: Planar / Cartesian 1D geometry
        - alpha = 1: Cylindrical radial geometry
        - alpha = 2: Spherical radial geometry
    with the closure relation for the dependent component:
        X_dep(r, t) = 1.0 - sum_{j != dep} X_j(r, t).

    At the coordinate origin r=0 (core / symmetry axis), the conservative finite-volume
    discretization integrates over the cell control volume:
        V_0 = int_0^{r_{1/2}} r^alpha dr = r_{1/2}^{alpha+1} / (alpha + 1),
    avoiding any singular division by r=0 and naturally enforcing the symmetry zero-flux
    condition J(0) = 0.

    Time stepping is performed using an unconditionally stable implicit Backward Euler
    scheme. For constant diffusion matrices, the system matrix is assembled and factorized
    once for rapid repeated stepping. For composition-dependent diffusion matrices,
    Picard (fixed-point) iteration is supported.

    Parameters
    ----------
    grid : array_like
        1D array of strictly increasing coordinates r with r[0] >= 0.
    diffusivity : array_like, scalar, or callable
        Diffusion matrix or diffusivity:
        - (N-1) x (N-1) reduced interdiffusion matrix
        - N x N full diffusion matrix (automatically reduced via closure)
        - Scalar (for 2-component binary system or 1-component tracer)
        - Callable D(X) or D(r, X) returning matrix or scalar
    geometry : {'planar', 'cylindrical', 'spherical'} or int, optional
        Coordinate geometry. 0/'planar', 1/'cylindrical', 2/'spherical'.
        Default is 'spherical'.
    components : list of str or int, optional
        List of component names (e.g. ['Alm', 'Prp', 'Grs', 'Sps']) or total number
        of components N. Default is ['C0', 'C1'] (N=2, binary system).
    dependent_component : str or int, optional
        Name or index of the dependent component. Default is the last component.
    bcs : dict, optional
        Dictionary specifying boundary conditions for 'left' and 'right'.
        Format: {'left': ('neumann', 0.0), 'right': ('dirichlet', {...})}.
        Default is zero-flux Neumann at both ends.
    initial_composition : dict, array_like, or callable, optional
        Initial composition profile(s).
    picard_max_iter : int, optional
        Maximum iterations for nonlinear/callable D (default: 20).
    picard_tol : float, optional
        Convergence tolerance for Picard iteration (default: 1e-8).

    Notes
    -----
    Internal point Dirichlet constraints can be imposed at interior vertices
    with :meth:`set_internal_bcs` and removed with :meth:`clear_internal_bcs`.
    These behave like Dirichlet boundary conditions applied to individual grid
    vertices and are used e.g. by segment-wise growth models that pin the
    not-yet-grown shell boundaries while the active region diffuses.
    """

    def __init__(
        self,
        grid: Sequence[float] | np.ndarray,
        diffusivity: float | Sequence[Any] | np.ndarray | Callable[..., Any],
        geometry: GeometryType | int | str = "spherical",
        components: Sequence[str] | int = 2,
        dependent_component: str | int | None = None,
        bcs: dict[str, Any] | None = None,
        initial_composition: dict[str, Any]
        | np.ndarray
        | Callable[..., Any]
        | None = None,
        picard_max_iter: int = 20,
        picard_tol: float = 1e-8,
        centre: str = "zero_flux",
    ) -> None:
        self._init_grid(grid)
        self._init_geometry(geometry)
        self._init_components(components, dependent_component)
        self._init_measures()

        if centre not in ("zero_flux", "mirror"):
            raise ValueError(
                f"centre must be 'zero_flux' or 'mirror', got {centre!r}."
            )
        self.centre = centre

        self.picard_max_iter = int(picard_max_iter)
        self.picard_tol = float(picard_tol)

        # Set boundary conditions
        self._bcs: dict[str, BoundaryCondition] = {}
        if bcs is not None:
            for b_name, b_val in bcs.items():
                self.set_bc(b_name, b_val)
        else:
            self.set_bc("left", "neumann", 0.0)
            self.set_bc("right", "neumann", 0.0)

        # Internal point Dirichlet constraints ("pins"): vertex index ->
        # {independent component index: prescribed value}. Empty by default.
        self._internal_bcs: dict[int, dict[int, float]] = {}

        # Diffusivity
        self.set_diffusivity(diffusivity)

        # State initialization
        self.time: float = 0.0
        self._X = np.zeros((self.n_points, self.n_components), dtype=float)

        if initial_composition is not None:
            self.set_initial_composition(initial_composition)
        else:
            # Default uniform composition equal for all components
            val = 1.0 / self.n_components if self.n_components > 0 else 1.0
            self._X.fill(val)

        # Cache for linear solver factorization
        self._cached_dt: float | None = None
        self._cached_factor: Any = None
        self._cached_solver_valid: bool = False

    def _init_grid(self, grid: Sequence[float] | np.ndarray) -> None:
        """Validate and set up the 1D spatial grid."""
        r = np.asarray(grid, dtype=float)
        if r.ndim != 1:
            raise ValueError(f"Grid must be a 1D array, got shape {r.shape}.")
        if len(r) < 2:
            raise ValueError("Grid must contain at least 2 vertices.")
        if np.any(np.diff(r) <= 0.0):
            raise ValueError("Grid coordinates r must be strictly increasing.")
        if r[0] < 0.0:
            raise ValueError(
                f"Grid coordinates must be non-negative, got r[0] = {r[0]}."
            )

        self.r = r
        self.n_points = len(r)
        # Full grid retained so a growing-domain run (apply_growth with
        # grow_domain=True) can extend the active extent back out to it.
        self._r_full = r

    def _init_geometry(self, geometry: GeometryType | int | str) -> None:
        """Configure coordinate geometry factor alpha."""
        if isinstance(geometry, int):
            if geometry not in (0, 1, 2):
                raise ValueError(
                    f"Integer geometry must be 0, 1, or 2, got {geometry}."
                )
            self.alpha = geometry
            self.geometry = ["planar", "cylindrical", "spherical"][geometry]
        elif isinstance(geometry, str):
            g_str = geometry.strip().lower()
            if g_str in ("planar", "cartesian", "slab", "1d", "0"):
                self.alpha = 0
                self.geometry = "planar"
            elif g_str in ("cylindrical", "cylinder", "1"):
                self.alpha = 1
                self.geometry = "cylindrical"
            elif g_str in ("spherical", "sphere", "2"):
                self.alpha = 2
                self.geometry = "spherical"
            else:
                raise ValueError(
                    f"Unknown geometry '{geometry}'. Expected 'planar', 'cylindrical', or 'spherical'."
                )
        else:
            raise TypeError(f"Invalid geometry type: {type(geometry)}.")

    def _init_components(
        self,
        components: Sequence[str] | int,
        dependent_component: str | int | None,
    ) -> None:
        """Set up component names and identify independent vs dependent components."""
        if isinstance(components, int):
            if components < 1:
                raise ValueError(
                    f"Number of components must be >= 1, got {components}."
                )
            self.components = [f"C{i}" for i in range(components)]
        elif isinstance(components, (list, tuple)):
            if len(components) < 1:
                raise ValueError("Components sequence cannot be empty.")
            self.components = [str(c) for c in components]
        else:
            raise TypeError("components must be a list of strings or an integer.")

        self.n_components = len(self.components)

        if self.n_components == 1:
            # Single field / tracer diffusion
            self.dependent_component: str | None = None
            self.dependent_idx: int = -1
            self.independent_components = list(self.components)
            self.independent_indices = [0]
            self.n_independent = 1
        else:
            # Multicomponent diffusion with closure
            if dependent_component is None:
                dep_idx = self.n_components - 1
            elif isinstance(dependent_component, str):
                if dependent_component not in self.components:
                    raise ValueError(
                        f"dependent_component '{dependent_component}' not in {self.components}."
                    )
                dep_idx = self.components.index(dependent_component)
            elif isinstance(dependent_component, int):
                dep_idx = dependent_component % self.n_components
            else:
                raise TypeError(
                    "dependent_component must be a string or integer index."
                )

            self.dependent_idx = dep_idx
            self.dependent_component = self.components[dep_idx]
            self.independent_indices = [
                k for k in range(self.n_components) if k != dep_idx
            ]
            self.independent_components = [
                self.components[k] for k in self.independent_indices
            ]
            self.n_independent = len(self.independent_indices)

    def _init_measures(self) -> None:
        """Compute control volume cell measures V_k and interface transmissibilities T_{k+1/2}.

        Cell boundaries are defined at midpoints between vertices:
            r_{k+1/2} = (r_k + r_{k+1}) / 2 for k = 0, ..., M-2,
            r_{-1/2} = r_0,
            r_{M-1/2} = r_{M-1}.

        Control volume measure:
            V_k = int_{r_{k-1/2}}^{r_{k+1/2}} r^alpha dr = (r_{k+1/2}^{alpha+1} - r_{k-1/2}^{alpha+1}) / (alpha + 1).

        Interface transmissibility:
            T_{k+1/2} = r_{k+1/2}^alpha / (r_{k+1} - r_k).
        """
        M = self.n_points
        r = self.r
        alpha = self.alpha

        # Midpoint interfaces between vertices
        r_faces = 0.5 * (r[:-1] + r[1:])
        self.r_faces = r_faces

        # Control volume boundary positions: shape (M + 1,)
        r_bounds = np.empty(M + 1, dtype=float)
        r_bounds[0] = r[0]
        r_bounds[1:-1] = r_faces
        r_bounds[-1] = r[-1]
        self.r_bounds = r_bounds

        # Cell measures V_k
        p = alpha + 1
        self.cell_volumes = (r_bounds[1:] ** p - r_bounds[:-1] ** p) / float(p)

        # Interface transmissibility T_{k+1/2} = r_face^alpha / delta_r
        dr = np.diff(r)
        if alpha == 0:
            self.T_faces = 1.0 / dr
        else:
            self.T_faces = (r_faces**alpha) / dr

    def _set_extent(self, n: int) -> None:
        """Make the first ``n`` vertices the active domain.

        Used by :meth:`apply_growth` with ``grow_domain=True`` to reproduce the
        CZGM MATLAB driver, which runs its diffusion step on the sub-grid
        ``Rr(1:i)`` -- the grain grows, so vertices beyond the growth front do
        not exist and no flux crosses it.  Truncating the active extent gives
        exactly that: the rim vertex is then the *last* vertex and carries the
        half control volume a boundary vertex gets, whereas pinning it on the
        full grid would leave it an interior vertex with a full one.

        Only the active extent changes; the full grid in ``self._r_full`` is
        kept so the domain can grow again.  Any state in the hidden tail is
        discarded, which is safe because ``apply_growth`` writes the rim
        composition into every vertex it reveals.
        """
        n = int(n)
        if n < 2:
            raise ValueError(f"Active extent must be at least 2 vertices, got {n}.")
        if n > len(self._r_full):
            raise ValueError(
                f"Active extent {n} exceeds the full grid ({len(self._r_full)})."
            )
        if n == self.n_points:
            return

        old_n = self.n_points
        self.r = self._r_full[:n]
        self.n_points = n

        X_new = np.zeros((n, self.n_components), dtype=float)
        keep = min(old_n, n)
        X_new[:keep] = self._X[:keep]
        self._X = X_new

        self._init_measures()
        self._internal_bcs = {}
        self._cached_solver_valid = False

    def set_diffusivity(
        self,
        diffusivity: float | Sequence[Any] | np.ndarray | Callable[..., Any],
    ) -> None:
        """Set the diffusion matrix or callable diffusivity function.

        Parameters
        ----------
        diffusivity : float, array_like, or callable
            Diffusion coefficient(s):
            - (N-1) x (N-1) array: directly used as interdiffusion matrix
            - N x N array: reduced consistently using the dependent component
            - Scalar: for N=2 or single field
            - Callable: D(X) or D(r, X) returning matrix or scalar
        """
        self._cached_solver_valid = False
        if callable(diffusivity):
            self._diffusivity_callable = diffusivity
            self._is_constant_D = False
            self._constant_D_matrix = None
        else:
            self._diffusivity_callable = None
            self._is_constant_D = True
            self._constant_D_matrix = self._process_numeric_D(diffusivity)

    def _process_numeric_D(self, D_val: Any) -> np.ndarray:
        """Process and validate a numeric diffusion matrix."""
        P = self.n_independent
        N = self.n_components

        if np.isscalar(D_val):
            if P == 1:
                return np.array([[float(D_val)]], dtype=float)
            raise ValueError(
                f"Scalar diffusivity is only valid for 1 independent component (got P={P})."
            )

        D_arr = np.asarray(D_val, dtype=float)
        if D_arr.ndim == 0:
            if P == 1:
                return np.array([[float(D_arr)]], dtype=float)
            raise ValueError(
                f"Scalar diffusivity is only valid for 1 independent component (got P={P})."
            )

        if D_arr.shape == (P, P):
            return D_arr.copy()

        if D_arr.shape == (N, N) and N > 1:
            return reduce_diffusion_matrix(D_arr, dependent_idx=self.dependent_idx)

        raise ValueError(
            f"Invalid diffusivity shape {D_arr.shape}. "
            f"Expected ({P}, {P}) reduced matrix or ({N}, {N}) full matrix."
        )

    def _eval_D_at_state(
        self,
        r_val: float,
        X_full: np.ndarray,
    ) -> np.ndarray:
        """Evaluate diffusivity matrix at a given coordinate and composition.

        Supports callable diffusivities with a range of signatures:

        - ``f(X_dict)``, ``f(X_array)``                     composition-dependent
        - ``f(r, X_dict)``, ``f(r, X_array)``               radius + composition
        - ``f(t, X_dict)``, ``f(t, X_array)``               time + composition
        - ``f(t)``                                          time-only
        - ``f(t, r, X_dict)``, ``f(t, r, X_array)``         full state

        The time ``t`` used is the model's current solution time
        (``self.time``).  Two-argument callables are disambiguated by the
        name of their first parameter ('t'/'time'/... treat time first; a
        radius-like name, or anything else, keeps the historical radius-first
        behaviour for backwards compatibility).
        """
        if self._is_constant_D:
            assert self._constant_D_matrix is not None
            return self._constant_D_matrix

        func = self._diffusivity_callable
        assert func is not None

        # Build dict of component names -> values
        comp_dict = {name: float(X_full[i]) for i, name in enumerate(self.components)}

        # Best-effort signature introspection; degrade gracefully for
        # builtins, numpy ufuncs, functools.partial, C callables, etc.
        try:
            params = list(inspect.signature(func).parameters.values())
            if any(
                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params
            ):
                first_name = None  # *args/**kwargs => unknown arity
            else:
                first_name = params[0].name if params else None
        except (ValueError, TypeError):
            first_name = None

        t_current = self.time

        # Candidate interpretations in evaluation order.
        # Radius-first is the historical default; time-first is selected
        # when the first argument is time-like, so D(t, X) callables are not
        # silently evaluated with r in the time slot.
        common = [
            ("f(X_dict)", (comp_dict,)),
            ("f(X_array)", (X_full,)),
        ]
        radius_candidates = [
            ("f(r, X_dict)", (r_val, comp_dict)),
            ("f(r, X_array)", (r_val, X_full)),
        ]
        time_candidates = [
            ("f(t)", (t_current,)),
            ("f(t, X_dict)", (t_current, comp_dict)),
            ("f(t, X_array)", (t_current, X_full)),
        ]
        full_candidates = [
            ("f(t, r, X_dict)", (t_current, r_val, comp_dict)),
            ("f(t, r, X_array)", (t_current, r_val, X_full)),
        ]

        if first_name is not None and first_name.lower() in _TIME_PARAM_NAMES:
            order = [
                *common,
                *time_candidates,
                *radius_candidates,
                *full_candidates,
            ]
        else:
            # Default / radius-first ordering (backwards compatible)
            order = [
                *common,
                *radius_candidates,
                *time_candidates,
                *full_candidates,
                ("f(r)", (r_val,)),  # 1-arg radius-only fallback
            ]

        last_err: Exception | None = None
        for label, args in order:
            try:
                res = func(*args)
            except TypeError as err:
                # Wrong arity for this interpretation -> try the next one.
                last_err = err
                continue
            except (ValueError, KeyError, IndexError) as err:
                # Dict-key / shape / domain mismatch: try the next one.
                last_err = err
                continue
            return self._process_numeric_D(res)

        raise TypeError(
            f"Diffusivity callable {getattr(func, '__name__', func)!r} could not be "
            f"evaluated with any supported signature. Tried: "
            f"{', '.join(label for label, _ in order)}. Last error: {last_err}"
        ) from last_err

    def set_bc(
        self,
        boundary: Literal["left", "right", "inner", "outer", "core", "rim"],
        bc_type: Literal["dirichlet", "neumann"]
        | BoundaryCondition
        | tuple[str, Any]
        | dict[str, Any],
        value: float | dict[str, float] | np.ndarray | None = None,
    ) -> None:
        """Set boundary condition for left (inner) or right (outer) boundary.

        Parameters
        ----------
        boundary : {'left', 'right', 'inner', 'outer', 'core', 'rim'}
            Boundary to configure.
        bc_type : str, BoundaryCondition, tuple, or dict
            Boundary condition type ('dirichlet' or 'neumann') or full spec.
        value : float, dict, or np.ndarray, optional
            Prescribed boundary value. If bc_type is a BoundaryCondition, tuple,
            or dict containing 'value', this parameter is ignored.
        """
        b_norm = boundary.strip().lower()
        if b_norm in ("left", "inner", "core", "center"):
            side = "left"
        elif b_norm in ("right", "outer", "rim"):
            side = "right"
        else:
            raise ValueError(
                f"Unknown boundary '{boundary}'. Expected 'left' or 'right'."
            )

        if isinstance(bc_type, BoundaryCondition):
            bc = copy.deepcopy(bc_type)
        elif isinstance(bc_type, tuple):
            bc = BoundaryCondition(bc_type=bc_type[0], value=bc_type[1])
        elif isinstance(bc_type, dict):
            b_t = bc_type.get("type", bc_type.get("bc_type", "neumann"))
            b_v = bc_type.get("value", 0.0)
            bc = BoundaryCondition(bc_type=b_t, value=b_v)
        elif isinstance(bc_type, str):
            val = 0.0 if value is None else value
            bc = BoundaryCondition(bc_type=bc_type, value=val)
        else:
            raise TypeError(f"Invalid bc_type format: {type(bc_type)}.")

        self._bcs[side] = bc
        self._cached_solver_valid = False

    def _resolve_bc_values(
        self,
        bc: BoundaryCondition,
    ) -> np.ndarray:
        """Extract array of boundary values for the independent components."""
        P = self.n_independent
        val = bc.value

        if np.isscalar(val):
            return np.full(P, float(val), dtype=float)

        if isinstance(val, dict):
            res = np.zeros(P, dtype=float)
            for j_indep, c_idx in enumerate(self.independent_indices):
                name = self.components[c_idx]
                if name in val:
                    res[j_indep] = float(val[name])
                else:
                    raise KeyError(
                        f"Boundary condition dict is missing component '{name}'."
                    )
            return res

        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:
            if len(arr) == P:
                return arr.copy()
            if len(arr) == self.n_components:
                # Extract independent indices
                return arr[self.independent_indices].copy()

        raise ValueError(
            f"Cannot resolve BC value {val} to {P} independent components."
        )

    def set_internal_bcs(
        self,
        positions: int | float | Sequence[float | int] | np.ndarray,
        value: float | dict[str, float] | np.ndarray | None = None,
    ) -> None:
        """Replace all internal point Dirichlet constraints ("pins") on the grid.

        Pinned vertices are held at the prescribed composition for every
        independent component for all subsequent solves, exactly like a
        Dirichlet boundary condition but at an interior vertex. This is the 1D
        finite-volume equivalent of pinning individual mesh-boundary labels
        inside the domain (e.g. the not-yet-grown shell boundaries of the 2D
        annulus growth models): the already-grown active region between the
        pins can still diffuse while the pinned vertices remain fixed.

        Calling this method replaces any previously set internal pins; use
        ``clear_internal_bcs`` to remove them again. Pins at the same vertex as
        a left/right boundary condition take precedence over the boundary rows
        (they are applied last during assembly).

        Parameters
        ----------
        positions : int, float, array_like, or np.ndarray
            Vertices to pin, given as integer index(s), radius/radii (matched
            to the nearest grid vertex), or a boolean mask over the grid (see
            ``_resolve_vertex_positions``). Passing an empty selection removes
            all pins.
        value : float, dict, or array_like
            Prescribed value for the independent components at the pinned
            vertices (same accepted forms as ``set_bc`` values): a scalar
            applied to every independent component, a dict mapping component
            names to values, an (N-1)-vector of independent-component values,
            or an N-vector (independent entries extracted).

        Examples
        --------
        Pin the vertex nearest r=0.6 to A=0.9 in a binary system::

            model.set_internal_bcs(0.6, {"A": 0.9})

        Pin every vertex at/after the growth front to a growth composition::

            model.set_internal_bcs(model.r >= r_growth, growth_comp)
        """
        if value is None:
            raise TypeError("set_internal_bcs requires a ``value`` argument.")
        bc = BoundaryCondition(bc_type="dirichlet", value=value)
        vals = self._resolve_bc_values(bc)  # (P,) independent-component values
        idx = _resolve_vertex_positions(positions, self.r)

        pins: dict[int, dict[int, float]] = {}
        for k in idx:
            pins[int(k)] = {j: float(vals[j]) for j in range(self.n_independent)}
        self._internal_bcs = pins
        self._cached_solver_valid = False

    def clear_internal_bcs(self) -> None:
        """Remove all internal point Dirichlet constraints set with ``set_internal_bcs``.

        After clearing, every vertex that was pinned returns to the ordinary
        finite-volume solve (subject only to the left/right boundary
        conditions).
        """
        self._internal_bcs = {}
        self._cached_solver_valid = False

    def set_initial_composition(
        self,
        composition: dict[str, Any] | np.ndarray | Callable[[np.ndarray], Any] | float,
    ) -> None:
        """Set initial composition on the grid and update the dependent component.

        Parameters
        ----------
        composition : dict, array_like, callable, or float
            Initial composition field(s):
            - Dict: {comp_name: scalar, array, or callable f(r)}
            - 2D array: shape (M, N) or (M, N-1)
            - 1D array: for N=1 or binary N=2
            - Callable: f(r) returning dict or array
            - Float: uniform value across grid (for single/binary)
        """
        M = self.n_points
        N = self.n_components
        r = self.r

        if callable(composition):
            val = composition(r)
            self.set_initial_composition(val)
            return

        if isinstance(composition, dict):
            # Fill specified components
            new_X = np.zeros((M, N), dtype=float)
            specified = set()
            for comp_name, data in composition.items():
                if comp_name not in self.components:
                    raise KeyError(
                        f"Component '{comp_name}' in initial_composition not in model components: {self.components}."
                    )
                idx = self.components.index(comp_name)
                specified.add(idx)

                if callable(data):
                    arr = np.asarray([float(data(rk)) for rk in r], dtype=float)
                elif np.isscalar(data):
                    arr = np.full(M, float(data), dtype=float)
                else:
                    arr = np.asarray(data, dtype=float)
                    if arr.shape != (M,):
                        raise ValueError(
                            f"Initial array for '{comp_name}' must have shape ({M},), got {arr.shape}."
                        )
                new_X[:, idx] = arr

            if N > 1:
                dep_idx = self.dependent_idx
                if dep_idx not in specified:
                    # Automatically compute from closure
                    indep_sum = np.sum(
                        [new_X[:, k] for k in self.independent_indices], axis=0
                    )
                    new_X[:, dep_idx] = 1.0 - indep_sum
                else:
                    # Verify closure
                    tot = np.sum(new_X, axis=1)
                    if not np.allclose(tot, 1.0, atol=1e-4):
                        warnings.warn(
                            f"Initial compositions do not sum to 1.0 (min={tot.min():.4f}, max={tot.max():.4f}). "
                            "Dependent component will enforce closure.",
                            UserWarning,
                        )
                        indep_sum = np.sum(
                            [new_X[:, k] for k in self.independent_indices], axis=0
                        )
                        new_X[:, dep_idx] = 1.0 - indep_sum
            self._X = new_X
            return

        # Array or scalar
        arr = np.asarray(composition, dtype=float)
        if arr.ndim == 0:
            if N == 1:
                self._X = np.full((M, 1), float(arr), dtype=float)
                return
            if N == 2:
                self._X[:, self.independent_indices[0]] = float(arr)
                self._X[:, self.dependent_idx] = 1.0 - float(arr)
                return
            raise ValueError(
                f"Scalar initial composition only valid for N=1 or N=2, got N={N}."
            )

        if arr.ndim == 1 and arr.shape == (M,):
            if N == 1:
                self._X[:, 0] = arr
                return
            if N == 2:
                self._X[:, self.independent_indices[0]] = arr
                self._X[:, self.dependent_idx] = 1.0 - arr
                return
            raise ValueError(
                f"1D array initial composition of length {M} requires N=1 or N=2, got N={N}."
            )

        if arr.ndim == 2:
            if arr.shape == (M, N):
                self._X = arr.copy()
                if N > 1:
                    indep_sum = np.sum(
                        [self._X[:, k] for k in self.independent_indices], axis=0
                    )
                    self._X[:, self.dependent_idx] = 1.0 - indep_sum
                return
            if arr.shape == (M, self.n_independent):
                for j_indep, c_idx in enumerate(self.independent_indices):
                    self._X[:, c_idx] = arr[:, j_indep]
                if N > 1:
                    indep_sum = np.sum(arr, axis=1)
                    self._X[:, self.dependent_idx] = 1.0 - indep_sum
                return

        raise ValueError(
            f"Incompatible shape for initial composition: {arr.shape}. "
            f"Expected ({M}, {N}) or ({M}, {self.n_independent})."
        )

    def _resolve_rim_composition(
        self,
        composition: dict[str, Any] | np.ndarray | float,
    ) -> np.ndarray:
        """Resolve a rim-composition specification to a full (N,) vector.

        Accepts dict {name: value}, a full N-vector, an (N-1)-vector of
        independent components, or a scalar for single/binary systems.
        """
        M = self.n_points
        N = self.n_components
        P = self.n_independent

        if isinstance(composition, dict):
            missing = [
                c for c in self.independent_components if c not in composition
            ]
            if missing:
                raise KeyError(
                    f"Rim composition dict is missing independent component(s) "
                    f"{missing}. The dependent component may be omitted and is "
                    "inferred by closure."
                )
            X = np.zeros(N, dtype=float)
            for name, val in composition.items():
                if name not in self.components:
                    raise KeyError(
                        f"Component '{name}' not in model components: {self.components}."
                    )
                X[self.components.index(name)] = float(val)
            if N > 1:
                self._enforce_closure_on_vec(X)
            return X

        arr = np.asarray(composition, dtype=float)
        if arr.ndim == 0:
            if N == 1:
                return np.array([float(arr)])
            if N == 2:
                return np.array([float(arr), 1.0 - float(arr)])
            raise ValueError(
                f"Scalar rim composition only valid for N=1 or N=2, got N={N}."
            )

        if arr.ndim > 1:
            raise ValueError(
                f"Rim composition must be a scalar, dict, or 1D array, got shape {arr.shape}."
            )

        if arr.shape == (M,):
            # A full profile supplied: use the value at the rim (outer) edge
            arr = arr[-1:]
        if arr.shape == (N,):
            X = arr.copy()
            if N > 1:
                self._enforce_closure_on_vec(X)
            return X
        if arr.shape == (P,) and N > 1:
            X = np.zeros(N, dtype=float)
            for j_indep, c_idx in enumerate(self.independent_indices):
                X[c_idx] = arr[j_indep]
            self._enforce_closure_on_vec(X)
            return X
        if arr.shape == (1,) and N == 2:
            return np.array([arr[0], 1.0 - arr[0]])

        raise ValueError(
            f"Cannot interpret rim composition with shape {arr.shape}; "
            f"expected ({N},) or ({P},)."
        )

    def _enforce_closure_on_vec(self, X: np.ndarray) -> None:
        """Enforce sum_k X_k = 1 in-place on a single composition vector."""
        if self.n_components > 1:
            indep_sum = float(
                np.sum([X[k] for k in self.independent_indices])
            )
            X[self.dependent_idx] = 1.0 - indep_sum

    def apply_growth(
        self,
        new_radius: float,
        composition: dict[str, Any] | np.ndarray | float,
        set_rim_bc: bool = True,
        pin_front: bool = True,
        grow_domain: bool = False,
    ) -> np.ndarray:
        """Apply a concentric growth step with a new moving-rim composition.

        This emulates grain growth with a moving boundary on the 1D grid used
        in the CZGM benchmark
        (``Tutorials/Diffusion/CZGM_garnet_model_annulus.ipynb``): every grid
        vertex with ``r > new_radius`` is overwritten with the supplied rim
        composition (the newly added shell), the dependent component is
        recomputed for closure, and -- by default -- the growth front itself
        is pinned as a Dirichlet condition, holding the exposed rim at the
        growth composition until the next growth step.

        Parameters
        ----------
        new_radius : float
            New outer radius of the growing grain (same units as the grid).
        composition : dict, array_like, or float
            Rim composition for the newly grown material. See
            ``_resolve_rim_composition`` for accepted forms.
        set_rim_bc : bool, optional
            If True (default), pin the growth front as a Dirichlet condition
            on the new rim composition.
        pin_front : bool, optional
            If True (default), also freeze every vertex at or beyond
            ``new_radius`` via :meth:`set_internal_bcs`, so the grain
            effectively *ends* at the front.  Ignored when ``grow_domain`` is
            True (the domain genuinely ends there, so there is nothing to
            pin).  Set ``pin_front=False`` to recover plain stamping.
        grow_domain : bool, optional
            If True, **truncate the active domain** to ``new_radius`` so the
            grain really ends at the front -- the faithful reproduction of the
            CZGM MATLAB driver, which runs its diffusion step on the sub-grid
            ``Rr(1:i)``.

            This is the exact version of what ``pin_front`` approximates.  The
            difference is not cosmetic: on a truncated grid the rim vertex is
            the *last* vertex and gets the half control volume a boundary
            vertex has, whereas pinning it on the full grid leaves it an
            interior vertex with a full control volume.  Measured on the CZGM
            planar benchmark against ``combined_C_final_cart.csv``, the centre
            relaxes to 82% of the reference with ``pin_front`` and to ~100%
            with ``grow_domain``.

            The full grid is retained internally, so successive calls grow the
            active extent back out along it.  Vertices revealed by a growth
            step are initialised to the supplied rim composition.

        Returns
        -------
        np.ndarray
            Updated full composition array of shape (n_points, n_components).
        """
        new_radius = float(new_radius)
        if new_radius <= self._r_full[0]:
            raise ValueError(
                f"new_radius must be greater than r[0] = {self._r_full[0]}, "
                f"got {new_radius}."
            )
        if new_radius > self._r_full[-1]:
            raise ValueError(
                f"new_radius {new_radius} exceeds the max grid radius "
                f"{self._r_full[-1]}."
            )

        rim = self._resolve_rim_composition(composition)

        if grow_domain:
            # The grain ends at the front: make the first front_idx+1 vertices
            # the active domain, with the front vertex as the rim.  Locate the
            # front by index -- np.linspace rarely lands exactly on new_radius.
            front_idx = int(np.argmin(np.abs(self._r_full - new_radius)))
            self._set_extent(front_idx + 1)
            self._X[front_idx:, :] = rim[None, :]
            self._update_dependent_component()
            if set_rim_bc:
                self.set_bc(
                    "right",
                    "dirichlet",
                    {name: float(rim[i]) for i, name in enumerate(self.components)},
                )
            return self._X.copy()

        mask = self.r > new_radius
        if not np.any(mask):
            # A boundary-reset growth at the maximum grid radius (e.g. the
            # final CZGM rim step) is represented by re-stamping the edge
            # vertex itself.
            if np.isclose(new_radius, self.r[-1]):
                mask = self.r >= new_radius
            else:
                raise ValueError(
                    f"No grid vertices beyond new_radius={new_radius} "
                    f"(max r={self.r[-1]})."
                )

        self._X[mask, :] = rim[None, :]
        self._update_dependent_component()

        if set_rim_bc:
            rim_dict = {
                name: float(rim[i]) for i, name in enumerate(self.components)
            }
            self.set_bc("right", "dirichlet", rim_dict)
            if pin_front:
                # Freeze the front and everything beyond it: the grid now
                # behaves as though it ended at new_radius.  The vertex exactly
                # at the front is the rim, so it takes the rim composition too
                # -- otherwise the stored state would disagree with the pin
                # until the next solve.  Locate the front by index, not by
                # comparing radii: ``np.linspace`` rarely lands exactly on
                # ``new_radius``, so ``r >= new_radius`` silently drops the
                # front vertex itself.
                front_idx = int(np.argmin(np.abs(self.r - new_radius)))
                pinned = np.zeros(self.n_points, dtype=bool)
                pinned[front_idx:] = True
                self._X[pinned, :] = rim[None, :]
                self._update_dependent_component()
                self.set_internal_bcs(pinned, rim_dict)

        return self._X.copy()

    def run_to_time(
        self,
        target_time: float,
        dt: float,
        save_interval: int | None = None,
        callback: Callable[[float, np.ndarray], None] | None = None,
    ) -> SimulationResult:
        """Advance the simulation from the current time to an absolute target time.

        Convenience for segmented loads (e.g. prograde growth + retrograde
        diffusion) where each segment ends at a prescribed ``current_time``.
        Steps are taken with ``min(dt, remaining)`` exactly as in
        ``run_simulation``, so the final state lands on ``target_time``
        (up to floating-point precision).

        Parameters
        ----------
        target_time : float
            Absolute solution time to reach; must be greater than ``self.time``.
        dt : float
            Maximum time step size.
        save_interval : int, optional
            Record the profile every ``save_interval`` steps (the initial and
            final states are always recorded).
        callback : callable, optional
            Function callback(time, composition) called after each recorded step.

        Returns
        -------
        SimulationResult
            Object containing the time series recorded while advancing.
        """
        target_time = float(target_time)
        if target_time <= self.time:
            raise ValueError(
                f"target_time ({target_time}) must be greater than current time "
                f"({self.time})."
            )
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}.")

        recorded_times: list[float] = [self.time]
        recorded_compositions: list[np.ndarray] = [self._X.copy()]
        if callback is not None:
            callback(self.time, self._X.copy())

        step_idx = 0
        while self.time < target_time - 1e-14 * abs(target_time):
            step_idx += 1
            step_dt = min(dt, target_time - self.time)
            self.step(step_dt)
            if (save_interval is not None and step_idx % save_interval == 0) or (
                self.time >= target_time - 1e-14 * abs(target_time)
            ):
                recorded_times.append(self.time)
                recorded_compositions.append(self._X.copy())
                if callback is not None:
                    callback(self.time, self._X.copy())

        return SimulationResult(
            r=self.r.copy(),
            times=np.array(recorded_times, dtype=float),
            compositions=np.array(recorded_compositions, dtype=float),
            components=list(self.components),
            geometry=self.geometry,
            alpha=self.alpha,
        )

    def _update_dependent_component(self) -> None:
        """Update dependent component mole fractions via closure relation sum_i X_i = 1."""
        if self.n_components > 1:
            indep_sum = np.sum(
                [self._X[:, k] for k in self.independent_indices], axis=0
            )
            self._X[:, self.dependent_idx] = 1.0 - indep_sum

    def get_composition(
        self,
        component: str | int | None = None,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Get current concentration profile(s).

        Parameters
        ----------
        component : str, int, or None, optional
            Component name or index. If None, returns all components.
        as_dict : bool, optional
            If True, returns a dictionary mapping component names to arrays.

        Returns
        -------
        np.ndarray or dict of str -> np.ndarray
            Current composition profile(s).
        """
        if as_dict:
            return {
                name: self._X[:, i].copy() for i, name in enumerate(self.components)
            }

        if component is None:
            return self._X.copy()

        if isinstance(component, str):
            if component not in self.components:
                raise KeyError(f"Component '{component}' not in {self.components}.")
            idx = self.components.index(component)
        else:
            idx = int(component)

        return self._X[:, idx].copy()

    def total_mass(
        self,
        component: str | int | None = None,
    ) -> float | dict[str, float] | np.ndarray:
        """Calculate integrated species measure sum_k V_k X_{i,k}.

        For spherical coordinates, this is the integral of r^2 X dr (total moles
        per unit solid angle 4*pi). This quantity is strictly conserved under
        zero-flux Neumann boundary conditions.

        Parameters
        ----------
        component : str, int, or None, optional
            Component name or index. If None, returns dict or array for all components.

        Returns
        -------
        float, dict, or np.ndarray
            Integrated mass/amount for requested component(s).
        """
        masses = np.sum(self.cell_volumes[:, None] * self._X, axis=0)

        if component is None:
            return {name: float(masses[i]) for i, name in enumerate(self.components)}

        if isinstance(component, str):
            if component not in self.components:
                raise KeyError(f"Component '{component}' not in {self.components}.")
            idx = self.components.index(component)
        else:
            idx = int(component)

        return float(masses[idx])

    def _assemble_system(
        self,
        dt: float,
        X_eval: np.ndarray,
    ) -> tuple[sp.csc_matrix, np.ndarray]:
        """Assemble the linear system A * X^{n+1} = b for Backward Euler time stepping.

        Unknown ordering:
            Index for node k (0 <= k < M) and independent component i (0 <= i < P) is:
                row = k * P + i.
        """
        M = self.n_points
        P = self.n_independent
        V = self.cell_volumes
        T = self.T_faces

        # Extract independent component state at evaluation point
        X_indep_current = np.column_stack(
            [self._X[:, k] for k in self.independent_indices]
        )

        # Coordinate lists for building sparse matrix
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        b = np.zeros(M * P, dtype=float)

        # 1. Diagonal accumulation terms and RHS
        for k in range(M):
            Vk = V[k]
            for i in range(P):
                row = k * P + i
                rows.append(row)
                cols.append(row)
                data.append(Vk)
                b[row] = Vk * X_indep_current[k, i]

        # 2. Interface flux terms
        for k in range(M - 1):
            Tk = T[k]
            rf = self.r_faces[k]
            Xf = 0.5 * (X_eval[k, :] + X_eval[k + 1, :])
            D_face = self._eval_D_at_state(rf, Xf)

            Mflux = dt * Tk * D_face  # shape (P, P)
            for i in range(P):
                for j in range(P):
                    val = float(Mflux[i, j])
                    if val != 0.0:
                        rk = k * P + i
                        rkp1 = (k + 1) * P + i
                        cj = k * P + j
                        cjp1 = (k + 1) * P + j

                        rows.extend([rk, rk, rkp1, rkp1])
                        cols.extend([cj, cjp1, cj, cjp1])
                        data.extend([val, -val, -val, val])

        # Convert to LIL matrix to apply boundary conditions directly
        A = sp.coo_matrix((data, (rows, cols)), shape=(M * P, M * P)).tolil()

        # 3. Apply Left Boundary Condition (node k=0)
        left_bc = self._bcs["left"]
        left_vals = self._resolve_bc_values(left_bc)

        if self.centre == "mirror":
            # CZGM MATLAB treatment of the grain centre
            # (``make_diffusion.m``: ``C(1,:) = C(2,:)`` then ``A(1,:) = C(1,:)``):
            # vertex 0 is a *passive mirror* of vertex 1 -- it is held at vertex
            # 1's pre-update value and never evolves.  Implemented as an
            # identity row with the right-hand side set to vertex 1's current
            # value, which is exactly that.  The flux between vertices 0 and 1
            # then evaluates as ``D * (X1_prev - X1_new)``, i.e. the one-sided
            # Laplacian MATLAB's stencil produces.
            #
            # This is NOT a zero-gradient symmetry plane.  A true zero-flux
            # centre lets vertex 0 diffuse; this one does not.  Measured on the
            # CZGM planar benchmark (``combined_C_final_cart.csv``) the centre
            # relaxes to +0.0375 with ``mirror`` (matching the reference to
            # ~1e-4) but only +0.0314 with ``zero_flux``.  ``zero_flux`` is the
            # physically defensible condition for a symmetry plane; ``mirror``
            # is what the benchmark was produced with.
            for i in range(P):
                row = 0 * P + i
                A[row, :] = 0.0
                A[row, row] = 1.0
                b[row] = float(X_eval[1, self.independent_indices[i]])
        elif left_bc.bc_type == "dirichlet":
            for i in range(P):
                row = 0 * P + i
                A[row, :] = 0.0
                A[row, row] = 1.0
                b[row] = left_vals[i]
        elif left_bc.bc_type == "neumann" and np.any(left_vals != 0.0):
            # Inward flow from left: if r[0] > 0 and non-zero gradient
            r0 = self.r[0]
            D0 = self._eval_D_at_state(r0, X_eval[0, :])
            # Flux into domain from left boundary
            area0 = r0**self.alpha if self.alpha > 0 or r0 > 0 else 1.0
            # flow = - area0 * D * gradX
            flow_in_left = -area0 * (D0 @ left_vals)
            for i in range(P):
                row = 0 * P + i
                b[row] += dt * flow_in_left[i]

        # 4. Apply Right Boundary Condition (node k=M-1)
        right_bc = self._bcs["right"]
        right_vals = self._resolve_bc_values(right_bc)

        if right_bc.bc_type == "dirichlet":
            for i in range(P):
                row = (M - 1) * P + i
                A[row, :] = 0.0
                A[row, row] = 1.0
                b[row] = right_vals[i]
        elif right_bc.bc_type == "neumann" and np.any(right_vals != 0.0):
            # Inward flow from right: if non-zero gradient
            rM = self.r[-1]
            DM = self._eval_D_at_state(rM, X_eval[-1, :])
            areaM = rM**self.alpha
            flow_in_right = areaM * (DM @ right_vals)
            for i in range(P):
                row = (M - 1) * P + i
                b[row] += dt * flow_in_right[i]

        # 5. Apply internal point Dirichlet constraints (pins): each pinned
        # vertex is held at the prescribed value for every independent
        # component, exactly like a Dirichlet boundary row but at an interior
        # vertex. Applied last so pins on the left/right boundary vertices take
        # precedence over the boundary rows.
        for k, compvals in self._internal_bcs.items():
            for i, val in compvals.items():
                row = int(k) * P + int(i)
                A[row, :] = 0.0
                A[row, row] = 1.0
                b[row] = val

        return A.tocsc(), b

    def step(self, dt: float) -> np.ndarray:
        """Advance the simulation forward in time by dt using Backward Euler.

        Parameters
        ----------
        dt : float
            Time step size (must be > 0).

        Returns
        -------
        np.ndarray
            Updated full composition array of shape (n_points, n_components).
        """
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}.")

        M = self.n_points
        P = self.n_independent

        if self._is_constant_D:
            # Constant D: assemble system and use pre-factorization if dt matches
            if not self._cached_solver_valid or self._cached_dt != dt:
                A, b = self._assemble_system(dt, self._X)
                self._cached_factor = spla.factorized(A)
                self._cached_dt = dt
                self._cached_solver_valid = True
            else:
                # b depends on current state self._X
                _, b = self._assemble_system(dt, self._X)

            x_new = self._cached_factor(b)
            X_new_indep = x_new.reshape((M, P))

            # Store updated independent fields
            for j_indep, c_idx in enumerate(self.independent_indices):
                self._X[:, c_idx] = X_new_indep[:, j_indep]

        else:
            # Composition-dependent / nonlinear D: Picard iteration
            X_iter = self._X.copy()
            converged = False

            for it in range(self.picard_max_iter):
                A, b = self._assemble_system(dt, X_iter)
                x_new = spla.spsolve(A, b)
                X_new_indep = x_new.reshape((M, P))

                # Update temporary full state for next iteration
                X_next = X_iter.copy()
                for j_indep, c_idx in enumerate(self.independent_indices):
                    X_next[:, c_idx] = X_new_indep[:, j_indep]
                if self.n_components > 1:
                    indep_sum = np.sum(
                        [X_next[:, k] for k in self.independent_indices], axis=0
                    )
                    X_next[:, self.dependent_idx] = 1.0 - indep_sum

                err = np.max(np.abs(X_next - X_iter))
                X_iter = X_next

                if err < self.picard_tol:
                    converged = True
                    break

            if not converged:
                warnings.warn(
                    f"Picard iteration did not converge within {self.picard_max_iter} iterations "
                    f"(residual = {err:.2e}, tol = {self.picard_tol:.2e}).",
                    UserWarning,
                )

            self._X = X_iter

        # Always update dependent component via closure
        self._update_dependent_component()
        self.time += dt
        return self._X.copy()

    def run_simulation(
        self,
        dt: float,
        total_time: float | None = None,
        n_steps: int | None = None,
        save_interval: int | None = None,
        output_times: Sequence[float] | None = None,
        callback: Callable[[float, np.ndarray], None] | None = None,
    ) -> SimulationResult:
        """Run the diffusion simulation through time and record history.

        Parameters
        ----------
        dt : float
            Target time step size.
        total_time : float, optional
            Total simulation time duration. Either total_time or n_steps must be given.
        n_steps : int, optional
            Total number of steps to run.
        save_interval : int, optional
            Save profiles every `save_interval` steps.
        output_times : sequence of float, optional
            Specific simulation times at which to record the composition profile.
            The time-stepper will adjust time steps to hit each output time precisely.
        callback : callable, optional
            Function callback(time, composition) called after each recorded step.

        Returns
        -------
        SimulationResult
            Object containing time series of composition profiles.
        """
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}.")

        recorded_times: list[float] = [self.time]
        recorded_compositions: list[np.ndarray] = [self._X.copy()]

        if callback is not None:
            callback(self.time, self._X.copy())

        if output_times is not None:
            # Output times driven simulation
            target_times = sorted([float(t) for t in output_times if t > self.time])
            for t_target in target_times:
                while self.time < t_target - 1e-14 * abs(t_target):
                    step_dt = min(dt, t_target - self.time)
                    self.step(step_dt)

                recorded_times.append(self.time)
                recorded_compositions.append(self._X.copy())
                if callback is not None:
                    callback(self.time, self._X.copy())

        else:
            # Fixed total_time or n_steps
            if total_time is not None:
                if total_time <= 0.0:
                    raise ValueError(f"total_time must be positive, got {total_time}.")
                steps_to_run = int(np.ceil(total_time / dt))
            elif n_steps is not None:
                if n_steps <= 0:
                    raise ValueError(f"n_steps must be positive, got {n_steps}.")
                steps_to_run = n_steps
            else:
                raise ValueError(
                    "Must specify either total_time, n_steps, or output_times."
                )

            for step_idx in range(1, steps_to_run + 1):
                # Adjust final dt if total_time was specified
                if total_time is not None and step_idx == steps_to_run:
                    actual_dt = total_time - (self.time - recorded_times[0])
                    if actual_dt <= 1e-14 * total_time:
                        actual_dt = dt
                else:
                    actual_dt = dt

                self.step(actual_dt)

                is_saved = (
                    save_interval is not None and (step_idx % save_interval == 0)
                ) or (step_idx == steps_to_run)

                if is_saved:
                    recorded_times.append(self.time)
                    recorded_compositions.append(self._X.copy())
                    if callback is not None:
                        callback(self.time, self._X.copy())

        compositions_arr = np.array(recorded_compositions, dtype=float)
        times_arr = np.array(recorded_times, dtype=float)

        return SimulationResult(
            r=self.r.copy(),
            times=times_arr,
            compositions=compositions_arr,
            components=list(self.components),
            geometry=self.geometry,
            alpha=self.alpha,
        )

    def reset(self, time: float = 0.0) -> None:
        """Reset current simulation clock."""
        self.time = float(time)
        self._cached_solver_valid = False

    def copy(self) -> MulticomponentDiffusion1D:
        """Create a deep copy of the diffusion model."""
        return copy.deepcopy(self)
