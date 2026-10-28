import os
import underworld3 as uw
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import functools
import re

from underworld3.systems import Poisson 

from .utilities import _adams_moulton_flux
from underworld3.utilities._api_tools import Template


def _print_header(title):
    print(f"\n# {title}")


def _mesh_summary(mesh):
    mesh_type = type(mesh).__name__
    dim = getattr(mesh, "dim", "?")
    min_radius = "?"
    try:
        min_radius = f"{mesh.get_min_radius():.6g}"
    except Exception:
        pass
    return f"{mesh_type} (dim={dim}, min_radius={min_radius})"


def _safe_symbol_text(value):
    try:
        return str(value)
    except Exception:
        return "<unavailable>"


def _view_in_notebook():
    return bool(getattr(uw, "is_notebook", False))


def _view_in_vscode():
    return bool(os.environ.get("VSCODE_PID") or os.environ.get("TERM_PROGRAM") == "vscode")


def _latex_of(value):
    try:
        return value._repr_latex_()
    except Exception:
        try:
            return sp.latex(sp.sympify(value))
        except Exception:
            return _safe_symbol_text(value)


def _view_markdown(text):
    if _view_in_notebook():
        from IPython.display import Markdown, display

        display(Markdown(text))
    else:
        print(text)


def _view_latex(text):
    if _view_in_notebook() and not _view_in_vscode():
        from IPython.display import Latex, display

        display(Latex(text))
    else:
        print(text)


def _view_symbol(label, value):
    if _view_in_notebook() and not _view_in_vscode():
        _view_latex(rf"$\quad {label} = {_latex_of(value)}$")
    else:
        _view_markdown(f"- {label}: `{_safe_symbol_text(value)}`")


def _non_dimensionalise_symbolic_value(value):
    if isinstance(value, sp.Piecewise):
        pairs = []
        for expr_cond in value.args:
            branch_value = expr_cond.args[0]
            branch_condition = expr_cond.args[1]
            pairs.append((_non_dimensionalise_symbolic_value(branch_value), branch_condition))
        return sp.Piecewise(*pairs)

    if hasattr(value, "sym"):
        try:
            sym_value = value.sym
            if sym_value is not value:
                return _non_dimensionalise_symbolic_value(sym_value)
        except Exception:
            pass

    if hasattr(value, "units"):
        return uw.scaling.non_dimensionalise(value)

    return value

class MulticomponentDiffusionModel:
    """Coupled diffusion for one or more components, solved in a single UW3 solve.

    Strong form, for the ``n_independent`` solved components ``i``::

        dx_i/dt = div( sum_j D_ij(x) grad(x_j) ) + S_i

    ``D_ij`` entries are supplied symbolically and resolved at runtime; ``S_i``
    is an optional source term.

    Component bookkeeping
    ---------------------
    With ``closure=True`` (the default, and the multicomponent convention) the
    LAST named component is implicit::

        x_n = 1 - sum_{i<n} x_i

    so ``n_independent = n_components - 1``.  With ``closure=False`` every
    named component is solved and there is no constraint linking them;
    ``n_independent = n_components``, and ``n_components=1`` is exactly the
    scalar diffusion problem :class:`DiffusionModel` solves.

    Why one solve
    -------------
    The previous implementation used ``n_independent`` separate
    :class:`underworld3.systems.Poisson` solvers.  Each owned only its own
    unknown, so the cross terms ``D_ij grad(x_j)`` entered that solver's
    residual as *data* rather than as unknowns -- the Jacobian block
    ``dF_i/d(grad x_j)`` was structurally zero, and the implicit coupling was
    approximated by a single Gauss-Seidel sweep per step.

    This class uses one :class:`underworld3.systems.SNES_MultiComponent` solve
    whose ``F1`` has shape ``(n_independent, mesh.dim)`` with row ``i`` equal to
    ``sum_j D_ij grad(x_j)``.  UW3 differentiates each row against EVERY
    component gradient (the ``G3`` block), so the full ``(n-1)x(n-1)`` coupling
    is implicit and exact.  Composition-dependent ``D`` is handled fully
    implicitly: ``consistent_jacobian`` (UW3 default) unwraps the expressions
    before differentiating, so no hand-frozen coefficient surrogate is needed.

    Time integration
    ----------------
    ``order`` sets the BDF order of the time derivative.  The diffusive flux is
    always evaluated at the new time level (fully implicit), so every order is
    A- and L-stable:

    * ``order=1`` -- backward Euler, ``O(dt)``.
    * ``order=2`` -- BDF2, ``O(dt^2)``:
      ``(3 x^{n+1} - 4 x^n + x^{n-1}) / (2 dt)``.
    * ``order=3`` -- BDF3, ``O(dt^3)``.

    The historical Adams-Moulton average of the flux history is deliberately
    NOT applied: it is only conditionally stable (amplification 1.25 at
    ``mu = kappa k^2 dt = 20``, reached by the highest finite-element modes),
    which makes the error stop converging with ``dt``.  BDF with an implicit
    flux is unconditionally stable at every order.

    Orders above 3 are not implemented (``NotImplementedError``).

    Composition symbols
    -------------------
    ``D`` entries may reference component abundances as ``xi_1 .. xi_n``:
    ``xi_k`` for ``k <= n_independent`` resolves to the solved field ``x_k``,
    and ``xi_n`` (the closure component) resolves to ``1 - sum(x_1..x_{n-1})``.
    These resolve to the *current* unknown, so composition-dependent ``D`` is
    treated fully implicitly and the step is genuinely nonlinear.

    Recommended workflow:
    1. Configure non-dimensional scaling in Underworld before setting values.
    2. Construct ``MulticomponentDiffusionModel(component_names, mesh, ...)``.
    3. ``set_diffusion_matrix(...)`` / ``set_diffusion_values(...)``.
    4. Set initial conditions on ``mesh_vars`` and add boundary conditions.
    5. ``run_simulation(...)``.
    """

    def __init__(self,
                 component_names,
                 mesh,
                 diffusion_matrix=None,
                 diffusion_values=None,
                 degree=2,
                 order=1,
                 closure=True):
        """
        Parameters:
        - component_names: list of str. With ``closure=True`` the last entry is
            implicit (``1 - sum`` of the others).
        - mesh: underworld3 mesh object
        - diffusion_matrix: symbolic (n_independent x n_independent) matrix.
            If None, a zero matrix is used until set.
        - diffusion_values: optional dict mapping diffusivity symbols to values
            (dimensional Pint or non-dimensional float).
        - degree: FE basis degree (default 2)
        - order: time integration order, 1-3 (default 1 = backward Euler)
        - closure: if True the last component is ``1 - sum`` of the rest.
        """
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if order > 3:
            raise NotImplementedError(
                "Adams-Moulton flux is implemented for order <= 3; "
                f"got order={order}"
            )

        self.component_names = list(component_names)
        self.n_components = len(self.component_names)
        self.closure = bool(closure)
        self.n_independent = (
            self.n_components - 1 if self.closure else self.n_components
        )
        if self.n_independent < 1:
            raise ValueError(
                "Need at least one solved component: with closure=True pass "
                ">= 2 component_names, with closure=False pass >= 1."
            )

        self.degree = degree
        self.order = order
        self._ddt_order = max(1, order)
        self.mesh = mesh
        # Frozen-tangent warm-up steps before Newton (see
        # _MonolithicCoupledSolver.step): -1 = none (default), 0 = automatic,
        # N = up to N.  Measured 1:1 with Newton on this operator, so it is off
        # by default; set 0 to restore UW3's automatic warm-up.
        self.picard = -1
        self.current_time = 0.0
        self.step = 0

        # Public per-component variables: one MeshVariable per named component,
        # including the closure component.  These are what callers read and
        # write (initial conditions, diagnostics, output), matching the API of
        # the previous implementation.
        #
        # Reject collisions before constructing anything: UW3 would silently
        # reuse an existing variable of the same name.
        base = self.component_names[0]
        self._require_free_variable_names(
            list(self.component_names) + [f"{base}_indep"]
        )

        self.mesh_vars = [
            uw.discretisation.MeshVariable(
                name, self.mesh, 1, degree=self.degree, continuous=True
            )
            for name in self.component_names
        ]

        # Solver unknown: a single (1, n_independent) MATRIX variable carrying
        # the solved components as columns.  Kept in sync with
        # mesh_vars[:n_independent] around each solve -- see _pull_state /
        # _push_state.
        self._u = uw.discretisation.MeshVariable(
            f"{base}_indep",
            self.mesh,
            (1, self.n_independent),
            vtype=uw.VarType.MATRIX,
            degree=self.degree,
            continuous=True,
        )

        # History symbols must be unique per model.  Eulerian_DDt defaults
        # varsymbol to "u", so every instance names its history layers u^* /
        # u^** ...  UW3 expressions are persistent containers keyed by name, so
        # two models on one mesh then alias each other's history: the residual
        # of the first model reads the second model's (empty) psi_star and the
        # solve collapses to zero (measured: one model preserves the field, two
        # drive it to ~1e-4).  Deriving the symbol from the component name --
        # which is the parent name for a decay/ingrowth model -- keeps each
        # model's history distinct.  Component names may be LaTeX (the zircon
        # example uses r"{}^{238}\text{U}"), so strip to a safe identifier.
        history_symbol = re.sub(r"[^0-9A-Za-z]", "", str(self.component_names[0]))
        self._ddt = uw.systems.Eulerian_DDt(
            self.mesh,
            self._u,
            uw.VarType.MATRIX,
            self.degree,
            True,
            order=self._ddt_order,
            num_components=(1, self.n_independent),
            varsymbol=rf"x_{{{history_symbol}}}",
        )

        self._diffusion_matrix = sp.zeros(self.n_independent, self.n_independent)
        self.diffusion_values = {}
        if diffusion_matrix is not None:
            self.set_diffusion_matrix(diffusion_matrix)
        if diffusion_values is not None:
            self.set_diffusion_values(diffusion_values)

        # Source term, one entry per solved component.  Zero by default.
        self.S = sp.Matrix.zeros(self.n_independent, 1)
        self._source_row = sp.Matrix.zeros(1, self.n_independent)

        # Flux history for Adams-Moulton averaging at order > 1.  Entry i is
        # the component flux matrix evaluated at history level i.
        self._flux_history = []

        self.solver = _MonolithicCoupledSolver(
            self.mesh,
            self._u,
            self,
            degree=self.degree,
            DuDt=self._ddt,
        )
        self.solver.constitutive_model = uw.constitutive_models.Constitutive_Model
        # Kept as a list so existing callers (e.g. _tighten_solver) can iterate
        # `model.solvers` uniformly across single- and multi-component models.
        self.solvers = [self.solver]

        self.pre_solve_hooks = []
        self.post_solve_hooks = []
        self._is_initialized = False

    # ------------------------------------------------------------------
    # Variable name safety
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_variable_name(name):
        """Match UW3's MeshVariable name normalisation exactly.

        ``MeshVariable.__new__`` keys the mesh registry on
        ``re.sub(r"[^a-zA-Z0-9_]", "", name)``, so two names that differ only
        in punctuation/braces are the SAME key to UW3.
        """
        return re.sub(r"[^a-zA-Z0-9_]", "", str(name))

    def _require_free_variable_names(self, names):
        """Raise if any of ``names`` is already registered on the mesh.

        UW3's ``MeshVariable.__new__`` does NOT create a new variable when a
        name collides -- it prints ``Variable with name ... already exists on
        the mesh - Skipping`` and silently returns the existing object.  A
        second model built with the same component names therefore writes into
        the first model's fields and fails in the same quiet way the shared
        history-symbol bug did.  Fail loudly here instead.
        """
        taken = []
        for name in names:
            clean = self._clean_variable_name(name)
            if clean in self.mesh.vars:
                taken.append(f"{name!r} (registry key {clean!r})")
        if taken:
            raise ValueError(
                "MeshVariable name collision on this mesh -- UW3 would silently "
                "reuse the existing variable(s) instead of creating new ones: "
                + "; ".join(taken)
                + ".  Use distinct component_names / variable_name per model, "
                "or build the models on separate meshes."
            )

    # ------------------------------------------------------------------
    # State synchronisation between the public mesh_vars and the solver
    # ------------------------------------------------------------------

    def _pull_state(self):
        """Copy public mesh_vars into the solver unknown before a solve."""
        with self.mesh.access(self._u, *self.mesh_vars[: self.n_independent]):
            for k in range(self.n_independent):
                self._u.data[:, k] = self.mesh_vars[k].data[:, 0]

    def _push_state(self):
        """Copy the solver unknown back and refresh the closure component."""
        with self.mesh.access(self._u, *self.mesh_vars):
            for k in range(self.n_independent):
                self.mesh_vars[k].data[:, 0] = self._u.data[:, k]
            if self.closure:
                total = np.zeros_like(self.mesh_vars[-1].data[:, 0])
                for k in range(self.n_independent):
                    total += self.mesh_vars[k].data[:, 0]
                self.mesh_vars[-1].data[:, 0] = 1.0 - total

    def _mark_rewire(self):
        """Flag the solver's pointwise functions for a rebuild.

        UW3's ``Template`` descriptor creates its expression ONCE, on first
        access, and freezes the result (a lambda that raises ``AttributeError``
        is silently replaced by a permanent ``0``).  So changing the diffusion
        matrix, diffusivity values or source term is not enough on its own: the
        cached template expressions have to be dropped so the next build
        re-evaluates them against the current state.

        Safe to call before the solver exists (e.g. from ``set_diffusion_matrix``
        during ``__init__``).
        """
        solver = getattr(self, "solver", None)
        if solver is None:
            return
        solver._needs_function_rewire = True
        for attr in ("_expr_F0", "_expr_F1"):
            if hasattr(solver, attr):
                delattr(solver, attr)

    # ------------------------------------------------------------------
    # Growth / re-initialisation support
    # ------------------------------------------------------------------

    @property
    def ddt(self):
        """The Eulerian DDt history manager for the solved components.

        Exposed so growth/injection workflows can inspect or manage the time
        derivative history.  For the usual case use :meth:`sync_history_to_state`.
        """
        return self._ddt

    def sync_history_to_state(self):
        """Point the time-derivative history at the current fields.

        Call this after manually overwriting ``mesh_vars`` -- grain growth,
        material injection, or re-initialisation -- so the BDF stencil does not
        interpret the overwrite as a physical jump in the solution.

        Without this, ``bdf()`` compares the new state against a stale
        ``psi_star`` and produces a spurious ``dC/dt`` of order
        ``(overwrite - old) / dt``, which can swamp the diffusion flux.

        Every history level is set to the current state and the BDF order ramp
        restarts, so the next step is first-order (backward Euler) from the new
        state before ``order`` ramps back up.  That matters: merely filling the
        history slots would leave BDF2 evaluating
        ``1.5 (psi^{n+1} - psi) / dt``, i.e. a 1.5x overshoot of the derivative,
        because it would treat the injected state as two identical past levels.
        """
        self._pull_state()
        self._ddt.initialise_history()
        # Restart the order ramp so the step after a reset is BDF1.
        if hasattr(self._ddt, "_n_solves_completed"):
            self._ddt._n_solves_completed = 0
        self._refresh_flux_history()

    def clear_boundary_conditions(self):
        """Remove every Dirichlet and Neumann condition from the solver.

        Growth workflows rebuild the boundary set as the grain front advances;
        this clears the stored conditions reliably (the underlying solver keeps
        an ``essential_bcs``/``bcs`` alias pair, so rebinding only one of them
        is fragile).
        """
        solver = self.solver
        solver.essential_bcs = []
        solver.natural_bcs = []
        solver.bcs = solver.essential_bcs
        solver._needs_function_rewire = True

    # ------------------------------------------------------------------
    # Symbolic diffusion matrix and composition symbols
    # ------------------------------------------------------------------

    @property
    def diffusion_matrix(self):
        return self._diffusion_matrix

    @diffusion_matrix.setter
    def diffusion_matrix(self, value):
        self.set_diffusion_matrix(value)

    def set_diffusion_matrix(self, diffusion_matrix):
        """Set the symbolic coupled diffusion matrix (n_independent x n_independent)."""
        diffusion_matrix = sp.Matrix(diffusion_matrix)
        if diffusion_matrix.shape != (self.n_independent, self.n_independent):
            raise ValueError(
                f"Diffusion matrix must be {self.n_independent} x "
                f"{self.n_independent}, got {diffusion_matrix.shape}"
            )
        self._diffusion_matrix = diffusion_matrix

        # Drop stored symbol values that no longer appear in the matrix.
        valid = self._diffusion_matrix.free_symbols
        self.diffusion_values = {
            sym: val for sym, val in self.diffusion_values.items() if sym in valid
        }
        self._mark_rewire()

    @staticmethod
    def _symbol_name(symbol):
        return symbol.name.replace("\\", "").replace("{", "").replace("}", "")

    def _resolve_diffusion_symbol(self, key):
        if isinstance(key, sp.Symbol):
            return key
        if isinstance(key, str):
            key_name = self._symbol_name(sp.Symbol(key))
            for symbol in self._diffusion_matrix.free_symbols:
                if self._symbol_name(symbol) == key_name:
                    return symbol
        raise KeyError(f"Could not resolve diffusion symbol key: {key}")

    def set_diffusion_values(self, values):
        """Map diffusivity symbols (or their names) to numeric values."""
        if not isinstance(values, dict):
            raise TypeError("diffusion values must be provided as a dictionary")
        for key, value in values.items():
            symbol = self._resolve_diffusion_symbol(key)
            self.diffusion_values[symbol] = _non_dimensionalise_symbolic_value(value)
        self._mark_rewire()

    def _composition_substitutions(self, history_index=None):
        """Resolve ``xi_k`` composition symbols to the (optionally historical) unknowns.

        ``history_index=None`` uses the current unknown; an integer uses that
        DDt history level, which is what the flux history needs.

        Scans both the diffusion matrix and the source term: reaction terms
        (decay / ingrowth) reference the species abundances too, and those
        symbols must resolve to the current unknown or the residual is not
        convertible to C.

        The composition is always taken at the *current* unknown, so a
        composition-dependent ``D`` is treated fully implicitly and the step is
        genuinely nonlinear.  A Picard lag is not offered: the coupled Newton
        solve handles the nonlinearity directly (measured at ~4 total
        iterations with bit-identical results to a lagged solve), so a separate
        linearised path would only add a second way to get a different answer.
        """
        if history_index is None:
            indep = [self._u.sym[0, k] for k in range(self.n_independent)]
        else:
            source = self._ddt.psi_star[history_index]
            indep = [source.sym[0, k] for k in range(self.n_independent)]

        candidate_symbols = set(self._diffusion_matrix.free_symbols)
        candidate_symbols |= set(self.S.free_symbols)

        subs = {}
        for symbol in candidate_symbols:
            name = self._symbol_name(symbol)
            m = re.match(r"^xi_(\d+)$", name)
            if not m:
                continue
            idx = int(m.group(1))
            if 1 <= idx <= self.n_independent:
                subs[symbol] = indep[idx - 1]
            elif self.closure and idx == self.n_components:
                subs[symbol] = 1 - sum(indep)
        return subs

    def _substituted_diffusion_matrix(self, history_index=None):
        out = self._diffusion_matrix
        if self.diffusion_values:
            out = out.subs(self.diffusion_values)
        return out.subs(self._composition_substitutions(history_index=history_index))

    def _flux_matrix(self, history_index=None):
        """Component flux matrix ``D @ grad(X)``, shape (n_independent, dim)."""
        diffusion = self._substituted_diffusion_matrix(history_index=history_index)
        if history_index is None:
            grad = self._u.jacobian()
        else:
            grad = self._ddt.psi_star[history_index].jacobian()
        return diffusion * grad

    def _refresh_flux_history(self):
        """Rebuild the Adams-Moulton flux history from the DDt history levels."""
        if self.order <= 1:
            self._flux_history = []
            return
        depth = min(self.order, len(self._ddt.psi_star))
        self._flux_history = [
            self._flux_matrix(history_index=i) for i in range(depth)
        ]

    # ------------------------------------------------------------------
    # Diffusivity convenience accessors
    # ------------------------------------------------------------------

    @property
    def diffusivity(self):
        """Current substituted diagonal diffusivity terms for the solved components."""
        mat = self._substituted_diffusion_matrix()
        return [mat[i, i] for i in range(self.n_independent)]

    @diffusivity.setter
    def diffusivity(self, values):
        if self.n_independent == 1 and np.isscalar(values):
            values = [values]
        values = list(values)
        if len(values) != self.n_independent:
            raise ValueError(
                f"Expected {self.n_independent} diffusivity values, got {len(values)}"
            )
        updates = {}
        for i, value in enumerate(values):
            diagonal = self._diffusion_matrix[i, i]
            if not isinstance(diagonal, sp.Symbol):
                raise ValueError(
                    f"Diagonal diffusion_matrix[{i},{i}] is not a symbol; "
                    "set values explicitly with set_diffusion_values({...})."
                )
            updates[diagonal] = value
        self.set_diffusion_values(updates)

    def set_component_diffusivity(self, component_index, value):
        """Set the diagonal diffusivity symbol for one solved component."""
        if component_index >= self.n_independent:
            raise ValueError(f"Component index must be < {self.n_independent}")
        diagonal = self._diffusion_matrix[component_index, component_index]
        if not isinstance(diagonal, sp.Symbol):
            raise ValueError(
                f"Diagonal diffusion_matrix[{component_index},{component_index}] "
                "is not a symbol; set values explicitly with set_diffusion_values({...})."
            )
        self.set_diffusion_values({diagonal: value})

    # ------------------------------------------------------------------
    # Source term
    # ------------------------------------------------------------------

    @property
    def source(self):
        """Source term, one value per solved component."""
        return self.S

    @source.setter
    def source(self, value):
        """Set the source term.

        Accepts a scalar (applied to every solved component), a sequence of
        length ``n_independent``, or a SymPy matrix.  Entries may be symbolic;
        composition symbols are resolved like the diffusion matrix.
        """
        self.set_source(value)

    def set_source(self, value):
        """Set the per-component source term ``S_i``."""
        if value is None:
            value = 0.0
        if isinstance(value, (int, float, np.floating)):
            self.S = sp.Matrix([sp.sympify(value)] * self.n_independent)
        else:
            mat = sp.Matrix(value)
            flat = list(mat)
            if len(flat) != self.n_independent:
                raise ValueError(
                    f"source must have {self.n_independent} entries, got {len(flat)}"
                )
            self.S = sp.Matrix(flat)
        self._mark_rewire()

    def _source_row_matrix(self):
        """The source as a ``(1, n_independent)`` row for the residual, with
        composition symbols resolved."""
        row = sp.Matrix(1, self.n_independent, lambda i, j: self.S[j])
        subs = self._composition_substitutions()
        return row.subs(subs) if subs else row

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _boundary_label(self, boundary):
        """Resolve a boundary spec to the label the solver expects.

        Accepts either a ``mesh.boundaries`` key (``'Left'``, ``'Boundary0'``)
        or an already-resolved label string, so callers can pass either the
        attribute name or ``boundary.name``.
        """
        boundaries = self.mesh.boundaries
        if hasattr(boundaries, boundary):
            return getattr(boundaries, boundary).name
        return boundary

    def _bc_row(self, component_index, value):
        """Build a length-n_independent conds row with one entry set."""
        row = [None] * self.n_independent
        row[component_index] = value
        return row

    def add_dirichlet_bc(self, boundary_conditions, component_index=None):
        """Apply Dirichlet BCs to one or all solved components.

        ``boundary_conditions`` maps boundary names to fixed values.  Keys may
        be ``mesh.boundaries`` attribute names or raw boundary labels.
        """
        indices = (
            range(self.n_independent) if component_index is None else [component_index]
        )
        for idx in indices:
            if idx >= self.n_independent:
                raise ValueError(f"Component index must be < {self.n_independent}")
            for boundary, value in boundary_conditions.items():
                self.solver.add_dirichlet_bc(
                    self._bc_row(idx, value), self._boundary_label(boundary)
                )

    def add_neumann_bc(self, boundary_conditions, component_index=None):
        """Apply Neumann BCs to one or all solved components."""
        indices = (
            range(self.n_independent) if component_index is None else [component_index]
        )
        for idx in indices:
            if idx >= self.n_independent:
                raise ValueError(f"Component index must be < {self.n_independent}")
            for boundary, value in boundary_conditions.items():
                self.solver.add_neumann_bc(
                    self._bc_row(idx, value), self._boundary_label(boundary)
                )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def register_pre_solve_hook(self, name, hook_function, *args, **kwargs):
        self.pre_solve_hooks.append(
            (name, functools.partial(hook_function, *args, **kwargs))
        )

    def run_pre_solve_hooks(self):
        for name, hook in self.pre_solve_hooks:
            hook()

    def register_post_solve_hook(self, name, hook_function, *args, **kwargs):
        self.post_solve_hooks.append(
            (name, functools.partial(hook_function, *args, **kwargs))
        )

    def run_post_solve_hooks(self):
        for name, hook in self.post_solve_hooks:
            hook()

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def init_model(self):
        """Seed the DDt history from the current state and mark as ready."""
        self._pull_state()
        self._ddt.update_pre_solve(0.0)
        self._ddt.update_post_solve(0.0)
        self._refresh_flux_history()
        self._is_initialized = True

    def update_kappa(self):
        """No-op kept for API compatibility (the coupled matrix is rebuilt per solve)."""

    def update_history_terms(self, dt=0.0):
        """Roll the DDt history forward and rebuild the flux history."""
        self._ddt.update_post_solve(dt)
        self._refresh_flux_history()

    def get_flux_history_vector(self, component_index=0, history_index=0):
        """Return a 1 x dim SymPy row: one component's flux at one history level."""
        if history_index >= len(self._flux_history):
            return sp.Matrix([[0] * self.mesh.dim])
        return sp.Matrix(self._flux_history[history_index][component_index, :])

    def get_flux_history_snapshot(self, history_index=0):
        """Return (n_independent x dim) flux at one history level."""
        if history_index >= len(self._flux_history):
            return sp.Matrix.zeros(self.n_independent, self.mesh.dim)
        return sp.Matrix(self._flux_history[history_index])

    def _estimate_max_coupled_diffusivity(self):
        """Largest |eigenvalue| of the substituted matrix, over the mesh.

        Used only to suggest a timestep that resolves the diffusion operator --
        it is an ACCURACY heuristic, not a stability bound.  Backward Euler is
        A- and L-stable, so the step is safe at any size; exceeding this just
        costs O(dt) accuracy (measured: ~0.6% at the suggested step, ~6% at
        10x it).  ``min_dt`` deliberately overrides it.

        The spectral radius is what matters here, not the largest stored
        constant: composition dependence and the signs of cross terms are both
        taken into account.
        """
        mat = self._substituted_diffusion_matrix()
        if not mat.free_symbols:
            try:
                eigvals = np.linalg.eigvals(np.array(mat.tolist(), dtype=float))
                return float(np.max(np.abs(eigvals)))
            except Exception:
                return float(np.max(np.abs(np.array(mat.tolist(), dtype=float))))

        # Composition-dependent: sample the matrix over the mesh nodes.
        try:
            coords = self._u.coords
            samples = [np.asarray(uw.function.evaluate(expr, coords)).ravel()
                       for expr in mat]
            per_node = np.array(samples).reshape(self.n_independent, self.n_independent, -1)
            radius = np.array([
                np.max(np.abs(np.linalg.eigvals(per_node[:, :, k])))
                for k in range(per_node.shape[-1])
            ])
            return float(np.max(radius)) if radius.size else 0.0
        except Exception:
            return 0.0

    def _effective_order(self):
        """BDF/AM order actually used at this step: ramps up from 1."""
        return int(min(self.order, self.step + 1))

    def _make_solver(self, mesh_variable):
        """Back-compat shim: the model now owns a single coupled solver."""
        return self.solver

    def _advance(self, time_step, time_units, unit_name, picard=None):
        """One timestep: hooks, pull, solve, push, history roll.

        ``picard`` overrides the model-level ``picard`` setting for this step.
        """
        if uw.mpi.rank == 0:
            print(
                f"\nStep {self.step}, dt: "
                f"{uw.scaling.dimensionalise(time_step, time_units).m} {unit_name}, "
                f"Time: {uw.scaling.dimensionalise(self.current_time, time_units).m:.2f} "
                f"{unit_name}",
                flush=True,
            )

        warmup = self.picard if picard is None else picard
        self.run_pre_solve_hooks()
        self._pull_state()
        self.solver.step(
            time_step,
            self._ddt,
            source=self._source_row_matrix(),
            effective_order=self._effective_order(),
            flux_history=self._flux_history,
            picard=warmup,
        )
        self._push_state()
        self.run_post_solve_hooks()

        # NB: ``solver.step`` already rolled the DDt history via
        # ``update_post_solve``.  Calling ``update_history_terms`` here would
        # shift the history chain a second time, making ``psi_star[1]``
        # identical to ``psi_star[0]`` and silently breaking BDF2 and above.
        self._refresh_flux_history()

        self.current_time += time_step
        self.step += 1

    def run_simulation(self, duration, max_dt=None, min_dt=None, time_step_factor=0.1,
                       picard=None):
        """Advance the model for ``duration``.

        Parameters:
        - duration: total simulation time with units (e.g. 1e5 * uw.year)
        - max_dt, min_dt: optional timestep clamps
        - time_step_factor: scales the *suggested* step
            ``time_step_factor * h^2 / kappa_max``.  Accuracy heuristic only --
            backward Euler is unconditionally stable, so this never guards
            against blow-up.  ``min_dt`` is a hard floor and wins over it, by
            design, when you want larger steps than the operator would choose.
        - picard: override the model-level frozen-tangent warm-up count for
            this run (None keeps ``self.picard``)
        """
        time_units = (
            duration.units if hasattr(duration, "units") else uw.scaling.units.year
        )
        unit_name = str(time_units).split()[-1]

        if self.current_time == 0.0 or self.step == 0 or not self._is_initialized:
            self.init_model()

        target = uw.scaling.non_dimensionalise(duration)
        while self.current_time < target:
            max_kappa = self._estimate_max_coupled_diffusivity()
            if max_kappa > 0:
                max_time_step = (
                    time_step_factor * (self.mesh.get_min_radius() ** 2 / max_kappa)
                )
            else:
                max_time_step = np.inf

            dt_specified_max = (
                uw.scaling.non_dimensionalise(max_dt) if max_dt is not None else np.inf
            )
            dt_specified_min = (
                uw.scaling.non_dimensionalise(min_dt) if min_dt is not None else 0.0
            )
            time_step = max(min(dt_specified_max, max_time_step), dt_specified_min)
            if self.current_time + time_step > target:
                time_step = target - self.current_time

            self._advance(time_step, time_units, unit_name, picard=picard)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def view(self, verbose: int = 0, show_solver: bool = False):
        _print_header("Multicomponent diffusion model (single coupled solve)")
        _mesh_summary(self.mesh)
        _view_markdown(
            f"- Components: `{self.component_names}` "
            f"({self.n_independent} solved, closure={self.closure})"
        )
        _view_markdown(
            f"- Time stepper: `order={self.order}, step={self.step}, "
            f"current_time={self.current_time}`"
        )
        if verbose > 0:
            _view_symbol(r"\mathbf{D}", self._diffusion_matrix)
            if self.diffusion_values:
                _view_markdown(f"- Diffusion values: `{self.diffusion_values}`")
            _view_symbol(r"\mathbf{S}", self.S)
        if show_solver:
            _view_markdown(f"- Solver: `{type(self.solver).__name__}`")


class _MonolithicCoupledSolver(uw.systems.SNES_MultiComponent):
    """Single-solve coupled multicomponent diffusion.

    Residual contract of ``SNES_MultiComponent``
    (``petsc_generic_snes_solvers.pyx::_setup_pointwise_functions``)::

        F0 shape (1, Nc)    -- per-component source
        F1 shape (Nc, cdim) -- per-component flux row

    The assembled weak form is ``integral grad(v).F1 + v.F0 = 0``, i.e.
    ``div(F1) = F0``.  With::

        F0_i = dx_i/dt - S_i            (DDt time derivative minus source)
        F1_i = sum_j D_ij grad(x_j)     (row i of D @ grad(X))

    the strong form is ``dx_i/dt = div(sum_j D_ij grad(x_j)) + S_i`` exactly.
    This is the same convention ``SNES_Poisson`` uses (``F0 = -f`` with
    ``f = -du/dt + S``), so a single-component run reproduces ``DiffusionModel``.

    ``u.jacobian()`` returns shape ``(var_dim, mesh_dim)``, which for a
    ``(1, Nc)`` unknown is ``(Nc, cdim)`` -- exactly the ``F1`` layout, one
    flux row per component.
    """

    F0 = Template(
        r"\partial_t \mathbf{x} - \mathbf{S}",
        lambda self: self.DuDt.time_derivative() - self.Owner._source_row_matrix(),
        "per-component BDF time derivative minus the source term",
    )

    F1 = Template(
        r"\sum_j D_{ij} \nabla x_j",
        lambda self: self._residual_flux(),
        "coupled flux: row i = sum_j D_ij grad(x_j), shape (Nc, cdim)",
    )

    def __init__(self, mesh, u_Field, owner, degree=2, DuDt=None, verbose=False):
        n_indep = int(u_Field.shape[0]) * int(u_Field.shape[1])
        # These must exist before anything touches F0/F1: the Template
        # descriptor calls value_fn on first access and freezes the result,
        # replacing any AttributeError with a permanent 0.
        self.Owner = owner
        self._effective_order = 1
        self._flux_history = []
        self._dt_now = None
        self._dt_history = []

        super().__init__(
            mesh,
            u_Field=u_Field,
            n_components=n_indep,
            degree=degree,
            verbose=verbose,
            DuDt=DuDt,
        )

        # Drop any template expression created during super().__init__ while
        # the attributes above were still missing, so the first build
        # re-evaluates F0/F1 against the real state.
        for attr in ("_expr_F0", "_expr_F1"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._needs_function_rewire = True

        # SNES_MultiComponent forces ksp_rtol=1e-3, which is 100x looser than
        # the PETSc default the scalar Poisson path inherits.  That shows up as
        # visible drift on exact fixed points (uniform states, machine-precision
        # closure).  Match the scalar path so the two agree to solver tolerance.
        self.petsc_options["ksp_rtol"] = 1.0e-6
        self.petsc_options["ksp_atol"] = 1.0e-14

    def step(self, dt, ddt, source=None, effective_order=1, flux_history=None,
             picard=-1, raise_on_failure=True):
        """Advance one step: pre-solve history update, one coupled solve.

        ``effective_order`` selects the BDF order of the time derivative.
        ``flux_history`` is accepted for API compatibility (the flux is always
        evaluated at the new time level).

        ``picard`` controls UW3's frozen-tangent warm-up before Newton:

        * ``-1`` -- (default) no warm-up; go straight to Newton.
        * ``0``  -- automatic: up to one step on a cold start of an
          unknown-dependent flux, none otherwise.
        * ``N>0`` -- up to N frozen-tangent steps.

        The warm-up is deliberately run with ``max_it = N``, so PETSc reports
        ``DIVERGED_MAX_IT`` after exactly N iterations.  That message is
        expected, not a failure.  Measured on a composition-dependent coupled
        operator it is 1:1 with Newton (same total iteration count, identical
        fields), which is why the default is off.

        UW3 prints a divergence warning but still returns a vector, so a
        diverged step can silently masquerade as success.  By default this
        raises unless the nonlinear solve actually converged.
        """
        self._effective_order = int(effective_order)
        self._flux_history = list(flux_history) if flux_history else []
        self._dt_now = dt
        self._dt_history = list(getattr(ddt, "_dt_history", []))

        # The residual flux is D @ grad(X) with live field symbols, and the
        # time derivative reads the DDt's live coefficient expressions (updated
        # by update_pre_solve without a JIT rebuild), so no template rebuild is
        # needed here regardless of order.

        ddt.update_pre_solve(dt)
        self.solve(picard=picard)
        ddt.update_post_solve(dt)

        reason = self.snes.getConvergedReason()
        if raise_on_failure and reason <= 0:
            raise RuntimeError(
                f"SNES solve diverged (reason={reason}, "
                f"its={self.snes.getIterationNumber()}); "
                "pass raise_on_failure=False to ignore."
            )
        return int(reason)

    def _residual_flux(self):
        """Component flux, evaluated at the new time level.

        The flux is always treated fully implicitly: ``D @ grad(X)`` with
        ``X`` the current unknown.  Higher-order accuracy comes from the BDF
        order of the time derivative (see ``time_derivative``), not from
        averaging the flux over history.

        An Adams-Moulton average of the flux history was tried here and is
        *not* used: it is only conditionally stable.  A single-mode analysis
        gives amplification factor 1.25 at ``mu = kappa k^2 dt = 20``, which
        the highest finite-element modes reach even when the resolved test
        mode is stable at ``mu ~ 0.04``.  The spurious high modes then grow and
        contaminate the solution, and the error stops converging with ``dt``.
        BDF with the flux at the new level is A- and L-stable at every order
        (amplification 0.15 at the same ``mu = 20``).
        """
        return self.Owner._flux_matrix(history_index=None)


class DiffusionModel(MulticomponentDiffusionModel):
    """Single scalar field diffusion, the ``n=1, closure=False`` special case.

    Strong form::

        dC/dt = div(kappa grad(C)) + S

    ``S`` is an optional source term.  This class is retained for backwards
    compatibility and thin convenience: it is the one-component instance of
    :class:`MulticomponentDiffusionModel` with no closure constraint, so the
    same single-solve machinery drives it.  The two agree to solver tolerance
    (measured 6.7e-9 on a unit-square eigenmode).

    Recommended workflow for new users:
    1. Configure non-dimensional scaling before setting diffusivity/time.
    2. Construct ``DiffusionModel(variable_name, mesh, ...)``.
    3. Set initial conditions on ``mesh_var`` and assign ``diffusivity``.
    4. Add boundary conditions with ``add_dirichlet_bc`` / ``add_neumann_bc``.
    5. Optionally register hooks, then ``run_simulation(...)``.
    """

    def __init__(self, variable_name, mesh, degree=2, order=1):
        """
        Parameters:
        - variable_name: name of the scalar field (str)
        - mesh: underworld3 mesh object
        - degree: FE basis degree (default 2)
        - order: time integration order, 1-3 (default 1 = backward Euler)
        """
        self.variable_name = variable_name
        super().__init__(
            [variable_name],
            mesh,
            diffusion_matrix=sp.Matrix([[sp.Symbol(f"kappa_{variable_name}")]]),
            degree=degree,
            order=order,
            closure=False,
        )

    # -- scalar-oriented conveniences --------------------------------------

    @property
    def mesh_var(self):
        """The single scalar MeshVariable (alias for ``mesh_vars[0]``)."""
        return self.mesh_vars[0]

    @property
    def diffusion_solver(self):
        """The underlying solver (alias for ``solver``)."""
        return self.solver

    @property
    def diffusivity(self):
        """Current dimensional diffusivity."""
        nd_value = float(self._substituted_diffusion_matrix()[0, 0])
        return uw.scaling.dimensionalise(
            nd_value, uw.scaling.units.meter ** 2 / uw.scaling.units.second
        )

    @diffusivity.setter
    def diffusivity(self, value):
        """Set the diffusivity (accepts a dimensional value)."""
        self.set_diffusion_values(
            {self._diffusion_matrix[0, 0]: _non_dimensionalise_symbolic_value(value)}
        )

    def add_dirichlet_bc(self, boundary_conditions, component_index=None):
        """Apply Dirichlet BCs (scalar form: dict of boundary -> value)."""
        super().add_dirichlet_bc(boundary_conditions, component_index=0)

    def add_neumann_bc(self, boundary_conditions, component_index=None):
        """Apply Neumann BCs (scalar form: dict of boundary -> flux)."""
        super().add_neumann_bc(boundary_conditions, component_index=0)

    def view(self, verbose: int = 0, show_solver: bool = False):
        _print_header("Scalar diffusion model (single solve)")
        _mesh_summary(self.mesh)
        _view_markdown(
            f"- Field: `{self.variable_name}`, "
            f"step={self.step}, current_time={self.current_time}"
        )
        if verbose > 0:
            _view_symbol(r"\kappa", self._substituted_diffusion_matrix()[0, 0])
            _view_symbol(r"S", self._source_row_matrix()[0, 0])

class _ComponentSolverView:
    """Back-compat view exposing one component of a coupled solver.

    The historical ``DiffusionDecayIngrowthModel`` owned a separate scalar
    ``Poisson`` solver per species, and user code drives them as::

        model.parent_diffusion.add_dirichlet_bc([value], boundary_name)

    Now that both species advance in a single coupled solve, this adapter maps
    that per-component call onto the coupled solver's component row, leaving
    the other component unconstrained.  Any other attribute is delegated, so
    ``petsc_options`` and friends keep working.
    """

    def __init__(self, solver, component_index, n_independent):
        self._solver = solver
        self._component_index = component_index
        self._n_independent = n_independent

    def _cond_row(self, conds):
        if isinstance(conds, (list, tuple)):
            value = conds[0] if len(conds) == 1 else conds[self._component_index]
        else:
            value = conds
        row = [None] * self._n_independent
        row[self._component_index] = value
        return row

    def add_dirichlet_bc(self, conds, boundary, components=None):
        self._solver.add_dirichlet_bc(self._cond_row(conds), boundary)

    def add_neumann_bc(self, conds, boundary, components=None):
        self._solver.add_neumann_bc(self._cond_row(conds), boundary)

    def __getattr__(self, name):
        return getattr(self._solver, name)


class DiffusionDecayIngrowthModel(MulticomponentDiffusionModel):
    """Coupled parent-daughter diffusion with radioactive decay / ingrowth.

    Strong form::

        dC_p/dt = div(kappa_p grad C_p) - lambda C_p
        dC_d/dt = div(kappa_d grad C_d) + lambda C_p

    with ``lambda = ln 2 / half_life``.  Parent loss is daughter gain, so the
    total ``C_p + C_d`` is conserved under zero normal flux.

    This is the two-component, no-closure instance of
    :class:`MulticomponentDiffusionModel` (``closure=False``: the two species
    are concentrations, not fractions summing to one) with a linear reaction
    source ``S = [-lambda x_1, +lambda x_1]``.

    Why the single coupled solve
    ----------------------------
    The previous implementation owned a separate ``Poisson`` solver per
    species and assembled the ingrowth term from a symbolic reference to the
    parent field.  The parent solved first, so the daughter's ``+lambda C_p``
    saw the already-updated parent -- a block Gauss-Seidel lag within the
    timestep.  Here both species advance together, so the reaction coupling is
    implicit at the same time level.  The system is linear in the unknowns
    (the source is linear in ``C_p``), so each step is a single linear solve.

    Recommended workflow:
    1. Configure Underworld non-dimensional scaling first (length/time).
    2. Construct ``DiffusionDecayIngrowthModel(...)`` with ``half_life``.
    3. Set initial fields on ``parent_mesh_var`` / ``daughter_mesh_var``.
    4. Set ``parent_diffusivity`` and ``daughter_diffusivity``.
    5. Apply boundary conditions (parent, daughter, or both).
    6. ``run_simulation(...)``.

    Stability guidance:
    - ``run_simulation`` caps the step at ``1/lambda`` so the reaction stays
      resolved, in addition to the diffusive cap.
    - Prefer a smaller ``max_dt`` / ``diffusion_time_step_factor`` for short
      half-lives, sharp gradients, or strongly varying diffusivity.
    """

    def __init__(self, parent_name, daughter_name, mesh, half_life,
                 degree=2, order=1):
        """
        Parameters:
        - parent_name: Name of the parent isotope (str)
        - daughter_name: Name of the daughter isotope (str)
        - mesh: underworld3 mesh object
        - half_life: Half-life of the parent isotope in years (float)
        - degree: FE basis degree (default 2)
        - order: time integration order, 1-3 (default 1)
        """
        self.parent_name = parent_name
        self.daughter_name = daughter_name
        self.half_life = half_life
        self.lambda_decay = np.log(2) / half_life

        parent_sym = sp.Symbol(f"kappa_{parent_name}")
        daughter_sym = sp.Symbol(f"kappa_{daughter_name}")
        # No cross-diffusion: each species diffuses down its own gradient.
        diffusion_matrix = sp.Matrix([[parent_sym, 0], [0, daughter_sym]])

        super().__init__(
            [parent_name, daughter_name],
            mesh,
            diffusion_matrix=diffusion_matrix,
            degree=degree,
            order=order,
            closure=False,
        )

        # Linear reaction source, expressed against the composition symbols so
        # it resolves to the *current unknown* -- fully implicit.  xi_1 is the
        # parent field (n_independent = 2, no closure component).
        lambda_nd = uw.scaling.non_dimensionalise(self.lambda_decay)
        parent_symbol = sp.Symbol("xi_1")
        self.set_source(
            sp.Matrix([-lambda_nd * parent_symbol, lambda_nd * parent_symbol])
        )
        self._parent_sym = parent_sym
        self._daughter_sym = daughter_sym

    # -- per-species views (backwards compatibility) ------------------------

    @property
    def parent_mesh_var(self):
        return self.mesh_vars[0]

    @property
    def daughter_mesh_var(self):
        return self.mesh_vars[1]

    @property
    def parent_diffusion(self):
        return _ComponentSolverView(self.solver, 0, self.n_independent)

    @property
    def daughter_diffusion(self):
        return _ComponentSolverView(self.solver, 1, self.n_independent)

    @property
    def parent_DuDt(self):
        return self._ddt

    @property
    def daughter_DuDt(self):
        return self._ddt

    @property
    def parent_diffusivity(self):
        return float(self._substituted_diffusion_matrix()[0, 0])

    @parent_diffusivity.setter
    def parent_diffusivity(self, value):
        self.set_diffusion_values(
            {self._parent_sym: _non_dimensionalise_symbolic_value(value)}
        )

    @property
    def daughter_diffusivity(self):
        return float(self._substituted_diffusion_matrix()[1, 1])

    @daughter_diffusivity.setter
    def daughter_diffusivity(self, value):
        self.set_diffusion_values(
            {self._daughter_sym: _non_dimensionalise_symbolic_value(value)}
        )

    @property
    def parent_S(self):
        return self.S[0]

    @parent_S.setter
    def parent_S(self, value):
        new = list(self.S)
        new[0] = sp.Matrix(value)[0]
        self.set_source(new)

    @property
    def daughter_S(self):
        return self.S[1]

    @daughter_S.setter
    def daughter_S(self, value):
        new = list(self.S)
        new[1] = sp.Matrix(value)[0]
        self.set_source(new)

    # -- boundary conditions ----------------------------------------------

    def add_dirichlet_bc(self, boundary_conditions, solver_type='both'):
        """Apply Dirichlet BCs to the parent and/or daughter species.

        ``boundary_conditions`` maps boundary names to fixed values
        (``mesh.boundaries`` keys or raw labels); ``solver_type`` is 'parent',
        'daughter' or 'both'.
        """
        targets = {'parent': [0], 'daughter': [1], 'both': [0, 1]}[solver_type]
        for idx in targets:
            for boundary, value in boundary_conditions.items():
                self.solver.add_dirichlet_bc(
                    self._bc_row(idx, value), self._boundary_label(boundary)
                )

    def add_neumann_bc(self, boundary_conditions, solver_type='both'):
        """Apply Neumann BCs to the parent and/or daughter species."""
        targets = {'parent': [0], 'daughter': [1], 'both': [0, 1]}[solver_type]
        for idx in targets:
            for boundary, value in boundary_conditions.items():
                self.solver.add_neumann_bc(
                    self._bc_row(idx, value), self._boundary_label(boundary)
                )

    # -- time stepping ----------------------------------------------------

    def _decay_capped_timestep(self, time_step, lambda_nd):
        """Never step past 1/lambda: the reaction must stay resolved."""
        if lambda_nd > 0:
            return min(time_step, 1.0 / lambda_nd)
        return time_step

    def run_simulation(self, duration, min_dt=None, max_dt=None,
                       diffusion_time_step_factor=0.5, picard=None):
        """
        Run the decay simulation for the specified duration.

        Notes:
        - The diffusion-decay-ingrowth coupling is highly timestep sensitive.
          If results are noisy, unstable, or timestep dependent, reduce
          max_dt and/or diffusion_time_step_factor, and set a physically
          meaningful min_dt.

        Parameters:
        - duration: Total simulation time with units (e.g. 1e5*u.second)
        - min_dt: Minimum time step (optional)
        - max_dt: Maximum time step (optional)
        - diffusion_time_step_factor: scales the *suggested* diffusive step
            ``time_step_factor * h^2 / kappa_max`` (default 0.5).  Accuracy
            heuristic only; the scheme is unconditionally stable.  Note the
            reaction term is additionally capped at ``1/lambda`` so decay stays
            resolved.
        """
        time_units = (
            duration.units if hasattr(duration, "units") else uw.scaling.units.year
        )
        unit_name = str(time_units).split()[-1]

        if self.current_time == 0.0 or self.step == 0 or not self._is_initialized:
            self.init_model()

        lambda_nd = uw.scaling.non_dimensionalise(self.lambda_decay)

        target = uw.scaling.non_dimensionalise(duration)
        while self.current_time < target:
            max_kappa = self._estimate_max_coupled_diffusivity()
            if max_kappa > 0:
                max_time_step_diff = (
                    diffusion_time_step_factor
                    * self.mesh.get_min_radius() ** 2
                    / max_kappa
                )
            else:
                max_time_step_diff = np.inf

            max_time_step = min(max_time_step_diff, 1.0 / lambda_nd if lambda_nd > 0 else np.inf)

            dt_specified_max = (
                uw.scaling.non_dimensionalise(max_dt) if max_dt is not None else np.inf
            )
            dt_specified_min = (
                uw.scaling.non_dimensionalise(min_dt) if min_dt is not None else 0.0
            )
            time_step = max(min(dt_specified_max, max_time_step), dt_specified_min)
            if self.current_time + time_step > target:
                time_step = target - self.current_time

            self._advance(time_step, time_units, unit_name, picard=picard)

    # -- inspection -------------------------------------------------------

    def view(self, verbose: int = 0, show_solver: bool = False):
        _print_header("UWDiffusion DiffusionDecayIngrowthModel (single coupled solve)")
        _mesh_summary(self.mesh)
        _view_markdown(f"- Parent: `{self.parent_name}`")
        _view_markdown(f"- Daughter: `{self.daughter_name}`")
        _view_markdown(f"- Half life: `{self.half_life}`")
        _view_markdown(f"- Decay constant (dim): `{self.lambda_decay}`")
        if verbose > 0:
            lambda_nd = uw.scaling.non_dimensionalise(self.lambda_decay)
            _view_symbol(r"\lambda_{nd}", lambda_nd)
            _view_markdown(f"- Source term: `{_safe_symbol_text(self.S)}`")
        if show_solver:
            _view_markdown(f"- Solver: `{type(self.solver).__name__}`")
