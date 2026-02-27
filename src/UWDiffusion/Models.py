import os
import underworld3 as uw
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import functools
import re

from underworld3.systems import Poisson 

from .utilities import _adams_moulton_flux

class DiffusionModel:
    """Single-component diffusion model for one scalar field on a UW mesh.

    Use this class when you want to solve a standard diffusion problem for a
    single quantity (for example, one isotope concentration) with optional
    higher-order time integration.

        What this model handles for you:
        - Creates and stores one Underworld mesh variable for the component.
        - Solves the implicit diffusion equation each timestep.
        - Supports BDF time derivatives with Adams-Moulton flux history terms
            when ``order > 1``.
        - Provides pre/post solve hooks for custom runtime logic.

     Recommended workflow for new users:
     1. Configure non-dimensional scaling in Underworld before setting
         diffusivity or time values.
     2. Construct ``DiffusionModel(variable_name, mesh, ...)``.
     3. Set initial conditions on ``mesh_var`` and assign ``diffusivity``.
     4. Add boundary conditions with ``add_dirichlet_bc`` and/or
         ``add_neumann_bc``.
     5. Optionally register hooks, then call ``run_simulation(...)``.

        Notes:
        - ``diffusivity`` accepts dimensional Pint values and is stored internally
            in non-dimensional form.
        - ``run_simulation`` chooses timesteps from mesh size and diffusivity,
            then clamps by ``min_dt`` / ``max_dt`` if provided.
        """
    def __init__(self, 
                 variable_name, 
                 mesh,
                 degree=2,
                 order=1):
        """
        Initialize the diffusion model for a single variable.

        Parameters:
        - variable_name: Name of the variable (str)
        - mesh: underworld3 mesh object
        - degree: Degree of the finite element basis functions (int, default=2)
        - order: Order of the diffusion solver (int, default=1)
        """
        self.variable_name = variable_name
        self.degree = degree

        self.order = order
        self._ddt_order = max(1, order)

        self.current_time = 0.0
        self.step = 0

        self.mesh = mesh

        # Initialize mesh variables
        self.mesh_var = uw.discretisation.MeshVariable(
            f"{self.variable_name}", self.mesh, 1, degree=self.degree, continuous=True
        )

        # Hooks for pre-solve and post-solve
        self.pre_solve_hooks = []  # List of functions to call before each solve
        self.post_solve_hooks = []  # List of functions to call after each solve

        self.DuDt = uw.systems.Eulerian_DDt(
            mesh,
            self.mesh_var,
            vtype=uw.VarType.SCALAR,
            degree=self.mesh_var.degree,
            continuous=self.mesh_var.continuous,
            varsymbol=self.mesh_var.symbol,
            verbose=False,
            order=self._ddt_order,
        )

        self._diffusivity_expr = uw.function.expression(
            rf'\upkappa_{self.mesh_var.name}',
            0.
        )

        self.kappa_star = [
            uw.function.expression(
                rf'\upkappa^{"*"*(i+1)}_{self.mesh_var.name}',
                0.
            )
            for i in range(order)
        ]

        ### source term
        self.S = sp.Matrix.zeros(1, 1)

        self.flux_history = [sp.Matrix([[0] * mesh.dim]) for _ in range(order)]

        self.diffusion_solver = self._setup_diffusion_solver(self.mesh_var)
        self._is_initialized = False

    def _setup_diffusion_solver(self, mesh_variable):
        """
        Private method to set up a diffusion solver for a mesh variable.
        """
        solver = Poisson(self.mesh, u_Field=mesh_variable)
        solver.constitutive_model = uw.constitutive_models.GenericFluxModel

        return solver

    def init_model(self):
        """
        Initialize the diffusion solver for the variable.
        """
        self.update_history_terms(dt=0.0)
        self._is_initialized = True

    def update_kappa(self):
        if self.order == 0:
            return
        ### copy down the chain
        if self.order > 1:
            for i in range(self.order - 1, 0, -1):
                self.kappa_star[i].sym = self.kappa_star[i-1].sym
        ### update the first one
        self.kappa_star[0].sym = self._diffusivity_expr.sym
        

    def update_history_terms(self, dt=0.0):
        if self.order == 0:
            self.DuDt.update_post_solve(dt)
            return
        
        self.update_kappa()

        self.DuDt.update_post_solve(dt)

        history_len = min(self.order, len(self.DuDt.psi_star), len(self.flux_history), len(self.kappa_star))

        for i in range(history_len):
            # Update historical flux
            self.flux_history[i] = self.DuDt.psi_star[i].jacobian() * self.kappa_star[i]

    
    @property
    def diffusivity(self):
        """
        Get the current (dimensional) diffusivity value.
        """
        nd_value = float(self._diffusivity_expr.sym)
        return uw.scaling.dimensionalise(nd_value, uw.scaling.units.meter**2 / uw.scaling.units.second)

    @diffusivity.setter
    def diffusivity(self, value):
        """
        Set the diffusivity of the variable (accepts dimensional value).
        """
        new_diffusivity_nd = uw.scaling.non_dimensionalise(value)
        self._diffusivity_expr.sym = new_diffusivity_nd

    def add_dirichlet_bc(self, boundary_conditions, ):
        """
        Apply Dirichlet boundary conditions to the parent and daughter diffusion solvers.

        A list of boundary names can be accessed via 'self.mesh.boundaries.'

        Parameters:
        - boundary_conditions: A dictionary where the keys are boundary names
                               ('Left', 'Right', 'Top', 'Bottom', etc.)
                               and the values are the fixed boundary values (float).
        
        """
        for boundary, value in boundary_conditions.items():
            self.diffusion_solver.add_dirichlet_bc(value, getattr(self.mesh.boundaries, boundary).name)

        return

    def add_neumann_bc(self, boundary_conditions,):
        """
        Apply Neumann boundary conditions to the parent and daughter diffusion solvers.

        A list of boundary names can be accessed via 'self.mesh.boundaries.'

        Parameters:
        - boundary_conditions: A dictionary where the keys are boundary names
                               ('Left', 'Right', 'Top', 'Bottom', etc.)
                               and the values are the specified fluxes (float).
        """
        for boundary, flux in boundary_conditions.items():
            self.diffusion_solver.add_neumann_bc(flux, getattr(self.mesh.boundaries, boundary).name)

    def register_pre_solve_hook(self, name, hook_function, *args, **kwargs):
        """
        Register a named function to be called before each solve step.

        Parameters:
        - name: str, a unique name for the hook.
        - hook_function: Callable, a function to execute before each solve step.
        - args, kwargs: Optional positional and keyword arguments to pass to the hook.
        """
        self.pre_solve_hooks.append((name, functools.partial(hook_function, *args, **kwargs)))

    def run_pre_solve_hooks(self):
        """
        Execute all registered pre-solve hooks.
        """
        for name, hook in self.pre_solve_hooks:
            hook()  # Optionally, you can use the name for logging or debugging

    def register_post_solve_hook(self, name, hook_function, *args, **kwargs):
        """
        Register a named function to be called after each solve step.

        Parameters:
        - name: str, a unique name for the hook.
        - hook_function: Callable, a function to execute after each solve step.
        - args, kwargs: Optional positional and keyword arguments to pass to the hook.
        """
        self.post_solve_hooks.append((name, functools.partial(hook_function, *args, **kwargs)))

    def run_post_solve_hooks(self):
        """
        Execute all registered post-solve hooks.
        """
        for name, hook in self.post_solve_hooks:
            hook()  # Optionally, you can use the name for logging or debugging

    def run_simulation(self, duration, max_dt=None, min_dt=None, time_step_factor=0.1):
        """
        Run the decay simulation for the specified duration.

        Parameters:
        - duration: Total simulation time with units (e.g., 1e5*u.second)
        - max_dt: Maximum time step (optional)
        - min_dt: Minimum time step (optional)
        - time_step_factor: CFL condition for stable/accurate timesteps (default=0.1)
        """
        # Extract units from duration parameter
        time_units = duration.units if hasattr(duration, 'units') else uw.scaling.units.year
        unit_name = str(time_units).split()[-1]

        if self.current_time == 0.0 or self.step == 0 or not self._is_initialized:
            self.init_model()

        while self.current_time < uw.scaling.non_dimensionalise(duration):


            kappa_vals = uw.function.evaluate(self._diffusivity_expr, self.mesh.data)
            max_time_step = time_step_factor * self.mesh.get_min_radius()**2 / np.max(kappa_vals)


            dt_specified_max = uw.scaling.non_dimensionalise(max_dt) if max_dt is not None else np.inf
            dt_specified_min = uw.scaling.non_dimensionalise(min_dt) if min_dt is not None else 0

            time_step = max(min(dt_specified_max, max_time_step), dt_specified_min)


            if self.current_time + time_step > uw.scaling.non_dimensionalise(duration):
                time_step = uw.scaling.non_dimensionalise(duration) - self.current_time

            if uw.mpi.rank == 0:
                print(f"\nStep {self.step}, dt: {uw.scaling.dimensionalise(time_step, time_units).m} {unit_name}, Time: {uw.scaling.dimensionalise(self.current_time, time_units).m:.2f} {unit_name}", flush=True)


            # Pre-solve hook
            self.run_pre_solve_hooks()

            # Solve diffusion equations
            # Update source term using BDF time stepping
            effective_order = min(self.order, self.step + 1)
            bdf_order = max(1, effective_order)
            self.diffusion_solver.f = - (sp.simplify(self.DuDt.bdf(order=bdf_order)) / time_step) + self.S

            # Construct current flux
            flux = self.DuDt._psi_meshVar.jacobian() * self._diffusivity_expr

            # Adams-Moulton flux term
            flux_term = _adams_moulton_flux(
                flux,
                self.flux_history,
                order=effective_order,
                dt_current=time_step,
                dt_history=getattr(self.DuDt, "_dt_history", []),
            )
            self.diffusion_solver.constitutive_model.Parameters.flux = flux_term

            ### solve 
            self.diffusion_solver.solve()

            ### update hisotry terms
            self.update_history_terms(time_step)

            # Post-solve hook
            self.run_post_solve_hooks()

            # Update time and step
            self.current_time += time_step
            self.step += 1

        if uw.mpi.rank == 0:
            print(f"\nStep {self.step}, Time: {uw.scaling.dimensionalise(self.current_time, time_units).m:.2f} {unit_name}", flush=True)


class DiffusionDecayIngrowthModel:
    """Coupled parent-daughter diffusion model with radioactive decay/ingrowth.

    Use this class when both species diffuse and are linked by radioactive
    decay, where parent loss is daughter gain during each timestep.

    What this model solves:
    - Parent diffusion + decay sink term.
    - Daughter diffusion + ingrowth source term.
    - Consistent time integration for both fields using the same timestep,
        with optional higher-order BDF + Adams-Moulton history treatment.

    Recommended workflow for new users:
    1. Configure Underworld non-dimensional scaling first (length/time).
    2. Construct ``DiffusionDecayIngrowthModel(...)`` with ``half_life``.
    3. Set initial parent/daughter fields on
            ``parent_mesh_var`` / ``daughter_mesh_var``.
    4. Set ``parent_diffusivity`` and ``daughter_diffusivity``.
    5. Apply boundary conditions (parent, daughter, or both).
    6. Run ``run_simulation(...)`` with conservative timestep controls.

    Stability guidance:
    - This coupled problem is often timestep-sensitive.
    - Prefer smaller ``max_dt`` and/or ``diffusion_time_step_factor`` for
        short half-lives, sharp gradients, or strongly varying diffusivity.
    """
    def __init__(self, 
                 parent_name, 
                 daughter_name,
                 mesh, 
                 half_life,
                 degree=2,
                 order=1):
        """
        Initialize the decay simulation for a parent-daughter decay chain.

                Notes:
                - The coupled diffusion-decay-ingrowth solve can be very sensitive to
                    timestep size. Use conservative min_dt / max_dt settings,
                    particularly for short half-lives, sharp concentration gradients,
                    or rapidly varying diffusivity.

        Parameters:
        - parent_name: Name of the parent isotope (str)
        - daughter_name: Name of the daughter isotope (str)
        - mesh: underworld3 mesh object
        - half_life: Half-life of the parent isotope in years (float)
        - degree: Degree of the finite element basis functions (int, default=2)
        - order: Order of the diffusion solver (int, default=1)
        """
        self.parent_name = parent_name
        self.daughter_name = daughter_name
        self.half_life = half_life
        self.lambda_decay = np.log(2) / half_life  # Decay constant
        self.degree = degree
        self.order = order
        self._ddt_order = max(1, order)

        self.mesh = mesh

        # Initialize mesh variables
        self.parent_mesh_var = uw.discretisation.MeshVariable(
            f"{self.parent_name}", self.mesh, 1, degree=self.degree, continuous=True
        )
        self.daughter_mesh_var = uw.discretisation.MeshVariable(
            f"{self.daughter_name}", self.mesh, 1, degree=self.degree, continuous=True
        )

        # Hooks for pre-solve and post-solve
        self.pre_solve_hooks = []  # List of functions to call before each solve
        self.post_solve_hooks = []  # List of functions to call after each solve

        self.current_time = 0.0
        self.step = 0

        # Setup parent Eulerian DDt for time history
        self.parent_DuDt = uw.systems.Eulerian_DDt(
            mesh,
            self.parent_mesh_var,
            vtype=uw.VarType.SCALAR,
            degree=self.parent_mesh_var.degree,
            continuous=self.parent_mesh_var.continuous,
            varsymbol=self.parent_mesh_var.symbol,
            verbose=False,
            order=self._ddt_order,
        )

        self._parent_diffusivity_expr = uw.function.expression(
            rf'\upkappa_{self.parent_mesh_var.name}',
            0.
        )

        # Setup daughter Eulerian DDt for time history
        self.daughter_DuDt = uw.systems.Eulerian_DDt(
            mesh,
            self.daughter_mesh_var,
            vtype=uw.VarType.SCALAR,
            degree=self.daughter_mesh_var.degree,
            continuous=self.daughter_mesh_var.continuous,
            varsymbol=self.daughter_mesh_var.symbol,
            verbose=False,
            order=self._ddt_order,
        )

        self._daughter_diffusivity_expr = uw.function.expression(
            rf'\upkappa_{self.daughter_mesh_var.name}',
            0.
        )

        # Parent diffusivity chain
        self.parent_kappa_star = [
            uw.function.expression(
                rf'\upkappa^{"*"*(i+1)}_{self.parent_mesh_var.name}',
                0.
            )
            for i in range(order)
        ]

        # Daughter diffusivity chain
        self.daughter_kappa_star = [
            uw.function.expression(
                rf'\upkappa^{"*"*(i+1)}_{self.daughter_mesh_var.name}',
                0.
            )
            for i in range(order)
        ]

        # Source terms for decay and ingrowth
        self.parent_S = sp.Matrix.zeros(1, 1)
        self.daughter_S = sp.Matrix.zeros(1, 1)

        # Flux history for parent and daughter
        self.parent_flux_history = [sp.Matrix([[0] * mesh.dim]) for _ in range(order)]
        self.daughter_flux_history = [sp.Matrix([[0] * mesh.dim]) for _ in range(order)]
        self._is_initialized = False

        # # Diffusion solvers
        self.parent_diffusion = self._setup_diffusion_solver(self.parent_mesh_var)
        self.daughter_diffusion = self._setup_diffusion_solver(self.daughter_mesh_var)


    def update_parent_kappa(self):
        """Update parent diffusivity chain."""
        if self.order == 0:
            return
        if self.order > 1:
            for i in range(self.order - 1, 0, -1):
                self.parent_kappa_star[i].sym = self.parent_kappa_star[i-1].sym
        self.parent_kappa_star[0].sym = self._parent_diffusivity_expr.sym

    def update_daughter_kappa(self):
        """Update daughter diffusivity chain."""
        if self.order == 0:
            return
        if self.order > 1:
            for i in range(self.order - 1, 0, -1):
                self.daughter_kappa_star[i].sym = self.daughter_kappa_star[i-1].sym
        self.daughter_kappa_star[0].sym = self._daughter_diffusivity_expr.sym

    def update_parent_history_terms(self, dt=0.0):
        """Update parent history terms."""
        if self.order == 0:
            self.parent_DuDt.update_post_solve(dt)
            return
        self.update_parent_kappa()
        self.parent_DuDt.update_post_solve(dt)

        history_len = min(self.order, len(self.parent_DuDt.psi_star), len(self.parent_flux_history), len(self.parent_kappa_star))

        for i in range(history_len):
            self.parent_flux_history[i] = self.parent_DuDt.psi_star[i].jacobian() * self.parent_kappa_star[i]

    def update_daughter_history_terms(self, dt=0.0):
        """Update daughter history terms."""
        if self.order == 0:
            self.daughter_DuDt.update_post_solve(dt)
            return
        self.update_daughter_kappa()
        self.daughter_DuDt.update_post_solve(dt)

        history_len = min(self.order, len(self.daughter_DuDt.psi_star), len(self.daughter_flux_history), len(self.daughter_kappa_star))

        for i in range(history_len):
            self.daughter_flux_history[i] = self.daughter_DuDt.psi_star[i].jacobian() * self.daughter_kappa_star[i]

    def _setup_diffusion_solver(self, mesh_variable):
        """
        Private method to set up a diffusion solver for a mesh variable.
        """
        solver = Poisson(self.mesh, u_Field=mesh_variable)
        solver.constitutive_model = uw.constitutive_models.GenericFluxModel

        return solver

    def init_model(self):
        """
        Initialize the diffusion solvers for parent and daughter variables.
        """

        self.update_parent_history_terms(dt=0.0)
        self.update_daughter_history_terms(dt=0.0)
        self._is_initialized = True


    @property
    def parent_diffusivity(self):
        """
        Get the current (dimensional) diffusivity value for the parent.
        """
        nd_value = float(self._parent_diffusivity_expr.sym)
        return uw.scaling.dimensionalise(nd_value, uw.scaling.units.meter**2 / uw.scaling.units.second)

    @parent_diffusivity.setter
    def parent_diffusivity(self, value):
        """
        Set the diffusivity of the parent variable (accepts dimensional value).
        """
        new_diffusivity_nd = uw.scaling.non_dimensionalise(value)
        self._parent_diffusivity_expr.sym = new_diffusivity_nd

    @property
    def daughter_diffusivity(self):
        """
        Get the current (dimensional) diffusivity value for the daughter.
        """
        nd_value = float(self._daughter_diffusivity_expr.sym)
        return uw.scaling.dimensionalise(nd_value, uw.scaling.units.meter**2 / uw.scaling.units.second)

    @daughter_diffusivity.setter
    def daughter_diffusivity(self, value):
        """
        Set the diffusivity of the daughter variable (accepts dimensional value).
        """
        new_diffusivity_nd = uw.scaling.non_dimensionalise(value)
        self._daughter_diffusivity_expr.sym = new_diffusivity_nd

    def add_dirichlet_bc(self, boundary_conditions, solver_type='both'):
        """
        Apply Dirichlet boundary conditions to the parent and/or daughter diffusion solvers.

        A list of boundary names can be accessed via 'self.mesh.boundaries.'

        Parameters:
        - boundary_conditions: A dictionary where the keys are boundary names
                               ('Left', 'Right', 'Top', 'Bottom', etc.)
                               and the values are the fixed boundary values (float).
        - solver_type: 'parent', 'daughter', or 'both' (default='both')
        """
        for boundary, value in boundary_conditions.items():
            if solver_type in ('parent', 'both'):
                self.parent_diffusion.add_dirichlet_bc(value, getattr(self.mesh.boundaries, boundary).name)
            if solver_type in ('daughter', 'both'):
                self.daughter_diffusion.add_dirichlet_bc(value, getattr(self.mesh.boundaries, boundary).name)

    def add_neumann_bc(self, boundary_conditions, solver_type='both'):
        """
        Apply Neumann boundary conditions to the parent and/or daughter diffusion solvers.

        A list of boundary names can be accessed via 'self.mesh.boundaries.'

        Parameters:
        - boundary_conditions: A dictionary where the keys are boundary names
                               ('Left', 'Right', 'Top', 'Bottom', etc.)
                               and the values are the specified fluxes (float).
        - solver_type: 'parent', 'daughter', or 'both' (default='both')
        """
        for boundary, flux in boundary_conditions.items():
            if solver_type in ('parent', 'both'):
                self.parent_diffusion.add_neumann_bc(flux, getattr(self.mesh.boundaries, boundary).name)
            if solver_type in ('daughter', 'both'):
                self.daughter_diffusion.add_neumann_bc(flux, getattr(self.mesh.boundaries, boundary).name)

    def register_pre_solve_hook(self, name, hook_function, *args, **kwargs):
        """
        Register a named function to be called before each solve step.

        Parameters:
        - name: str, a unique name for the hook.
        - hook_function: Callable, a function to execute before each solve step.
        - args, kwargs: Optional positional and keyword arguments to pass to the hook.
        """
        self.pre_solve_hooks.append((name, functools.partial(hook_function, *args, **kwargs)))

    def run_pre_solve_hooks(self):
        """
        Execute all registered pre-solve hooks.
        """
        for name, hook in self.pre_solve_hooks:
            hook()  # Optionally, you can use the name for logging or debugging

    def register_post_solve_hook(self, name, hook_function, *args, **kwargs):
        """
        Register a named function to be called after each solve step.

        Parameters:
        - name: str, a unique name for the hook.
        - hook_function: Callable, a function to execute after each solve step.
        - args, kwargs: Optional positional and keyword arguments to pass to the hook.
        """
        self.post_solve_hooks.append((name, functools.partial(hook_function, *args, **kwargs)))

    def run_post_solve_hooks(self):
        """
        Execute all registered post-solve hooks.
        """
        for name, hook in self.post_solve_hooks:
            hook()  # Optionally, you can use the name for logging or debugging


    def _numerical_decay_ingrowth(self, time_step):
        """
        Update the parent and daughter abundances based on radioactive decay.
        """
        lambda_nd = uw.scaling.non_dimensionalise(self.lambda_decay)
        decay_factor = lambda_nd * time_step

        exp_decay = np.exp(-decay_factor)
        decayed_fraction = -np.expm1(-decay_factor)

        with self.mesh.access(self.parent_mesh_var, self.daughter_mesh_var):
            parent_old = self.parent_mesh_var.data[:, 0].copy()
            self.parent_mesh_var.data[:, 0] = parent_old * exp_decay
            self.daughter_mesh_var.data[:, 0] += parent_old * decayed_fraction

    def run_simulation(self, duration, min_dt=None, max_dt=None, diffusion_time_step_factor=0.5):
        """
        Run the decay simulation for the specified duration.

                Notes:
                - The diffusion-decay-ingrowth coupling is highly timestep sensitive.
                    If results are noisy, unstable, or timestep dependent, reduce
                    max_dt and/or diffusion_time_step_factor, and set a physically
                    meaningful min_dt.

        Parameters:
        - duration: Total simulation time with units (e.g., 1e5*u.second)
        - min_dt: Minimum time step (optional)
        - max_dt: Maximum time step (optional)
        - diffusion_time_step_factor: CFL condition for stable/accurate timesteps (default=0.5)
        """
        # Extract units from duration parameter
        time_units = duration.units if hasattr(duration, 'units') else uw.scaling.units.year
        unit_name = str(time_units).split()[-1]

        if self.current_time == 0.0 or self.step == 0 or not self._is_initialized:
            self.init_model()
        
        # doesn't change during the simulation
        lambda_nd = uw.scaling.non_dimensionalise(self.lambda_decay)
        max_time_step_decay = 1.0 / lambda_nd

        while self.current_time < uw.scaling.non_dimensionalise(duration):

            # Pre-solve hook first
            self.run_pre_solve_hooks()

            # may change during simulation
            parent_kappa_vals = uw.function.evaluate(
                self._parent_diffusivity_expr, self.mesh.data
            )
            daughter_kappa_vals = uw.function.evaluate(
                self._daughter_diffusivity_expr, self.mesh.data
            )

            kappa_vals = max(np.max(parent_kappa_vals), np.max(daughter_kappa_vals))

            max_time_step_diff = diffusion_time_step_factor * self.mesh.get_min_radius()**2 / kappa_vals

            max_time_step = min(max_time_step_diff, max_time_step_decay)

            dt_specified_max = uw.scaling.non_dimensionalise(max_dt) if max_dt is not None else np.inf
            dt_specified_min = uw.scaling.non_dimensionalise(min_dt) if min_dt is not None else 0

            time_step = max(min(dt_specified_max, max_time_step), dt_specified_min)

            if self.current_time + time_step > uw.scaling.non_dimensionalise(duration):
                time_step = uw.scaling.non_dimensionalise(duration) - self.current_time

            if uw.mpi.rank == 0:
                print(
                    f"\nStep {self.step}, dt: {uw.scaling.dimensionalise(time_step, time_units).m} {unit_name}, "
                    f"Time: {uw.scaling.dimensionalise(self.current_time, time_units).m:.2f} {unit_name}",
                    flush=True,
                )

            # Coupled reaction terms (decay / ingrowth) solved with diffusion in the same timestep
            self.parent_S = sp.Matrix([[-lambda_nd * self.parent_mesh_var.sym[0]]])
            self.daughter_S = sp.Matrix([[lambda_nd * self.parent_mesh_var.sym[0]]])

            # Previous operator-split approach (kept for reference)
            # self.parent_S = sp.Matrix.zeros(1, 1)
            # self.daughter_S = sp.Matrix.zeros(1, 1)
            # self._numerical_decay_ingrowth(time_step)

            effective_order = min(self.order, self.step + 1)
            bdf_order = max(1, effective_order)

            # Update parent history terms and solve
            self.parent_diffusion.f = -(sp.simplify(self.parent_DuDt.bdf(order=bdf_order)) / time_step) + self.parent_S
            parent_flux = self.parent_DuDt._psi_meshVar.jacobian() * self._parent_diffusivity_expr
            parent_flux_term = _adams_moulton_flux(
                parent_flux,
                self.parent_flux_history,
                order=effective_order,
                dt_current=time_step,
                dt_history=getattr(self.parent_DuDt, "_dt_history", []),
            )
            self.parent_diffusion.constitutive_model.Parameters.flux = parent_flux_term
            self.parent_diffusion.solve()
            self.update_parent_history_terms(time_step)

            # Update daughter history terms and solve
            self.daughter_diffusion.f = -(sp.simplify(self.daughter_DuDt.bdf(order=bdf_order)) / time_step) + self.daughter_S
            daughter_flux = self.daughter_DuDt._psi_meshVar.jacobian() * self._daughter_diffusivity_expr
            daughter_flux_term = _adams_moulton_flux(
                daughter_flux,
                self.daughter_flux_history,
                order=effective_order,
                dt_current=time_step,
                dt_history=getattr(self.daughter_DuDt, "_dt_history", []),
            )
            self.daughter_diffusion.constitutive_model.Parameters.flux = daughter_flux_term
            self.daughter_diffusion.solve()
            self.update_daughter_history_terms(time_step)

            # Post-solve hook
            self.run_post_solve_hooks()

            # Update time and step
            self.current_time += time_step
            self.step += 1

        if uw.mpi.rank == 0:
            print(
                f"\nStep {self.step}, Time: {uw.scaling.dimensionalise(self.current_time, time_units).m:.2f} {unit_name}",
                flush=True,
            )


class MulticomponentDiffusionModel:
    """Coupled multicomponent diffusion using a symbolic diffusion matrix.

    Use this model when component fluxes are composition-coupled (including
    cross-diffusion terms), not just independent scalar diffusion.

        Model structure:
        - For ``n`` named components, the first ``n-1`` are solved explicitly.
        - The last component is implicit and updated as
            ``x_n = 1 - sum(x_1, ..., x_{n-1})``.
        - A symbolic ``(n-1) x (n-1)`` diffusion matrix defines self/cross terms.
            Symbol values are supplied with ``set_diffusion_values(...)``.

     Recommended workflow for new users:
     1. Define component names with the implicit component last.
     2. Provide a symbolic diffusion matrix (SymPy) for coupled terms.
     3. Set symbol values with ``set_diffusion_values`` (supports symbol keys
         or symbol-name strings; dimensional Pint values are accepted).
     4. Set initial conditions and boundary conditions for independent
         components.
     5. Run ``run_simulation(...)``.

        Notes:
        - Composition-dependent symbols like ``xi_i`` are substituted from current
            (or history) fields before flux assembly.
        - Higher-order BDF and Adams-Moulton corrections are applied consistently
            across all independent components.
        """
    def __init__(self,
                 component_names,
                 mesh,
                 diffusion_matrix=None,
                 diffusion_values=None,
                 degree=2,
                 order=1):
        """
        Initialize a multicomponent diffusion model with coupled diffusion.

        Parameters:
        - component_names: List of component names (list of str). Last component is implicitly 1 - sum(others).
        - mesh: underworld3 mesh object
        - diffusion_matrix: Symbolic diffusion matrix ((n-1) x (n-1)) coupling first n-1 components.
                           If None, assumes diagonal matrix with zeros.
        - diffusion_values: Optional dictionary mapping symbolic diffusivity terms (e.g. D_12)
                    to values (dimensional pint or non-dimensional float).
        - degree: Degree of the finite element basis functions (int, default=2)
        - order: Order of the diffusion solver (int, default=1)
        """
        self.component_names = component_names
        self.n_components = len(component_names)
        self.n_independent = self.n_components - 1  # Last component is implicit
        self.degree = degree
        self.order = order
        self._ddt_order = max(1, order)
        self.mesh = mesh

        self.current_time = 0.0
        self.step = 0

        # Initialize mesh variables for all components (including implicit last one)
        self.mesh_vars = [
            uw.discretisation.MeshVariable(
                name, self.mesh, 1, degree=self.degree, continuous=True
            )
            for name in component_names
        ]

        # Hooks for pre-solve and post-solve
        self.pre_solve_hooks = []
        self.post_solve_hooks = []

        # Eulerian DDt for independent components only (first n-1)
        self.DuDt_list = [
            uw.systems.Eulerian_DDt(
                mesh,
                self.mesh_vars[i],
                vtype=uw.VarType.SCALAR,
                degree=self.mesh_vars[i].degree,
                continuous=self.mesh_vars[i].continuous,
                varsymbol=self.mesh_vars[i].symbol,
                verbose=False,
                order=self._ddt_order,
            )
            for i in range(self.n_independent)
        ]

        # Diffusivity chains for independent components only
        self.kappa_star_list = [
            [
                uw.function.expression(
                    rf'\upkappa^{"*"*(j+1)}_{self.mesh_vars[i].name}',
                    0.
                )
                for j in range(order)
            ]
            for i in range(self.n_independent)
        ]

        # Source terms for independent components only
        self.S_list = [sp.Matrix.zeros(1, 1) for _ in range(self.n_independent)]

        # Flux history for independent components only (scalar-valued SymPy matrix)
        # Shape: (n_independent, order * mesh.dim)
        # For component i, history k, dimension d -> column index = k*mesh.dim + d
        self.flux_history_list = sp.Matrix.zeros(
            self.n_independent,
            self.order * self.mesh.dim,
        )

        self.diffusion_values = {}

        # Store diffusion matrix (default zero until explicitly set)
        self._diffusion_matrix = sp.zeros(self.n_independent, self.n_independent)
        if diffusion_matrix is not None:
            self.set_diffusion_matrix(diffusion_matrix)

        if diffusion_values is not None:
            self.set_diffusion_values(diffusion_values)

        # Initialize solvers for independent components only
        self.solvers = [
            Poisson(self.mesh, u_Field=self.mesh_vars[i])
            for i in range(self.n_independent)
        ]

        for solver in self.solvers:
            solver.constitutive_model = uw.constitutive_models.GenericFluxModel
        self._is_initialized = False

    @property
    def diffusion_matrix(self):
        return self._diffusion_matrix

    @diffusion_matrix.setter
    def diffusion_matrix(self, value):
        self.set_diffusion_matrix(value)

    def set_diffusion_matrix(self, diffusion_matrix):
        """
        Set the symbolic coupled diffusion matrix.

        Parameters:
        - diffusion_matrix: Sympy matrix of shape (n-1, n-1)
        """
        if diffusion_matrix.shape != (self.n_independent, self.n_independent):
            raise ValueError(
                f"Diffusion matrix must be {self.n_independent} x {self.n_independent}, "
                f"got {diffusion_matrix.shape}"
            )

        self._diffusion_matrix = diffusion_matrix

        if not hasattr(self, "diffusion_values"):
            self.diffusion_values = {}

        # Remove any previously-set symbol values no longer present in the matrix
        valid_symbols = self._diffusion_matrix.free_symbols
        self.diffusion_values = {
            symbol: value
            for symbol, value in self.diffusion_values.items()
            if symbol in valid_symbols
        }

    def _symbol_name(self, symbol):
        return symbol.name.replace("\\", "").replace("{", "").replace("}", "")

    def _parse_composition_index(self, symbol):
        name = self._symbol_name(symbol)
        match = re.match(r"^xi_(\d+)$", name)
        if match:
            return int(match.group(1))
        return None

    def _build_composition_substitutions(self, history_index=None):
        substitutions = {}
        symbols = self.diffusion_matrix.free_symbols

        if history_index is None:
            independent_comp = [
                self.DuDt_list[i]._psi_meshVar.sym[0]
                for i in range(self.n_independent)
            ]
        else:
            independent_comp = [
                self.DuDt_list[i].psi_star[history_index].sym[0]
                for i in range(self.n_independent)
            ]

        for symbol in symbols:
            comp_idx = self._parse_composition_index(symbol)
            if comp_idx is None:
                continue

            if 1 <= comp_idx <= self.n_independent:
                substitutions[symbol] = independent_comp[comp_idx - 1]
            elif comp_idx == self.n_components:
                substitutions[symbol] = 1 - sum(independent_comp)

        return substitutions

    def _resolve_diffusion_symbol(self, key):
        if isinstance(key, sp.Symbol):
            return key

        if isinstance(key, str):
            key_name = key.replace("\\", "").replace("{", "").replace("}", "")
            for symbol in self.diffusion_matrix.free_symbols:
                if self._symbol_name(symbol) == key_name:
                    return symbol

        raise KeyError(f"Could not resolve diffusion symbol key: {key}")

    def set_diffusion_values(self, values):
        """
        Set symbolic diffusion-term values for substitution in the diffusion matrix.

        Parameters:
        - values: dict mapping symbols (or symbol names) to values.
                  Values can be dimensional pint quantities or non-dimensional floats.
        """
        if not isinstance(values, dict):
            raise TypeError("diffusion values must be provided as a dictionary")

        for key, value in values.items():
            symbol = self._resolve_diffusion_symbol(key)
            if hasattr(value, "units"):
                value_nd = uw.scaling.non_dimensionalise(value)
            else:
                value_nd = value
            self.diffusion_values[symbol] = value_nd

    def _substituted_diffusion_matrix(self, history_index=None):
        substitutions = {}
        substitutions.update(self.diffusion_values)
        substitutions.update(self._build_composition_substitutions(history_index=history_index))
        return self.diffusion_matrix.subs(substitutions)

    def _build_gradient_substitutions(self, history_index=None):
        grad_symbols = sp.Matrix([
            sp.Symbol(f"nabla_xi{i+1}")
            for i in range(self.n_independent)
        ])

        if history_index is None:
            gradients = [
                self.DuDt_list[i]._psi_meshVar.jacobian()
                for i in range(self.n_independent)
            ]
        else:
            gradients = [
                self.DuDt_list[i].psi_star[history_index].jacobian()
                for i in range(self.n_independent)
            ]

        substitutions = {
            grad_symbols[i, 0]: gradients[i]
            for i in range(self.n_independent)
        }
        return grad_symbols, substitutions

    def _compute_flux_matrix(self, history_index=None):
        diffusion_matrix = self._substituted_diffusion_matrix(history_index=history_index)
        grad_matrix, grad_subs = self._build_gradient_substitutions(history_index=history_index)
        return (diffusion_matrix * grad_matrix).subs(grad_subs)

    def _get_flux_history_vector(self, component_index, history_index):
        return sp.Matrix([[
            self.flux_history_list[component_index, history_index * self.mesh.dim + d]
            for d in range(self.mesh.dim)
        ]])

    def get_flux_history_vector(self, component_index=0, history_index=0):
        """Return a 1 x dim SymPy row vector for one component and history index."""
        return self._get_flux_history_vector(component_index, history_index)

    def get_flux_history_snapshot(self, history_index=0):
        """Return flux history at a given history index as (n_independent x dim) SymPy matrix."""
        return sp.Matrix(
            self.n_independent,
            self.mesh.dim,
            lambda comp_idx, d: self.flux_history_list[comp_idx, history_index * self.mesh.dim + d],
        )

    def _estimate_max_coupled_diffusivity(self):
        substituted_matrix = self._substituted_diffusion_matrix(history_index=None)
        max_values = []
        for i in range(self.n_independent):
            for j in range(self.n_independent):
                coeff = substituted_matrix[i, j]
                coeff_vals = uw.function.evaluate(coeff, self.mesh.data)
                max_values.append(np.max(np.abs(coeff_vals)))

        if len(max_values) == 0:
            return 0.0

        return float(np.max(max_values))

    def update_kappa(self):
        """Update diffusivity chains for independent components from substituted diffusion matrix."""
        if self.order == 0:
            return
        diffusion_matrix_current = self._substituted_diffusion_matrix(history_index=None)

        for comp_idx in range(self.n_independent):
            if self.order > 1:
                for i in range(self.order - 1, 0, -1):
                    self.kappa_star_list[comp_idx][i].sym = self.kappa_star_list[comp_idx][i-1].sym
            # Update the first kappa from substituted coupled diffusivity terms
            self.kappa_star_list[comp_idx][0].sym = diffusion_matrix_current[comp_idx, comp_idx]

    def update_history_terms(self, dt=0.0):
        """Update history terms for independent components."""
        if self.order == 0:
            for comp_idx in range(self.n_independent):
                self.DuDt_list[comp_idx].update_post_solve(dt)
            return
        self.update_kappa()

        # First update all component histories to a consistent time level
        for comp_idx in range(self.n_independent):
            self.DuDt_list[comp_idx].update_post_solve(dt)

        history_len = min(
            self.order,
            self.flux_history_list.shape[1] // self.mesh.dim,
            *(len(self.kappa_star_list[comp_idx]) for comp_idx in range(self.n_independent)),
            *(len(self.DuDt_list[comp_idx].psi_star) for comp_idx in range(self.n_independent)),
        )

        for i in range(history_len):
            flux_matrix_i = self._compute_flux_matrix(history_index=i)
            for comp_idx in range(self.n_independent):
                flux_vec = flux_matrix_i[comp_idx, 0]
                for d in range(self.mesh.dim):
                    self.flux_history_list[comp_idx, i * self.mesh.dim + d] = flux_vec[0, d]

    def init_model(self):
        """
        Initialize the diffusion solvers.
        """
        self.update_history_terms(dt=0.0)
        self._is_initialized = True

    @property
    def diffusivity(self):
        """
        Get the current substituted diagonal diffusivity terms for independent components.

        Returns a list of sympy expressions / scalars for first n-1 components.
        """
        diffusion_matrix_current = self._substituted_diffusion_matrix(history_index=None)
        return [
            diffusion_matrix_current[i, i]
            for i in range(self.n_independent)
        ]

    @diffusivity.setter
    def diffusivity(self, values):
        """
        Set diagonal diffusivity symbols via diffusion_values mapping.

        Parameters:
        - values: List or array of values for diagonal matrix terms.
        """
        if len(values) != self.n_independent:
            raise ValueError(f"Expected {self.n_independent} diffusivity values, got {len(values)}")

        diagonal_updates = {}
        for i, value in enumerate(values):
            diagonal_symbol = self.diffusion_matrix[i, i]
            if not isinstance(diagonal_symbol, sp.Symbol):
                raise ValueError(
                    f"Diagonal diffusion_matrix[{i},{i}] is not a symbol; "
                    "set values explicitly with set_diffusion_values({...})."
                )
            diagonal_updates[diagonal_symbol] = value

        self.set_diffusion_values(diagonal_updates)

    def set_component_diffusivity(self, component_index, value):
        """
        Set the diagonal diffusivity symbol for a specific independent component.

        Parameters:
        - component_index: Index of the component (0-based, must be < n-1)
        - value: Diffusivity value (dimensional or non-dimensional)
        """
        if component_index >= self.n_independent:
            raise ValueError(f"Component index must be < {self.n_independent}")

        diagonal_symbol = self.diffusion_matrix[component_index, component_index]
        if not isinstance(diagonal_symbol, sp.Symbol):
            raise ValueError(
                f"Diagonal diffusion_matrix[{component_index},{component_index}] is not a symbol; "
                "set values explicitly with set_diffusion_values({...})."
            )

        self.set_diffusion_values({diagonal_symbol: value})

    def add_dirichlet_bc(self, boundary_conditions, component_index=None):
        """
        Apply Dirichlet boundary conditions to one or all independent components.

        Parameters:
        - boundary_conditions: A dictionary where the keys are boundary names
                               ('Left', 'Right', 'Top', 'Bottom', etc.)
                               and the values are the fixed boundary values (float).
        - component_index: Index of the component to apply BC to. If None, applies to all independent components.
        """
        if component_index is None:
            # Apply to all independent components
            for comp_idx in range(self.n_independent):
                for boundary, value in boundary_conditions.items():
                    self.solvers[comp_idx].add_dirichlet_bc(
                        value, getattr(self.mesh.boundaries, boundary).name
                    )
        else:
            # Apply to specific component
            if component_index >= self.n_independent:
                raise ValueError(f"Component index must be < {self.n_independent}")
            for boundary, value in boundary_conditions.items():
                self.solvers[component_index].add_dirichlet_bc(
                    value, getattr(self.mesh.boundaries, boundary).name
                )

    def add_neumann_bc(self, boundary_conditions, component_index=None):
        """
        Apply Neumann boundary conditions to one or all independent components.

        Parameters:
        - boundary_conditions: A dictionary where the keys are boundary names
                               ('Left', 'Right', 'Top', 'Bottom', etc.)
                               and the values are the specified fluxes (float).
        - component_index: Index of the component to apply BC to. If None, applies to all independent components.
        """
        if component_index is None:
            # Apply to all independent components
            for comp_idx in range(self.n_independent):
                for boundary, flux in boundary_conditions.items():
                    self.solvers[comp_idx].add_neumann_bc(
                        flux, getattr(self.mesh.boundaries, boundary).name
                    )
        else:
            # Apply to specific component
            if component_index >= self.n_independent:
                raise ValueError(f"Component index must be < {self.n_independent}")
            for boundary, flux in boundary_conditions.items():
                self.solvers[component_index].add_neumann_bc(
                    flux, getattr(self.mesh.boundaries, boundary).name
                )

    def register_pre_solve_hook(self, name, hook_function, *args, **kwargs):
        """
        Register a named function to be called before each solve step.

        Parameters:
        - name: str, a unique name for the hook.
        - hook_function: Callable, a function to execute before each solve step.
        - args, kwargs: Optional positional and keyword arguments to pass to the hook.
        """
        self.pre_solve_hooks.append((name, functools.partial(hook_function, *args, **kwargs)))

    def run_pre_solve_hooks(self):
        """
        Execute all registered pre-solve hooks.
        """
        for name, hook in self.pre_solve_hooks:
            hook()

    def register_post_solve_hook(self, name, hook_function, *args, **kwargs):
        """
        Register a named function to be called after each solve step.

        Parameters:
        - name: str, a unique name for the hook.
        - hook_function: Callable, a function to execute after each solve step.
        - args, kwargs: Optional positional and keyword arguments to pass to the hook.
        """
        self.post_solve_hooks.append((name, functools.partial(hook_function, *args, **kwargs)))

    def run_post_solve_hooks(self):
        """
        Execute all registered post-solve hooks.
        """
        for name, hook in self.post_solve_hooks:
            hook()

    def run_simulation(self, duration, max_dt=None, min_dt=None, time_step_factor=0.1):
        """
        Run the multicomponent diffusion simulation for the specified duration.

        Parameters:
        - duration: Total simulation time with units (e.g., 1e5*u.second)
        - max_dt: Maximum time step (optional)
        - min_dt: Minimum time step (optional)
        - time_step_factor: CFL condition for stable/accurate timesteps (default=0.1)
        """
        # Extract units from duration parameter
        time_units = duration.units if hasattr(duration, 'units') else uw.scaling.units.year
        unit_name = str(time_units).split()[-1]

        if self.current_time == 0.0 or self.step == 0 or not self._is_initialized:
            self.init_model()
        
        while self.current_time < uw.scaling.non_dimensionalise(duration):

            # Evaluate maximum effective diffusivity from coupled substituted matrix
            max_kappa = self._estimate_max_coupled_diffusivity()
            if max_kappa > 0:
                max_time_step = time_step_factor * (self.mesh.get_min_radius()**2 / max_kappa)
            else:
                max_time_step = np.inf

            dt_specified_max = uw.scaling.non_dimensionalise(max_dt) if max_dt is not None else np.inf
            dt_specified_min = uw.scaling.non_dimensionalise(min_dt) if min_dt is not None else 0

            time_step = max(min(dt_specified_max, max_time_step), dt_specified_min)

            if self.current_time + time_step > uw.scaling.non_dimensionalise(duration):
                time_step = uw.scaling.non_dimensionalise(duration) - self.current_time

            if uw.mpi.rank == 0:
                print(
                    f"\nStep {self.step}, dt: {uw.scaling.dimensionalise(time_step, time_units).m} {unit_name}, "
                    f"Time: {uw.scaling.dimensionalise(self.current_time, time_units).m:.2f} {unit_name}",
                    flush=True,
                )

            # Pre-solve hook
            self.run_pre_solve_hooks()

            flux_matrix_current = self._compute_flux_matrix(history_index=None)
            effective_order = min(self.order, self.step + 1)
            bdf_order = max(1, effective_order)

            # Solve for each independent component using the coupled diffusion matrix
            for comp_idx in range(self.n_independent):
                # Update source term using BDF time stepping
                self.solvers[comp_idx].f = (
                    -(sp.simplify(self.DuDt_list[comp_idx].bdf(order=bdf_order)) / time_step) 
                    + self.S_list[comp_idx]
                )

                # Construct coupled flux from matrix form: (D_matrix * grad_matrix).subs(...)
                flux = flux_matrix_current[comp_idx, 0]

                # Apply Adams-Moulton flux correction using component's history
                flux_history_comp = [
                    self._get_flux_history_vector(comp_idx, i)
                    for i in range(effective_order)
                ]
                flux_term = _adams_moulton_flux(
                    flux,
                    flux_history_comp,
                    order=effective_order,
                    dt_current=time_step,
                    dt_history=getattr(self.DuDt_list[comp_idx], "_dt_history", []),
                )
                self.solvers[comp_idx].constitutive_model.Parameters.flux = flux_term

            # Solve all independent components
            for comp_idx in range(self.n_independent):
                self.solvers[comp_idx].solve()

            # Update implicit last component: x_n = 1 - sum(x_1, ..., x_{n-1})
            with self.mesh.access(*self.mesh_vars):
                self.mesh_vars[-1].data[:, 0] = 1.0
                for i in range(self.n_independent):
                    self.mesh_vars[-1].data[:, 0] -= self.mesh_vars[i].data[:, 0]

            # Post-solve hook
            self.run_post_solve_hooks()

            # Update history terms for all independent components
            self.update_history_terms(time_step)

            # Update time and step
            self.current_time += time_step
            self.step += 1

        # if uw.mpi.rank == 0:
        #     print(
        #         f"\nStep {self.step}, Time: {uw.scaling.dimensionalise(self.current_time, time_units).m:.2f} {unit_name}",
        #         flush=True,
        #     )