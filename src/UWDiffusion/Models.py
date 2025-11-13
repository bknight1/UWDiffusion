import os
import underworld3 as uw
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import functools

from .solvers import SNES_Diffusion

class DiffusionModel:
    def __init__(self, 
                 variable_name, 
                 initial_abundance,
                 degree=2,
                 order=1,
                 mesh=None):
        """
        Initialize the decay simulation for a parent-daughter decay chain.

        Parameters:
        - parent_name: Name of the parent isotope (str)
        - daughter_name: Name of the daughter isotope (str)
        - half_life: Half-life of the parent isotope in years (float)
        - initial_parent: Initial abundance of the parent isotope (float)
        - initial_daughter: Initial abundance of the daughter isotope (float, default=0.0)
        - degree: Degree of the finite element basis functions (int, default=2)
        - mesh: Underworld3 mesh object (optional)
        """
        self.variable_name = variable_name
        self.degree = degree
        self.initial_abundance = initial_abundance

        self.order = order

        self.current_time = 0.0
        self.step = 0

        # Create mesh
        if mesh is None:
            self.mesh = uw.meshing.UnstructuredSimplexBox(
                minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.05, qdegree=self.degree
            )
        else:
            self.mesh = mesh

        # Initialize mesh variables
        self.mesh_var = uw.discretisation.MeshVariable(
            f"{self.variable_name}", self.mesh, 1, degree=self.degree, continuous=True
        )

        # Set initial values
        with self.mesh.access(self.mesh_var):
            if isinstance(self.initial_abundance, sp.Function):
                self.mesh_var.data[:, 0] = uw.function.evaluate(self.initial_abundance, self.mesh_var.coords)
            else:
                self.mesh_var.data[:, 0] = self.initial_abundance

        # Diffusion solvers
        self.diffusion_solver = self._setup_diffusion_solver(self.mesh_var)

        # Hooks for pre-solve and post-solve
        self.pre_solve_hooks = []  # List of functions to call before each solve
        self.post_solve_hooks = []  # List of functions to call after each solve


    def _setup_diffusion_solver(self, mesh_variable):
        """
        Private method to set up a diffusion solver for a mesh variable.
        """
        solver = SNES_Diffusion(self.mesh, u_Field=mesh_variable, order=self.order)
        # diffusivity = uw.function.expression(r'\kappa_{%s}' % mesh_variable.name, diffusivity_value)
        solver.constitutive_model = uw.constitutive_models.DiffusionModel

        return solver
    
    @property
    def diffusivity(self):
        """
        Get the current (dimensional) diffusivity value.
        """
        # Get the non-dimensional value and convert to dimensional
        nd_value = float( self.diffusion_solver.constitutive_model.Parameters.diffusivity.sym )
        return uw.scaling.dimensionalise(nd_value, uw.scaling.units.meter**2 / uw.scaling.units.second)

    @diffusivity.setter
    def diffusivity(self, value):
        """
        Set the diffusivity of the variable (accepts dimensional value).
        """
        new_diffusivity_nd = uw.scaling.non_dimensionalise(value)
        self.diffusion_solver.constitutive_model.Parameters.diffusivity.sym = new_diffusivity_nd

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
        - duration: Total simulation time in Myr (float)
        - time_step_factor: CFL condition for stable/accurate timesteps (default=0.1)
        """


        while self.current_time < uw.scaling.non_dimensionalise(duration):


            kappa_vals = uw.function.evaluate(self.diffusion_solver.constitutive_model.diffusivity, self.mesh.data)
            max_time_step = time_step_factor * self.mesh.get_min_radius()**2 / np.max(kappa_vals)


            dt_specified_max = uw.scaling.non_dimensionalise(max_dt) if max_dt is not None else np.inf
            dt_specified_min = uw.scaling.non_dimensionalise(min_dt) if min_dt is not None else 0

            time_step = max(min(dt_specified_max, max_time_step), dt_specified_min)


            if self.current_time + time_step > uw.scaling.non_dimensionalise(duration):
                time_step = uw.scaling.non_dimensionalise(duration) - self.current_time

            if uw.mpi.rank == 0:
                print(f"\nStep {self.step}, dt: {uw.scaling.dimensionalise(time_step, uw.scaling.units.megayear).m} Myr, Time: {uw.scaling.dimensionalise(self.current_time, uw.scaling.units.megayear).m:.2f} Myr", flush=True)


            # Pre-solve hook
            self.run_pre_solve_hooks()

            # Solve diffusion equations
            self.diffusion_solver.solve(dt=time_step)

            # Post-solve hook
            self.run_post_solve_hooks()

            # Update time and step
            self.current_time += time_step
            self.step += 1

        if uw.mpi.rank == 0:
            print(f"\nStep {self.step}, Time: {uw.scaling.dimensionalise(self.current_time, uw.scaling.units.megayear).m:.2f} Myr", flush=True)



class DiffusionDecayIngrowthModel:
    def __init__(self, 
                 parent_name, 
                 daughter_name, 
                 half_life, 
                 initial_parent,
                 initial_daughter=0.0,
                 degree=2,
                 order=1,
                 mesh=None):
        """
        Initialize the decay simulation for a parent-daughter decay chain.

        Parameters:
        - parent_name: Name of the parent isotope (str)
        - daughter_name: Name of the daughter isotope (str)
        - half_life: Half-life of the parent isotope in years (float)
        - initial_parent: Initial abundance of the parent isotope (float)
        - initial_daughter: Initial abundance of the daughter isotope (float, default=0.0)
        - degree: Degree of the finite element basis functions (int, default=2)
        - mesh: Underworld3 mesh object (optional)
        """
        self.parent_name = parent_name
        self.daughter_name = daughter_name
        self.half_life = half_life
        self.lambda_decay = np.log(2) / half_life  # Decay constant
        self.initial_parent = initial_parent
        self.initial_daughter = initial_daughter
        self.degree = degree
        self.order = order

        # Create mesh
        if mesh is None:
            self.mesh = uw.meshing.UnstructuredSimplexBox(
                minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.05, qdegree=self.degree
            )
        else:
            self.mesh = mesh

        # Initialize mesh variables
        self.parent_mesh_var = uw.discretisation.MeshVariable(
            f"{self.parent_name}", self.mesh, 1, degree=self.degree, continuous=True
        )
        self.daughter_mesh_var = uw.discretisation.MeshVariable(
            f"{self.daughter_name}", self.mesh, 1, degree=self.degree, continuous=True
        )

        # Set initial values
        with self.mesh.access(self.parent_mesh_var, self.daughter_mesh_var):
            self.parent_mesh_var.data[:, 0] = self.initial_parent
            self.daughter_mesh_var.data[:, 0] = self.initial_daughter
        

        # Diffusion solvers
        self.parent_diffusion = self._setup_diffusion_solver(self.parent_mesh_var)
        self.daughter_diffusion = self._setup_diffusion_solver(self.daughter_mesh_var)

        # self.parent_diffusion.S = sp.Matrix([- (uw.scaling.non_dimensionalise( self.lambda_decay ) * self.parent_mesh_var.sym[0])])
        
        # self.daughter_diffusion.S = sp.Matrix([(uw.scaling.non_dimensionalise( self.lambda_decay ) * self.parent_mesh_var.sym[0])])


        # Hooks for pre-solve and post-solve
        self.pre_solve_hooks = []  # List of functions to call before each solve
        self.post_solve_hooks = []  # List of functions to call after each solve


        self.current_time = 0.0
        self.step = 0



    def _setup_diffusion_solver(self, mesh_variable):
        """
        Private method to set up a diffusion solver for a mesh variable.
        """
        solver = SNES_Diffusion(self.mesh, u_Field=mesh_variable, order=self.order)
        # diffusivity = uw.function.expression(r'\kappa_{%s}' % mesh_variable.name, diffusivity_value)
        solver.constitutive_model = uw.constitutive_models.DiffusionModel

        return solver
    
    @property
    def parent_diffusivity(self):
        """
        Get the current (dimensional) diffusivity value for the parent.
        """
        nd_value = float( self.parent_diffusion.constitutive_model.Parameters.diffusivity.sym )
        return uw.scaling.dimensionalise(nd_value, uw.scaling.units.meter**2 / uw.scaling.units.second)

    @parent_diffusivity.setter
    def parent_diffusivity(self, value):
        """
        Set the diffusivity of the parent variable (accepts dimensional value).
        """
        new_diffusivity_nd = uw.scaling.non_dimensionalise(value)
        self.parent_diffusion.constitutive_model.Parameters.diffusivity.sym = new_diffusivity_nd

    @property
    def daughter_diffusivity(self):
        """
        Get the current (dimensional) diffusivity value for the daughter.
        """
        nd_value = float( self.daughter_diffusion.constitutive_model.Parameters.diffusivity.sym )
        return uw.scaling.dimensionalise(nd_value, uw.scaling.units.meter**2 / uw.scaling.units.second)

    @daughter_diffusivity.setter
    def daughter_diffusivity(self, value):
        """
        Set the diffusivity of the daughter variable (accepts dimensional value).
        """
        new_diffusivity_nd = uw.scaling.non_dimensionalise(value)
        self.daughter_diffusion.constitutive_model.Parameters.diffusivity.sym = new_diffusivity_nd

    def add_dirichlet_bc(self, boundary_conditions, solver):
        """
        Apply Dirichlet boundary conditions to the parent and daughter diffusion solvers.

        A list of boundary names can be accessed via 'self.mesh.boundaries.'

        Parameters:
        - boundary_conditions: A dictionary where the keys are boundary names
                               ('Left', 'Right', 'Top', 'Bottom', etc.)
                               and the values are the fixed boundary values (float).
        
        """
        for boundary, value in boundary_conditions.items():
            solver.add_dirichlet_bc(value, getattr(self.mesh.boundaries, boundary).name)

        return

    def add_neumann_bc(self, boundary_conditions, solver):
        """
        Apply Neumann boundary conditions to the parent and daughter diffusion solvers.

        A list of boundary names can be accessed via 'self.mesh.boundaries.'

        Parameters:
        - boundary_conditions: A dictionary where the keys are boundary names
                               ('Left', 'Right', 'Top', 'Bottom', etc.)
                               and the values are the specified fluxes (float).
        """
        for boundary, flux in boundary_conditions.items():
            solver.add_neumann_bc(flux, getattr(self.mesh.boundaries, boundary).name)

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
        lambda_nd = uw.scaling.non_dimensionalise( self.lambda_decay )
        decay_factor = lambda_nd * time_step


        exp_decay = np.exp(-decay_factor)
        decayed_fraction = -np.expm1(-decay_factor)

        with self.mesh.access(self.parent_mesh_var, self.daughter_mesh_var):
            parent_old = self.parent_mesh_var.data[:, 0].copy()
            self.parent_mesh_var.data[:, 0] = parent_old * exp_decay
            self.daughter_mesh_var.data[:, 0] += parent_old * decayed_fraction

        self.parent_diffusion.DuDt.update_post_solve()
        self.daughter_diffusion.DuDt.update_post_solve()

    def run_simulation(self, duration, min_dt=None, max_dt=None, diffusion_time_step_factor=0.5): #, decay_time_step_factor=0.1):
        """
        Run the decay simulation for the specified duration.

        Parameters:
        - duration: Total simulation time in Myr (float)
        - diffusion_time_step_factor: Fraction of the decay constant to use as the time step (default=0.5)
        - decay_time_step_factor: Fraction of the decay constant to use as the time step (default=0.001)
        """
        ### doesn't change during the simulation
        max_time_step_decay = 1. / uw.scaling.non_dimensionalise( self.lambda_decay )


        while self.current_time < uw.scaling.non_dimensionalise(duration):

            # Pre-solve hook first
            self.run_pre_solve_hooks()

            ### may change during simulation
            parent_kappa_vals = uw.function.evaluate(self.parent_diffusion.constitutive_model.diffusivity, self.mesh.data)
            daughter_kappa_vals = uw.function.evaluate(self.daughter_diffusion.constitutive_model.diffusivity, self.mesh.data)

            kappa_vals = max(max(parent_kappa_vals), max(daughter_kappa_vals))

            max_time_step_diff = diffusion_time_step_factor * self.mesh.get_min_radius()**2 / kappa_vals

            max_time_step = min(max_time_step_diff, max_time_step_decay)

            dt_specified_max = uw.scaling.non_dimensionalise(max_dt) if max_dt is not None else np.inf
            dt_specified_min = uw.scaling.non_dimensionalise(min_dt) if min_dt is not None else 0

            time_step = max(min(dt_specified_max, max_time_step), dt_specified_min)


            
            if self.current_time + time_step > uw.scaling.non_dimensionalise(duration):
                time_step = uw.scaling.non_dimensionalise(duration) - self.current_time

            if uw.mpi.rank == 0:
                print(f"\nStep {self.step}, dt: {uw.scaling.dimensionalise(time_step, uw.scaling.units.megayear).m} Myr,  Time: {uw.scaling.dimensionalise(self.current_time, uw.scaling.units.megayear).m:.2f} Myr", flush=True)

            # Perform decay and ingrowth first
            self._numerical_decay_ingrowth(time_step)


            # Then solve diffusion equations
            self.daughter_diffusion.solve(dt=time_step)
            self.parent_diffusion.solve(dt=time_step)

            # Post-solve hook
            self.run_post_solve_hooks()

            # Update time and step
            self.current_time += time_step
            self.step += 1
        
        if uw.mpi.rank == 0:
            print(f"\nStep {self.step}, Time: {uw.scaling.dimensionalise(self.current_time, uw.scaling.units.megayear).m:.2f} Myr", flush=True)