# UWDiffusion Project Context

UWDiffusion is a Python library designed for modelling and solving a variety of diffusion problems, built on top of the [Underworld3](https://github.com/underworldcode/underworld3) geodynamics modeling framework. It provides high-level objects to streamline the setup of diffusion models, while maintaining the flexibility to interact with core Underworld3 functionality.

## Project Overview

- **Core Technology:** Python 3.11+, [Underworld3](https://underworldcode.github.io/underworld3/), PETSc, SymPy, Gmsh.
- **Purpose:** Model single-component scalar diffusion, coupled parent-daughter radioactive decay/ingrowth, and multicomponent diffusion with cross-coupling.
- **Architecture:** 
    - `Models.py`: Contains core classes (`DiffusionModel`, `DiffusionDecayIngrowthModel`, `MulticomponentDiffusionModel`) that interface with Underworld3's `Poisson` solver and `Eulerian_DDt` time-stepping system.
    - `meshing.py`: Utilities for generating 2D unstructured meshes using Gmsh, including specialized geometries for garnet and box models.
    - `utilities.py`: Plotting tools (Concordia/Terra-Wasserburg plots) and data sampling utilities.

## Building and Running

### Installation
UWDiffusion requires `underworld3` to be pre-installed. Once Underworld3 is available, install UWDiffusion from the root directory:
```bash
pip install -e .
```

### Key Commands (Inferred)
- **Linting:** No explicit linter configured, but follow PEP 8 standards.
- **Testing:** No formal test suite exists. Validation is primarily done through Jupyter notebooks in the `Tutorials/` directory.
- **Tutorials:** Explore `Tutorials/` for practical examples of various diffusion scenarios.

## Development Conventions

- **Scaling:** Always configure Underworld3's non-dimensional scaling (`uw.scaling`) before setting diffusivity values or time durations.
- **SymPy Integration:** Diffusivity and source terms often accept SymPy expressions for spatially or compositionally dependent variables.
- **Model Workflow:**
    1. Define non-dimensional scaling.
    2. Construct a mesh (using `uw.discretisation.Mesh` or `UWDiffusion.meshing` utilities).
    3. Initialize a model class (e.g., `DiffusionModel(variable_name, mesh, ...)`).
    4. Set initial conditions on `mesh_var` and assign `diffusivity`.
    5. Add boundary conditions (`add_dirichlet_bc`, `add_neumann_bc`).
    6. Run the simulation using `run_simulation(duration, ...)`.
- **Naming:** Follow `snake_case` for functions and variables, and `PascalCase` for classes.
- **Hooks:** Use `register_pre_solve_hook` and `register_post_solve_hook` for custom logic during the simulation loop.

## Key Files
- `src/UWDiffusion/Models.py`: The heart of the library; defines the solver logic.
- `src/UWDiffusion/meshing.py`: Gmsh-based mesh generation functions.
- `src/UWDiffusion/utilities.py`: Visualization and helper functions.
- `Tutorials/`: Essential for understanding library usage and validating results.
