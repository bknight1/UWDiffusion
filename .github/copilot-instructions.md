
# UWDiffusion – Copilot AI Agent Instructions

## Project Overview
UWDiffusion is a Python library for solving diffusion problems in geoscience workflows (including U-Pb zircon applications) on top of [Underworld3](https://github.com/underworldcode/underworld3). It provides high-level model wrappers while still exposing Underworld3 objects for low-level customization.

## Current Architecture (source of truth)
- **`src/UWDiffusion/Models.py`**: Main simulation models
  - `DiffusionModel`: single-component diffusion
  - `DiffusionDecayIngrowthModel`: coupled parent/daughter diffusion + decay/ingrowth
  - `MulticomponentDiffusionModel`: coupled multicomponent diffusion using symbolic diffusion matrices
- **`src/UWDiffusion/meshing.py`**
  - `generate_2D_mesh_from_points(...)`: builds gmsh mesh + dynamic `Boundary{i}` tags
- **`src/UWDiffusion/utilities.py`**
  - plotting helpers (e.g. Terra-Wasserburg)
  - spot sampling / plotting helpers
  - Adams-Moulton flux helper (`_adams_moulton_flux`)

> Note: keep instructions aligned to existing files only.

## Core Usage Patterns
- **Non-dimensionalization first** (required): configure scaling coefficients before setting model parameters.
  ```python
  scaling = uw.scaling.get_coefficients()
  scaling["[length]"] = 100 * u.micrometer
  scaling["[time]"] = (100 * u.micrometer) ** 2 / (diffusivity * u.meter**2 / u.second)
  ```
- **Dimensional inputs**: diffusivities and times are typically passed as Pint quantities and converted internally via `uw.scaling.non_dimensionalise(...)`.
- **Boundary conditions**: use boundary names defined on the mesh (`Boundary0`, `Boundary1`, ... for generated polygon meshes).
  ```python
  model.add_dirichlet_bc({"Boundary0": 0.0, "Boundary2": 1.0})
  ```
- **Hooks**: models support `register_pre_solve_hook(...)` and `register_post_solve_hook(...)` for sampling, output, and diagnostics during stepping.
- **Time stepping**: models expose `run_simulation(duration, max_dt=None, min_dt=None, time_step_factor=...)` with CFL-style control.
- **MPI-safe output**: gate console/file output with `if uw.mpi.rank == 0:`.

## Multicomponent-Specific Guidance
- Use `MulticomponentDiffusionModel(..., diffusion_matrix=..., diffusion_values=...)` with SymPy symbols in the matrix and values supplied via `set_diffusion_values(...)`.
- Last component is implicit (`x_n = 1 - Σx_i` over independent components).
- You may set values by symbol or symbol-name string keys; dimensional Pint values are supported.

## Tutorials & Notebooks (current)
- `Tutorials/Diffusion/Diffusion-benchmark.ipynb`
- `Tutorials/Diffusion/MC_diffusion-garnet_example.ipynb`
- `Tutorials/Diffusion-decay-ingrowth/zircon-U-Pb-example.ipynb`
- `Tutorials/Diffusion-decay-ingrowth/zircon-U-Pb-dual_growth-example.ipynb`

## Installation & Dependencies
- Underworld3 must be installed first (see UW3 docs).
- Package install:
  ```bash
  pip install .
  ```
- Runtime code paths currently rely on: `underworld3`, `gmsh`, `sympy`, `numpy`, `matplotlib`, and `scipy`.
- Packaging metadata in `setup.py` is minimal and currently lists only `underworld3`; avoid assuming packaging metadata is exhaustive.

## Project Conventions
- Common scales in tutorials: length in μm, time in Myr.
- Output folders are parameterized and usually written under notebook-local `output/` directories.
- Prefer reproducible naming that encodes key parameters (temperature, duration, mesh size, order/CFL).

## Agent Editing Guidance for this Repo
- Keep changes surgical and consistent with existing UWDiffusion API style.
- Preserve dimensional↔non-dimensional conversion patterns.
- When adding examples/docs, match existing notebook idioms (`uw.options.getReal`, explicit scaling setup, rank-0 guarded I/O).
- Validate tutorial/file names against the workspace before referencing them in docs.

---
If uncertain, inspect the relevant tutorial notebook in `Tutorials/` and the corresponding model implementation in `src/UWDiffusion/Models.py`.
