# AGENTS.md — UWDiffusion

Python library for diffusion problems in geochemistry and petrology, built on
[Underworld3](https://github.com/underworldcode/underworld3) (PETSc/SymPy FEM).

Two solver paths share the same physics:

- **2D/3D FEM** via Underworld3 (`Models.py`) — full geometry, the production path
- **1D radial FV** (`fd_1d.py`) — fast, used for inversions and for validating
  the 2D path

Read this before touching `src/`. Several conventions below are load-bearing and
fail *silently* if broken.

---

## Repository layout

```
src/UWDiffusion/
  Models.py                 2D/3D UW3 solvers (DiffusionModel, MulticomponentDiffusionModel,
                            DiffusionDecayIngrowthModel). Monolithic SNES_MultiComponent.
  fd_1d.py                  1D vertex-centred finite-volume solver (planar/cylindrical/spherical).
  diffusion_matrix.py       Lasaga mean-field matrix, numeric + symbolic. ONE home for the formula.
  tracer_diffusivities.py   Published Arrhenius parameters (CG92, Loomis 1985).
  meshing.py                Gmsh mesh generation (boxes, garnet annuli with shell labels).
  utilities.py              Plotting and sampling helpers.
tests/                      8 files, 81 tests. This is the source of truth for API usage.
examples/                   Runnable jupytext scripts, incl. cross-solver benchmarks.
Tutorials/                  Jupyter notebooks.
docs/                       Prose documentation.
development/                Local scratch. Gitignored — never relied on by anything.
```

`development/` is in `.gitignore`. Do not import from it, and do not assume
anything in it survives.

---

## Environment and commands

The UW3/PETSc/MPI stack lives in a dedicated venv. **Activate it first** — a
bare `import underworld3` from a plain shell fails with "cannot load MPI library".

```bash
source /Users/benknight/software/venv/uw3-dev/bin/activate
```

```bash
# tests (the command that matters — run it before and after any change)
PYTHONPATH=src python -m pytest tests/ -q

# one file
PYTHONPATH=src python -m pytest tests/test_fd_1d.py -q

# examples run from the repo root
python examples/compare_1d_2d_garnet_growth.py
```

**`nbstripout` is required.** A local git clean filter strips notebook outputs
before commit; without the module every notebook operation fails. If
`git diff` errors with `No module named nbstripout`, `pip install nbstripout`.

The package is **not** installed in the venv. Use `PYTHONPATH=src` (tests) or
`sys.path.insert` (examples). Do not `pip install -e .` unless asked.

**Scripts use jupytext `# %%` cell markers**, not `if __name__ == "__main__":`
blocks. Follow that when adding examples.

---

## Module map

### `Models.py` — 2D/3D Underworld3 solvers

One `SNES_MultiComponent` solve carries all independent components, so the
cross-diffusion Jacobian block is implicit and exact. An earlier per-component
`Poisson` split is gone — do not reintroduce it; its `dF_i/d(grad x_j)` block was
structurally zero and the coupling was only approximate.

`MulticomponentDiffusionModel(component_names, mesh, diffusion_matrix=D_sym,
degree=2, order=1)` where `D_sym` is a SymPy matrix from `lasaga_matrix`.

### `fd_1d.py` — 1D finite volume

`MulticomponentDiffusion1D(grid, diffusivity, geometry=..., components=...,
dependent_component=..., bcs=..., initial_composition=...)`.

- `geometry` is `alpha` in `r^alpha dr`: `"planar"`=0, `"cylindrical"`=1,
  `"spherical"`=2. A radially symmetric field on a UW3 Cartesian annulus is
  **`"cylindrical"`**, not spherical.
- `run_simulation(dt, total_time)` and `run_to_time(t, dt)` are both Backward
  Euler. There is no BDF2/3 here — if you compare against UW3, match `order=1`.

### `diffusion_matrix.py` — the reduced matrix

`lasaga_reduced_diffusivity(...)` (numeric), `lasaga_matrix(d_symbols,
x_symbols)` (SymPy), `reduce_diffusion_matrix(...)` (generic closure
reduction). Both solvers consume these; do not re-derive the formula inline.

### `tracer_diffusivities.py` — published parameters

`TracerDiffusivity` dataclass with `.at(T_K, P_GPa)`. `CG92` and `LOOMIS_1985`
are the two retained calibrations.

---

## Contracts and conventions

These are the things that break quietly.

### 1. Symbol naming in the diffusion matrix is a hard contract

`MulticomponentDiffusionModel` resolves composition symbols **by regex on the
symbol name**, `^xi_(\d+)$`. Nothing registers them.

- `xi_k` for `k <= n_independent` → solved field `k`
- `xi_n` (the *last*, when closure is on) → `1 - sum(x_1..x_{n-1})`

Name them anything else and they are left unsubstituted — no error, just a
matrix full of free symbols. The parameter symbols (`D_Fe`, …) are bound by
`set_diffusion_values` using their names.

### 2. The Lasaga matrix assumes isovalent species

`lasaga_matrix` implements Lasaga (1979) with the charge factors
cancelled. That is **exact only when every species carries the same charge** —
true for Fe/Mg/Mn/Ca on the garnet dodecahedral site (all divalent), **wrong for
mixed valence** (Fe²⁺/Fe³⁺). If you need mixed valence, use the charged form.

It is *not* the classical Darken relation `D = X_B D_A + X_A D_B`. Both close
the unknown off-diagonal Onsager coefficient, but partition the flux-conserving
drift differently. Do not test one against the other.

### 3. Units

| context | convention |
|---|---|
| `fd_1d` grid / time / diffusivity | metres / seconds / m²/s |
| `tracer_diffusivities` | `log_D0` (D0 in m²/s), `dE` kJ/mol, `dV` cm³/mol, T in K, **P in GPa** |

`1 GPa · cm³/mol = 1 kJ/mol` exactly — that identity is why this convention is
used and it removes the legacy `41.84` bar·cm³→cal factor. `pint` enforces it at
runtime.

`.at()` returns **m²/s**. Labelling an m²/s float as cm²/s is a silent 10⁴ error.

### 4. `run_simulation(duration=...)` is an absolute target

Not an increment. `run_simulation(duration=0.02)` advances *to* t = 0.02.
Passing a delta quietly stops early — the profiles still look plausible.

### 5. Sync history after any manual field overwrite

Growth stamps new material straight into `mesh_vars`. The time-derivative
history still holds the pre-stamp state, so the next solve sees a spurious
`dC/dt` and drifts. Call `model.sync_history_to_state()` afterwards.

### 6. `generate_garnet_annulus_mesh` labels the outermost shells

`growth_shell_count=G` labels `radii_nd[-G:]` as `shell_0..shell_{G-1}`, with
`shell_0` the *innermost* of those. Getting this pins the wrong radii and the
model still solves.

It builds a **full disk from the origin** (all shells embedded), not an annulus —
the `Centre` label is a zero-dimensional point marker carrying no boundary
condition.

### 7. The grain-centre boundary condition is a modelling choice

`MulticomponentDiffusion1D(centre="zero_flux"|"mirror")`.

- `"zero_flux"` (default) — true symmetry plane; the centre control volume evolves
- `"mirror"` — reproduces CZGM MATLAB (`C(1)=C(2)`, node 0 frozen)

These differ measurably at the core. `"mirror"` matches the CZGM reference to
~1e-4; `"zero_flux"` is the physically defensible choice. Default is
`"zero_flux"`. Do not change it silently.

### 8. Ca has no measured tracer diffusivity

Neither CG92 nor Loomis publish one. `D_Ca = D_Mn / 20` is a **package
convention** inherited from CZGM, exposed as `CG92_CA_D_MN_RATIO` and
`cg92_ca()`. Never cite it to Chakraborty & Ganguly.

### 9. Match the reference formulation before "fixing" a discrepancy

`examples/matlab_planar_replica.py` is a literal translation of the CZGM MATLAB
reference and reproduces `combined_C_final_cart.csv` to ~1e-5. If a benchmark
disagrees, check against it before changing solver code — the gap has so far
always been a boundary condition or a driving convention, not the operator.

---

## Testing

`tests/` is the source of truth for API usage. 81 tests, ~60 s.

| file | covers |
|---|---|
| `test_fd_1d.py` | measures, closure, pins, BCs, validation |
| `test_verification.py` | exact eigenmodes and convergence for the 1D solver |
| `test_2d_verification.py` | UW3 models vs exact solutions; generic component count |
| `test_1d_2d_coupling.py` | the two solvers against *each other*; mesh shell labels; history sync |
| `test_benchmarks.py` | CZGM benchmarks against reference CSVs; growth/centre options |
| `test_tracer_diffusivities.py` | unit convention, transcription, primary-source errors |
| `test_decay_ingrowth.py` | decay/ingrowth source terms |
| `test_time_integration_options.py` | BDF order, variable dt |

Tolerances in these tests are calibrated to *measured* values with documented
headroom. If one fails, the regression is usually real — read the docstring
before loosening an assertion.

Test data lives in `tests/data/` and `Tutorials/Diffusion/CZGM_data/`.

---

## Working on this repo

- **Run the suite before and after any change.** It is fast enough to be free.
- **One source of truth for a formula.** If you need the Lasaga matrix or an
  Arrhenius evaluation, import it — do not re-derive inline. Duplicated physics
  is how the `darken_*` misnomer survived.
- **Cite primary sources in docstrings**, not secondary tabulations. Two errors
  in the 2018 *Acta Geochimica* review were only caught by going back to the
  papers (misreported Mg activation volume; single-run point values printed in a
  `logD₀` column).
- **Flag assumed values as assumed.** Every registry entry carries a `source`
  field; a convention should read as a convention.
- **Notebooks are not the test suite.** `Tutorials/` is teaching material and
  gets re-executed by hand; `tests/` is what guards behaviour.
- Underworld3 source lives at `/Users/benknight/Documents/Research/GitHub/underworld3`.
  If you change it, rebuild with `./uw build` — never `pip install -e .` there.
