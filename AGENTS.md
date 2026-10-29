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
  utilities.py              Plotting and sampling helpers; time-integrated diffusion length / BC-regime screen.
tests/                      10 files, 103 tests. This is the source of truth for API usage.
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
source /path/to/venv/uw3-dev/bin/activate
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

The package **is** installed (editable) in the venv, so `import UWDiffusion`
works from anywhere in that environment — examples must NOT use
`sys.path.insert`. Tests keep `PYTHONPATH=src` so they pin to the working
tree rather than a stale install. Do not run `pip install -e .` unless asked.

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
x_symbols)` (SymPy), `lasaga_matrix_from_components(names)` (SymPy, builds the
symbols for you), `reduce_diffusion_matrix(...)` (generic closure
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

**Use `lasaga_matrix_from_components(component_names)`** and none of this is
your problem — it builds `xi_1..xi_n` and `D_<component>` correctly in one
call, replacing five lines of hand-rolled boilerplate. `set_diffusion_values`
then accepts either `{"Fe": ...}` or `{"D_Fe": ...}`. Keep `lasaga_matrix` for
custom symbols. The helper is named for its **formulation** because that is a
physical assumption the library cannot make for you — Lasaga mean field is one
closure among several (Darken, a measured matrix), so pass a matrix explicitly
if you want a different one.

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
| `time_integrated_diffusion_length` | **t in Myr, T in degC, P in kbar**, radius in metres; converts to s/K/GPa via pint |

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

## Choosing dt and mesh resolution

The question users ask most is "how fine is fine enough?". The evidence lives in
`docs/multicomponent_diffusion.md`, `examples/convergence_*.py`, and the timestep
docstrings in `Models.py`; this is the operational summary.

### dt is a free accuracy choice, not a stability constraint

Backward Euler is A- and L-stable. Nothing blows up at a large dt — you just
lose accuracy. **Never chase a stability limit; it does not exist.** Choose dt
to control the O(dt) error.

`run_simulation` suggests `dt = time_step_factor * h^2 / kappa_max`, where
`kappa_max` is the spectral radius of the *substituted* matrix sampled over the
mesh (so composition dependence and cross-term signs are included). The default
`time_step_factor=0.1` gives roughly **0.6% error in the benchmarked case;
10x that step gave ~6%**. `min_dt` deliberately overrides the suggestion.

That `h^2` scaling is the point: it keeps time error from swamping the
second-order spatial error. If you set dt by hand and it is much larger than
`h^2/kappa`, your result is a *time-discretisation* answer regardless of how
fine the mesh is.

### Time order must rise with element degree

**Scope: smooth profiles at `degree=2` or above**, where the spatial error is
small enough that time error can dominate. At `degree=1`, or at a sharp front,
the spatial error is much larger and `order=1` is fine — see "Which defaults
when".

This is the finding that changes how you set up a run. Measured against the exact
modal decay (closed form, no solver — see `examples/p_vs_h_projection_study.py`),
at the default heuristic `dt = 0.1 h^2/kappa` and `order=1`:

| h | steps | time err (`order=1`) | time err (`order=2`) | time err (`order=3`) | p=2 spatial err | `order=1` / spatial |
|---|---|---|---|---|---|---|
| 0.05 | 88 | 2.8e-04 | 1.9e-06 | 1.5e-08 | 1.0e-07 | **2685x** |
| 0.025 | 352 | 7.0e-05 | 1.2e-07 | 2.3e-10 | 1.3e-08 | **5389x** |
| 0.0125 | 1408 | 1.8e-05 | 7.5e-09 | 3.6e-12 | 1.6e-09 | **10783x** |
| 0.00625 | 5632 | 4.4e-06 | 4.7e-10 | 9.1e-14 | 2.0e-10 | **21568x** |

(Errors absolute on a modal amplitude of 0.15. `kappa` = 1.1, the spectral
radius of the reduced matrix.)

**With `order=1` and `degree=2` the answer is time error, by three to four
orders of magnitude.** The `time_step_factor=0.1` heuristic is calibrated for
Q1, where the spatial error is O(h^2) and matches O(dt) = O(h^2). At Q2 the
spatial error is an order finer and the same heuristic leaves time error
completely dominant — and the gap *widens* as you refine, so "just refine the
mesh" makes it worse.

Consequences:

- **Raising `degree` without raising `order` is wasted.** The extra spatial
  accuracy is buried under time error. This is the practical answer to the
  p-vs-h question: the cheap fix is not more elements or higher degree, it is
  `order=2` or `order=3`.
- **`order=2` roughly balances** Q2 (time error within ~1–18x of spatial across
  the range). **`order=3` makes spatial dominant**, at which point degree gains
  become realisable.
- The alternative — shrinking `time_step_factor` to ~1e-5 to balance
  `order=1` against Q2 — costs 1000x more steps. Raising `order` is far cheaper.
- The model ramps order up from 1 (`_effective_order`), so the first `order`
  steps are lower order. For short runs the asymptotic rate is not reached.

**This also explains a confusion in the validation.** The convergence study in
`docs/multicomponent_diffusion.md` measures *spatial* rates only because it
applies a Backward-Euler Richardson pair at every mesh, which cancels the
leading O(dt) term. A normal run gets no such cancellation — so do not expect a
typical run to reproduce those rates. The Richardson floor quoted there
(~2.8e-6 at cellSize 0.02) is what remains after the cancellation.

### Measured convergence rates

For smooth, constant-matrix eigenmodes (`docs/multicomponent_diffusion.md`):

| | rate | notes |
|---|---|---|
| 1D spatial (`fd_1d`) | ≈ O(h²) | under grid halving |
| 2D spatial (UW3 Q2) | ≈ O(h²) | measured 1.96–2.13; use *actual* h_eff ratios |
| temporal, `order=1` | ≈ O(dt) | Backward Euler |

`order=2/3` ramps up (`_effective_order`), so the **first steps are O(dt)
regardless of the nominal order**. For short runs this matters more than the
asymptotic rate.

These rates are for smooth profiles. A sharp compositional front converges more
slowly and is where nearly all the error sits.

### Can higher-order elements buy back resolution?

`MulticomponentDiffusionModel(degree=...)` sets the FE basis degree (Q2 by
default). Raising it raises accuracy *without changing h*, and since the
timestep heuristic `dt ~ time_step_factor * h^2 / kappa_max` depends on mesh
spacing and **not** on degree, p-refinement improves the spatial answer without
shrinking the allowed step. h-refinement pays for itself twice (DOF *and* a
quadratically smaller dt).

`examples/p_vs_h_projection_study.py` measures the **approximation-theory floor**
— the L2 projection of the exact radial eigenmode onto piecewise degree-p
polynomials. No solver can beat it, so it isolates the spatial question from
time error and solver noise. Numbers:

| target | p | rate L2 | rate max | max err at h=0.003125 |
|---|---|---|---|---|
| smooth eigenmode | 1 | 2.00 | 1.97 | 8.3e-06 |
| smooth eigenmode | 2 | 3.00 | 2.99 | 6.8e-09 |
| smooth eigenmode | 3 | 4.00 | 3.99 | 4.2e-12 |
| steep front (`tanh`, eps=0.01) | 1 | 1.46 | 1.31 | 1.6e-03 |
| steep front | 2 | 2.86 | 2.41 | 6.2e-05 |
| steep front | 3 | 2.50 | 1.82 | 1.4e-06 |

At **equal total DOF** on the smooth mode, `degree=3` beats `degree=2` by
**25x–200x** in max error. So yes: for a smooth profile, a coarse high-degree
mesh beats a fine low-degree one, by a lot.

**But that is not the binding constraint in this codebase.** The measured 2D
solver rate in `docs/multicomponent_diffusion.md` is **~O(h^2) (1.96–2.13) at
`degree=2`**, while the approximation floor for that same smooth mode at Q2 is
**O(h^3)**. The solver is an order *below* what its elements can represent.
Something else is limiting it — the unstructured gmsh mesh and its angular
scatter, the natural-Neumann boundary treatment, or the radial sampling. Raising
`degree` will therefore **not** deliver the textbook improvement on the current
2D path; it lifts a ceiling that is not the one you are hitting. Fix the mesh
and boundary errors first.

**Correction to a plausible-sounding claim.** The max-norm does *not* hide
higher-order behaviour — the table shows `O(h^3)` and `O(h^4)` in max-norm just
as in L2. If a convergence study reports `O(h^2)` at Q2, that is a property of
the *solve*, not of point sampling. Do not attribute it to the norm.

**Sharp interfaces are not about degree.** Two measurements change the story.
Against a *smooth but steep* `tanh`, higher degree still wins in absolute error
but the margin collapses (2–6x instead of 25–200x) and the fitted rates degrade
(`p=3` max rate 1.82 vs `p=2` 2.41) — so the clean advantage evaporates as the
solution stops being smooth.

Against a **true discontinuity** the finding is sharper and less intuitive:
*every* degree overshoots the physical range, by **10–40%**, and it does not
decay with refinement while an element straddles the jump. `p=3` is not
systematically worse than `p=1` — the worst offender rotates between degrees as
the straddling fraction changes. So "high order oscillates at a front" is the
wrong lesson.

What actually causes the overshoot is the **element that contains the jump**.
The fix is to **align element edges with the interface**, not to choose a
degree. This library already does that by construction —
`generate_garnet_annulus_mesh` places growth shells on grid vertices (see
convention 6) — which is the real protection against unphysical compositions,
and it works at any degree. Composition closure (`Ca = 1 - Mg - Fe`) propagates
any overshoot into the dependent component and can push it negative, so this
matters more here than in a passive tracer problem.

Practical rule: at a growth interface, **mesh-align first**. If the mesh cannot
be aligned, **drop to `degree=1`** — see the measured penalty in "Sharp fronts:
higher degree is actively harmful" below.

**The 1D solver has no degree knob.** `MulticomponentDiffusion1D` is a fixed
vertex-centred FV scheme; h is the only lever. If you need higher order in 1D,
refine or move the problem to 2D.

**`degree` is now exercised at 1, 2 and 3** (`examples/degree_order_sweep.py`
and the CZGM planar matrix), so the range is no longer untested — but nothing
validates `degree > 3`, and PETSc may not accept it on every mesh. Treat higher
degrees as an experiment and re-run the two-run check below.

Practical rule: *smooth profile* → `degree=2, order=2`; `degree=3` is a large
theoretical win but unreachable behind the mesh floor on the current 2D path.
*Sharp front* → `degree=1`, `order=1`, then refine h at the front.
*1D* → h only. See "Which defaults when" below.

### What the solver sweep actually shows

`examples/degree_order_sweep.py` runs the real 2D UW3 solver over
`degree in {1,2,3}` x `order in {1,2,3}` x `cellSize in {0.08,0.04}` (18 runs,
~13 min) against the exact eigenmode. It settles the theory above — and
overturns part of it.

**The error decomposes cleanly**, and the time term is predicted almost exactly:

| degree | cellSize | measured `order=1` | measured `order=2` | difference | exact BE time error | ratio |
|---|---|---|---|---|---|---|
| 2 | 0.08 | 8.847e-04 | 6.612e-04 | 2.235e-04 | 2.255e-04 | **0.991** |
| 2 | 0.04 | 2.211e-04 | 1.669e-04 | 5.421e-05 | 5.434e-05 | **0.998** |
| 3 | 0.08 | 8.931e-04 | 6.696e-04 | 2.234e-04 | 2.255e-04 | **0.991** |
| 3 | 0.04 | 2.217e-04 | 1.675e-04 | 5.422e-05 | 5.434e-05 | **0.998** |

So `total ~= mesh_floor(h) + time_error(dt, order)`, with the time error known
in closed form. (`degree=1` does not decompose this way — its own spatial error
is too large.)

**But there is a third term the projection study could not see.** Once the time
error is removed (`order=2`), the residual is `~0.136 * h_eff^2` — **O(h^2),
and completely insensitive to `degree`** (0.1360 vs 0.1365 for degree 2 vs 3 at
cellSize 0.04). It sits *far* above the Q3 approximation floor (~2500x at
cellSize 0.04). Its most likely source is the unstructured gmsh mesh breaking
radial symmetry, though the initial-condition projection and the angular
sampling are also candidates. Either way: **it is not element degree, and it is
not the timestep.**

**Measured verdicts:**

- **`degree=3` is a pure loss.** 1.83-1.90x the cost for 0.3-1.3% *worse* error
  than `degree=2`. Never useful on this mesh. This refutes the theoretical
  argument above: the Q3 approximation win is unreachable behind the mesh floor.
- **`order=3` is a pure loss.** Identical accuracy to `order=2` (to 3 decimal
  places) and 5-8% slower. The BDF2 time error is already below the mesh floor,
  so BDF3 refines nothing.
- **`order=2` is the sweet spot.** 5-8% more cost than `order=1`, 25-33% less
  error, and it removes the time term entirely.
- **Cost is driven by steps, which scale as `h^-2`.** Halving `cellSize` 0.08 ->
  0.04 is 4x the steps (138 -> 574) and ~14x the wall time (4.5s -> 65s) for 4x
  less error.

Pareto frontier (no config is both faster and more accurate) is topped by
**`degree=2, order=2`** at each mesh level. Cheapest run reaching 1e-3 is
`degree=2, order=1, cellSize=0.08` at 4.5 s; for 3e-4 it is
`degree=2, order=1, cellSize=0.04` at 65 s.

**Quadrature (`qdegree`) is a non-issue here.** The mesh is triangular, so the
basis is `Pp`, and on a triangle the mass-matrix integrand has total degree
`2p` — a `qdegree=2` rule is only exact for `P1`, so `degree=2` and `degree=3`
are formally under-integrated. Measured, it does not matter: sweeping
`qdegree in {2,4,6,8}` at fixed degree leaves the error identical to **four
significant figures**, while runtime grows **2-6x** (degree 2 at cellSize 0.08:
4.8 s at qdegree 2, 26.8 s at qdegree 8). Keep the default `qdegree=2`; it also
rules quadrature out as the source of the `0.136 h^2` floor. If you go beyond
`degree=3`, re-check rather than assume.

**Scope: this recommendation is for SMOOTH profiles only** (eigenmodes,
well-mixed grains). See "Which defaults when" below — sharp growth interfaces
get the opposite answer.

**For smooth problems: `degree=2, order=2`.** Do not raise degree to 3, and do
not raise order to 3. If you need more accuracy, spend it on mesh
quality/symmetry or on h-refinement — those are the binding constraints.

### Bound your own error with two runs

You rarely need a full convergence study. The cheap version:

1. Halve dt at fixed mesh. The change is your **time error**.
2. Halve h at fixed (small) dt. The change is your **space error**.
3. If the two are within a factor of ~2, you are balanced — stop. Otherwise
   refine whichever dominates.

Richardson extrapolation removes the leading term: with order p, use
`X_rich = (2^p * X(fine) - X(coarse)) / (2^p - 1)` — p=2 spatial, p=1 for
`order=1` time. The convergence examples do exactly this (a Richardson pair in
dt at every mesh) so the spatial rate is measured free of time error.

### Choosing `dt` along a PTt path

For a growth-and-diffusion model with retrograde rim exchange, the timestep you
need is set by **which limb you are on**, not by the hottest point on the path.
Measured on a representative path (P 1-14 kbar, T 500-780 degC, t 0-150 Myr)
against a fine Backward-Euler reference — see
`examples/min_dt_floor_study.py` (prograde) and
`examples/retrograde_rim_dt_study.py` (retrograde rim):

| limb | feature | error @ `dt=1 Myr` | error @ `dt=0.1 Myr` |
|---|---|---|---|
| prograde (sharp step at peak T) | step smearing | **1.1e-2** | 2.4e-3 |
| retrograde (rim held at matrix composition) | boundary layer growing inward | **3.1e-4** | 3.0e-5 |

The retrograde limb is **~35x more forgiving** at the same timestep. A `min_dt`
floor of `1 Myr` is fine for recording retrograde zoning (0.03% error) and it
is cheap (1.7 s vs 13 s at `0.1 Myr`). The prograde transient is the case that
needs care.

Why: the prograde case has a **sharp feature diffusing quickly at peak T**, and
Backward Euler over-smooths it. The retrograde case is boundary-layer growth in
a cooling, slowing system — inherently well-behaved for an L-stable scheme.

Two traps when measuring this, both of which cost time here:

- **Compare during the transient, not the end state.** Everything homogenises
  and every timestep lands on the same flat answer — an end-state comparison
  reports roundoff and says nothing at all. Snapshot along the way and take the
  max over snapshots. The whole signal in the prograde case lived in the first
  ~20 Myr.
- **Know your reference's own uncertainty.** Run the reference at two
  timesteps and report the difference. In the retrograde case the fine
  reference was self-consistent to only **8e-7** (rim resets are less clean
  than a closed system), so results below that are noise — three of the four
  timesteps tested were at or near that floor.

Error also accumulates at the **cold** end of the path, not the hot end: `D`
collapses ~100x while the timestep stays fixed relative to the physics, so the
boundary layer keeps growing into an ever-slower medium. Check the final
snapshot, not just the peak-T window.

**Cache the diffusivity by time.** `CG92 .at()` costs ~62 us and is called 4x
per matrix evaluation, at *every face, every step* — even though T and P depend
only on model time and not on position. Memoising the tracer vector on the time
value gave a **~90x speedup** on a 201-point 1D run (90 ms/step -> 1 ms/step).
Any model supplying a callable diffusivity over a PTt path should do this.

### Sharp fronts: higher degree is actively harmful

At a compositional step (growth shells, an initial Heaviside) the error
concentrates in a band of width ~dr. Report the front and the interior
separately — `convergence_1d_2d_garnet_growth.py` splits `err_point` /
`err_front` / `err_interior` and you should too. A single global max-norm tells
you almost nothing about the profile you actually care about.

Measured on the CZGM planar growth problem (`Tutorials/Diffusion/CZGM_garnet_model_planar.ipynb`),
all four `(order, degree)` combinations against both the 1D finite-volume leg
and the CZGM MATLAB reference — mean of `max |diff|` over Fe, Mg, Mn, Ca:

| order | degree | vs 1D FD | vs CZGM |
|---|---|---|---|
| 1 | 1 | **4.8e-04** | 3.2e-03 |
| 2 | 1 | 1.8e-03 | **1.9e-03** |
| 2 | 2 | 2.3e-03 | 3.9e-03 |
| 1 | 2 | 2.6e-03 | 5.1e-03 |

**`degree=1` beats `degree=2` in all eight component x reference comparisons.**
This is a robust ~2x penalty for `degree=2`, not noise — and it is stronger than
"degree does not help" at a front: higher degree is *worse*. Against an
unaligned compositional step the element straddling the jump has more freedom to
overshoot. The 1D grid is aligned with the growth shells by construction
(`x_grid_m = Rr * 1e-6`); if the UW3 mesh is not, that is the mechanism and the
fix is meshing (see convention 6), not solver settings.

**`order` is NOT robust** — it flips with the reference. `order=2` wins against
the CZGM target but loses against the 1D FD leg, which is itself Backward Euler
at `0.05 Myr` and so agrees with `order=1` by construction (convention 9). A
knob whose sign depends on which imperfect reference you compare against is not
worth tuning, so the sharp-front default is the simplest one that is not
robustly worse.

### Which defaults when

| problem | `degree` | `order` | why |
|---|---|---|---|
| **smooth profile** (eigenmodes, well-mixed, no steps) | `2` | `2` | `degree=3` and `order=3` are both pure losses; `order=2` removes the time term nearly for free |
| **sharp growth interfaces** (zoning, growth shells, Heaviside ICs) | `1` | `1` | `degree=1` robustly beats `degree=2` at an unaligned front; `order` is not robust, so take the simpler and cheaper one |

**The library default is `degree=1, order=1`** — the sharp-interface row, chosen
because growth/zoning problems are the common case here and it is also the
cheapest. Smooth problems must opt in to `degree=2, order=2` explicitly. The
defaults are guarded by `test_default_degree_and_order`.

These are genuinely opposite answers, so **look at the problem before copying a
setting**. A garnet growth model is the sharp case; a relaxation or tracer
decay test is the smooth case. Mixing them — for instance benchmarking a growth
model against an eigenmode-derived default — is how the `degree=2, order=2`
recommendation came to be over-applied.

### A 1D–2D mismatch is often not solver error

The 1D unknown is a **control-volume average**; the 2D field is a **point value
on a Q2 basis**. Away from a step they agree; inside a step they are different
objects and differ by construction. Compare control-volume-averaged 2D against
1D (`err_cv` in the growth convergence script) before concluding either solver
is wrong.

Two further caveats in that script worth reading before interpreting any rate:
the growth shells move slightly as the mesh is refined (shell radius is a
discretisation parameter there, not physics), and the fitted rate is therefore
an *upper bound* on the true spatial rate for a fixed interface set.

### Diagnostics to run anyway

Cheap and already in the suite's vocabulary: per-component mass drift under
homogeneous Neumann (measured 2.3e-14 in 1D, 5.2e-9 in 2D), closure residual
`sum(x_i) - 1` at roundoff, and SNES convergence flags per step. A drifting
mass is a bug, not a discretisation artefact.

---

## Choosing solver tolerance and preconditioner

Measured on a genuinely **nonlinear**, time-dependent problem
(`examples/solver_tolerance_sweep.py`): a four-component Lasaga matrix whose
entries depend on the composition unknowns through `xi_1..xi_4`, a synthetic
PTt path re-binding the diffusivities every step, and a rim Dirichlet condition
rebuilt every step. Fixed discretisation, 240 steps, self-consistency against
the tightest run.

**Why the operator has to be nonlinear for this question to mean anything.**
The constant-coefficient variant of the same script reports exactly **1 Newton
iteration at every tolerance from 1e-4 to 1e-10**, with bit-identical answers:
a linear operator converges in one step whatever `snes_atol` is, so there is no
knee to find. All the eigenmode verification in this repo is that linear case.
Solver tolerance is a *nonlinear* question and does not transfer from it.

### The knee is `1e-5`, not `1e-8`

| `snes_atol` | its/solve | Δ vs converged |
|---|---|---|
| 1e-03 | 2.00 | 6.96e-04 |
| 1e-04 | 2.00 | 6.96e-04 |
| **1e-05** | **3.00** | **2.67e-08** |
| 1e-06 | 3.00 | 2.67e-08 |
| 1e-07 | 3.00 | 2.67e-08 |
| 1e-08 | 3.65 | 0 |

There is a hard cliff between `1e-04` and `1e-05`, and **nothing below `1e-05`
buys anything**: `1e-08` costs 35% more Newton iterations and 4% more
wall-clock for a change of 2.7e-08. Use `snes_rtol = snes_atol = 1e-5`.

### Ordering rule

The linear solve must be *tighter* than the nonlinear one:

```
ksp_rtol  <   snes_rtol
ksp_atol  <<  ksp_rtol * ||b||     (a floor, not a criterion)
snes_atol <=  snes_rtol * ||F0||   (a floor, not a criterion)
```

`atol` and `rtol` are not ordered against *each other*: `rtol` is dimensionless
(a required fractional reduction) and `atol` is in residual units (an absolute
floor). Numerically `atol` normally sits far below `rtol` so that `rtol` binds.
The ordering that matters is across levels.

Inverting it (`ksp_rtol` looser than `snes_atol`) caps how far Newton can get,
because the linear error floor exceeds the requested nonlinear tolerance and
Newton grinds against it.  Worth getting right as good practice.

It was an early suspect for a `SNES_DIVERGED_LINE_SEARCH` in a production
garnet run and **is not the cause** — the sweep reproduced the inverted
hierarchy without failure, and the divergence turned out to be GAMG against an
indefinite matrix (below).

### GAMG is structurally invalid for this matrix

**`pc_type=gamg` DIVERGES on the production garnet operator while a direct
solve converges.** This is not a tuning problem, it is a mismatch of assumptions:

* The Lasaga reduced matrix is **strongly non-symmetric** and its **symmetric
  part is indefinite everywhere** — a negative eigenvalue across the whole
  composition range.  With a fast-Mn / slow-Ca tracer set the asymmetry is
  large (`M[1,2]` two orders of magnitude from `M[2,1]`), because the rank-1
  term carries `(D_j - D_dep)` weighted by a tiny `X_Mn`.
* GAMG constructs its coarse grid assuming an SPD-like operator.  On this
  matrix the interpolation is structurally wrong, so the preconditioner is
  invalid rather than merely poor: the linear solve fails, Newton receives a
  garbage direction, and `snes_type=newtonls` reports
  **`SNES_DIVERGED_LINE_SEARCH` (`reason=-3`) after a couple of iterations**.

Note this is *not* a conditioning problem — `cond(M)` is only ~20-40.  The
matrix is well-conditioned; GAMG is simply the wrong tool for its structure.

**Use `bjacobi` with `sub_pc_type=lu`.**  It converges where GAMG diverges.
On a single rank `bjacobi` is one block, so this is exactly `pc_type=lu` —
but unlike bare `pc_type=lu` it still works when the run is distributed.
Bare `pc_type=lu` also has two other costs: PETSc's default LU backend is
weak (use `mat_solver_type=mumps` or `umfpack` if you want a real one), and
it refactorises every timestep when diffusivities are rebound each step.

`bjacobi`/`asm` with `ilu` is the cheaper fallback and is robust for
non-symmetric operators, but it is approximate — prefer `lu` subdomains while
they fit in memory.

### Preconditioner speed is problem-size dependent

On the small study problem (`examples/solver_tolerance_sweep.py`, ~25 s per
run) the four preconditioners agreed to machine precision and differed by
**1.03x** — that is run-to-run noise, not a ranking.  The study has no
resolution to rank preconditioners and should not be quoted as evidence that
"bjacobi is fastest".  On the production-size model the ranking is set by
robustness (above), not by that table.

### Set diffusivities before `run_simulation`

The timestep heuristic evaluates the matrix at the *top* of the step loop,
before any pre-solve hook runs. If the parameter symbols are still
unsubstituted it now raises — it used to return `0.0`, making `max_time_step`
infinite so the entire run took one step and reported success. Bind them up
front; a pre-solve hook is too late for the *first* estimate.

---

## Testing

`tests/` is the source of truth for API usage. 103 tests, ~65 s.

| file | covers |
|---|---|
| `test_fd_1d.py` | measures, closure, pins, BCs, validation |
| `test_verification.py` | exact eigenmodes and convergence for the 1D solver |
| `test_2d_verification.py` | UW3 models vs exact solutions; generic component count |
| `test_1d_2d_coupling.py` | the two solvers against *each other*; mesh shell labels; history sync |
| `test_benchmarks.py` | CZGM benchmarks against reference CSVs; growth/centre options |
| `test_tracer_diffusivities.py` | unit convention, transcription, primary-source errors |
| `test_diffusion_matrix.py` | `lasaga_matrix_from_components` and the `xi_k` naming contract |
| `test_diffusion_length.py` | time-integrated diffusion length, unit conversion, BC regimes |
| `test_decay_ingrowth.py` | decay/ingrowth source terms |
| `test_time_integration_options.py` | BDF order, variable dt, the timestep heuristic |

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
- Underworld3 source lives in a sibling checkout of
  [underworldcode/underworld3](https://github.com/underworldcode/underworld3)
  (next to this repository).
  If you change it, rebuild with `./uw build` — never `pip install -e .` there.
