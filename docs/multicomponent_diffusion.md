# Coupled multicomponent diffusion setup and validation

This guide describes setup patterns and validation results for the genuinely
coupled 1D and 2D multicomponent solvers. In the example system, Mg and Fe are
independent and Ca is reconstructed by closure. The examples use unitless
values; configure Underworld3 scaling before assigning physical diffusivities
and times.

## Equation and closure

For `N` components, both multicomponent models solve the first `N-1`
independent components. The final component is dependent and is reconstructed
by

\[
X_{\mathrm{dep}} = 1 - \sum_{i\ne\mathrm{dep}} X_i.
\]

For radial fields, the independent-component PDE is

\[
\frac{\partial X_i}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}
\left(r\sum_j D_{ij}\frac{\partial X_j}{\partial r}\right),
\]

which is also the 2D Cartesian-annulus equation restricted to radial symmetry.
The tested reduced diffusion matrix was

\[
D = \begin{bmatrix}1.0 & 0.3\\0.2 & 0.5\end{bmatrix}.
\]

Its off-diagonal entries produce cross-diffusion: this system is not equivalent
to running two independent scalar diffusion models. With
`components = ["Mg", "Fe", "Ca"]`, Ca is dependent and `D` is the reduced
2-by-2 matrix for Mg and Fe.

Imports for the examples:

```python
import numpy as np
import sympy as sp
import underworld3 as uw

from UWDiffusion import MulticomponentDiffusion1D, MulticomponentDiffusionModel
```

If running from a source checkout rather than an installed UWDiffusion package,
install it in the active Underworld3 environment (for example, `pip install -e .`
from the repository root).

## 1D coupled finite-volume model

The 1D API accepts the numeric reduced diffusion matrix directly. The initial
composition array has shape `(number_of_grid_points, number_of_components)`;
provide all components with the dependent one consistent with closure.

```python
r_inner, r_outer = 0.5, 1.0
r = np.linspace(r_inner, r_outer, 51)
mode = np.cos(np.pi * (r - r_inner) / (r_outer - r_inner))
mg = 0.50 + 0.05 * mode
fe = 0.25 + 0.02 * mode
ca = 1.0 - mg - fe
initial = np.column_stack((mg, fe, ca))

D = np.array([[1.0, 0.3], [0.2, 0.5]])  # reduced matrix for Mg, Fe
components = ["Mg", "Fe", "Ca"]  # final component is dependent
model_1d = MulticomponentDiffusion1D(
    grid=r,
    diffusivity=D,
    geometry="cylindrical",
    components=components,
    bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    initial_composition=initial,
)
result_1d = model_1d.run_simulation(dt=0.001, total_time=0.02)
```

The final component in `components` is dependent by default. You can select
another with `dependent_component=...`; ensure the matrix and composition use
the corresponding independent-component ordering. The 1D model defaults to
zero-flux boundaries if `bcs` is omitted.

## 2D coupled finite-element model

The 2D API takes a SymPy matrix of symbolic diffusivities and a mapping from
symbol names to numeric (or appropriately scaled dimensional) values. The
matrix shape is `(N-1, N-1)`. Initialize every mesh variable consistently;
the last component is then maintained by closure during evolution. With no
boundary constraints on the annulus, both rims have natural zero normal flux.

```python
r_inner, r_outer = 0.5, 1.0
dt, final_time = 0.001, 0.02
components = ["Mg", "Fe", "Ca"]

mesh = uw.meshing.Annulus(
    radiusOuter=r_outer,
    radiusInner=r_inner,
    cellSize=0.025,
    qdegree=2,
)
D_symbolic = sp.Matrix(
    [
        [sp.Symbol("D_MgMg"), sp.Symbol("D_MgFe")],
        [sp.Symbol("D_FeMg"), sp.Symbol("D_FeFe")],
    ]
)
model_2d = MulticomponentDiffusionModel(
    components, mesh, diffusion_matrix=D_symbolic, degree=2, order=1
)
model_2d.set_diffusion_values(
    {"D_MgMg": 1.0, "D_MgFe": 0.3, "D_FeMg": 0.2, "D_FeFe": 0.5}
)

coords = model_2d.mesh_vars[0].coords
r_nodes = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
mode = np.cos(np.pi * (r_nodes - r_inner) / (r_outer - r_inner))
mg = 0.50 + 0.05 * mode
fe = 0.25 + 0.02 * mode
ca = 1.0 - mg - fe
initial = (mg, fe, ca)
with mesh.access(*model_2d.mesh_vars):
    for mesh_var, values in zip(model_2d.mesh_vars, initial):
        mesh_var.data[:, 0] = values

model_2d.run_simulation(duration=final_time, min_dt=dt, max_dt=dt)
```

The 1D model also accepts composition-dependent callable diffusivity matrices
and uses Picard iteration. The 2D model supports composition-dependent
symbolic entries; refer to [`Models.py`](../src/UWDiffusion/Models.py) for the
symbol conventions and solver options. These composition-dependent cases were
not part of the validation reported here.

## Matched single-run benchmark

The test used the annulus `0.5 <= r <= 1.0`, `D=[[1.0, 0.3], [0.2, 0.5]]`,
homogeneous-Neumann boundaries, Backward Euler (`dt=0.001`, final time
`0.02`), and a radial eigenmode of the diffusion matrix. It compared Mg, Fe,
and closure-derived Ca profiles on common radii.

| Component | 1D–2D max difference | 1D max error vs analytical | 2D max error vs analytical |
|---|---:|---:|---:|
| Mg | `2.032e-4` | `1.168e-3` | `9.800e-4` |
| Fe | `4.036e-5` | `3.894e-4` | `4.204e-4` |
| Ca (closure) | `2.037e-4` | `1.557e-3` | `1.400e-3` |

These values show close agreement for this coupled case at the stated
resolution; they are not error bounds for other matrices or compositions. The
1D finite-volume and 2D Q2 finite-element spatial errors differ, and the 2D
mesh produces small angular variations. See the
[comparison script](../examples/compare_1d_2d_cylindrical_multicomponent.py)
for the exact eigenmode initialization and sampling details.

## Convergence-study findings

The convergence study tested **both** eigenmodes of the reduced matrix
(`lambda=1.1` and `lambda=0.4`) and measured Mg, Fe, and Ca against the
analytical solution.

| Refinement | Tested range | Observed trend |
|---|---|---|
| 1D spatial | 26–201 vertices; `dr=0.02` to `0.0025` | Approximately second-order under grid halving for both modes. One intermediate rate for the slower mode is high due to error cancellation; rates before and after are near 2. |
| 2D spatial | `cellSize=0.08` to `0.02` | Approximately second-order for the reported radial-profile error; measured rates are about 1.96–2.13 across modes/components. Actual effective mesh-spacing ratios were used. |
| Temporal, both methods | `dt=0.002` to `0.00025` | Consistent with first-order Backward Euler. Rates against the finite same-mesh reference rise at the finest steps; error divided by `(dt - dt_ref)` stays nearly constant. |

The spatial study used a Backward-Euler Richardson pair at each mesh to reduce
the leading time error before estimating spatial rates. The temporal study
used fixed fine meshes and a smaller timestep reference on the same mesh. The
results describe these smooth, constant-matrix eigenmodes and this radial
profile norm; they do not establish accuracy or convergence rates for every
mesh, error norm, or composition-dependent problem. The 2D study compares
angularly averaged radial profiles and reports angular spread separately.

In the final runs, mass drift was at most `2.3e-14` in 1D and `5.2e-9` in 2D;
closure residuals were at roundoff level, and all recorded 2D nonlinear solves
converged. The convergence study took about 11 minutes in the tested UW3
environment.

## Optional verification commands

From the repository root, in an environment where Underworld3/PETSc and
UWDiffusion are available:

```bash
python examples/compare_1d_2d_cylindrical_multicomponent.py
python examples/convergence_1d_2d_cylindrical_multicomponent.py
```

The one-shot comparison is quick; the convergence study is more expensive. It
writes detailed results to
`examples/output/convergence_study_summary.json`, which is gitignored and is
not needed to use the model setup above.

## Source references

- [Coupled comparison script](../examples/compare_1d_2d_cylindrical_multicomponent.py)
- [Convergence study script](../examples/convergence_1d_2d_cylindrical_multicomponent.py)
- [1D multicomponent solver](../src/UWDiffusion/fd_1d.py)
- [2D multicomponent solver](../src/UWDiffusion/Models.py)
- [1D verification tests](../tests/test_verification.py)
