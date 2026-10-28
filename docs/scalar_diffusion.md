# Scalar diffusion setup and validation

This guide covers the scalar diffusion APIs and the tested comparison between
the 1D cylindrical and 2D annulus models. The examples use dimensionless values;
configure Underworld3 scaling before assigning physical diffusivities and times.

## Equation and geometry

The 1D solver supports planar, cylindrical, and spherical radial geometry. The
2D scalar solver operates on an Underworld3 Cartesian mesh. For a radial field
on an annulus,

\[
\nabla\cdot(D\nabla C) = \frac{1}{r}\frac{\partial}{\partial r}
\left(rD\frac{\partial C}{\partial r}\right),
\]

which is the same continuum operator as the 1D cylindrical solver. The spatial
discretizations differ: the 1D scalar tracer uses conservative vertex-centred
finite volumes; `DiffusionModel` uses Q2 finite elements.

The snippets import the required APIs as follows:

```python
import numpy as np
import underworld3 as uw

from UWDiffusion import DiffusionModel, MulticomponentDiffusion1D
```

If running from a source checkout rather than an installed UWDiffusion package,
install it in the active Underworld3 environment (for example, `pip install -e .`
from the repository root).

## 1D cylindrical finite-volume model

`components=1` selects a scalar tracer, without dependent-component closure.
The left and right boundaries correspond to the inner and outer radii.

```python
r_inner, r_outer = 0.5, 1.0
r = np.linspace(r_inner, r_outer, 51)
profile = 0.5 + 0.1 * np.cos(np.pi * (r - r_inner) / (r_outer - r_inner))

model_1d = MulticomponentDiffusion1D(
    grid=r,
    diffusivity=1.0,
    geometry="cylindrical",
    components=1,
    bcs={"left": ("neumann", 0.0), "right": ("neumann", 0.0)},
    initial_composition=profile,
)
result_1d = model_1d.run_simulation(dt=0.001, total_time=0.02)
```

`result_1d.get_profile(0, -1)` returns the final profile of the scalar field.
The 1D model defaults to zero-flux boundaries if `bcs` is omitted; they are
shown explicitly here to make the setup clear.

## 2D scalar model on an annulus

`DiffusionModel` handles one scalar field. Initialize its `mesh_var`, assign the
diffusivity, and run the model. Leaving the annulus rims unconstrained gives
natural zero-normal-flux boundaries.

```python
r_inner, r_outer = 0.5, 1.0
dt, final_time = 0.001, 0.02

mesh = uw.meshing.Annulus(
    radiusOuter=r_outer,
    radiusInner=r_inner,
    cellSize=0.04,
    qdegree=2,
)
model_2d = DiffusionModel("C", mesh, degree=2, order=1)
model_2d.diffusivity = 1.0

coords = model_2d.mesh_var.coords
r_nodes = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
initial = 0.5 + 0.1 * np.cos(np.pi * (r_nodes - r_inner) / (r_outer - r_inner))
with mesh.access(model_2d.mesh_var):
    model_2d.mesh_var.data[:, 0] = initial

model_2d.run_simulation(duration=final_time, min_dt=dt, max_dt=dt)
```

Setting `min_dt=max_dt=dt` requests a fixed timestep, apart from a possible
roundoff adjustment on the final step to land exactly on `duration`.

## Validation findings

The matched benchmark used the annulus `0.5 <= r <= 1.0`, constant diffusivity
`D=1`, homogeneous-Neumann boundaries, Backward Euler (`dt=0.001`, final time
`0.02`), and a radial Neumann Bessel eigenmode. It compares both code outputs
with each other and with the analytical transient solution.

| Metric | Result |
|---|---:|
| 1D vs 2D max radial-profile difference | `2.879e-4` |
| 1D vs analytical max error / RMSE | `1.952e-3` / `1.118e-3` |
| 2D vs analytical max error / RMSE | `2.029e-3` / `1.170e-3` |
| 2D circumferential RMS spread (max / mean) | `3.050e-5` / `3.905e-6` |

This is a close match at one resolution, not a scalar spatial- or
time-convergence study. The two spatial schemes have different errors, and the
2D annulus mesh introduces small circumferential variations. See the
[comparison script](../examples/compare_1d_2d_cylindrical.py) for the exact
benchmark setup and output calculations.

## Optional verification

From the repository root, in an environment where Underworld3/PETSc and
UWDiffusion are available:

```bash
python examples/compare_1d_2d_cylindrical.py
```

See also the [multicomponent diffusion guide](multicomponent_diffusion.md),
[scalar 2D verification tests](../tests/test_2d_verification.py), and
[1D verification tests](../tests/test_verification.py).
