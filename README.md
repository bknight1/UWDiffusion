# UWDiffusion

UWDiffusion is a Python library designed for modelling and solving a range of diffusion problems. The idea of the library is to streamline the model and solver setup to make diffusion-based problems easier to run. 

UWDiffusion is built on top of [Underworld3](https://github.com/underworldcode/underworld3), a geodynamics modeling framework. underworld3 provides the solvers and data handling, by leveraging PETSc, that UWDiffusion leverages for solving various diffusion-based problems. The user can easily break the high level objects and get back to core underworld3 functionality at any step of model design.

## Features
- Simplified workflow and predefined models for diffusion problems.
- Tutorials for practical examples.

## Installation
underworld3 is required to be installed before installing UWDiffusion. The install instructions for underworld3 can be found [here](https://underworldcode.github.io/underworld3/development/_quickstart/Installation.html).

After installing underworld3, you can install UWDiffusion by cloning the repository and use the following command:

```bash
pip install .
```

## Usage
The library can be used to model and solve diffusion problems. Tutorials are provided in the `Tutorials/` directory to help you get started.


## Licence
All UWDiffusion source code is released under the LGPL-3 open source licence. This covers all files in UWDiffusion constituting the UWDiffusion Python module. Notebooks, stand-alone documentation and Python scripts which show how the code is used and run are licensed under the Creative Commons Attribution 4.0 International Licence. For more details, see the [LICENCE.md](LICENCE.md) file.

## Testing on binder
Feel free to run the examples on binder!
[![Launch in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bknight1/uw3-binder-launcher/main?urlpath=git-pull%3Frepo%3Dhttps%25253A%25252F%25252Fgithub.com%25252Fbknight1%25252FUWDiffusion%26branch%3Dmain%26urlpath%3Dlab%25252Ftree%25252FUWDiffusion)

## References

### underworld3
- Moresi, L., Mansour, J., Giordani, J., Knepley, M., Knight, B., Graciosa, J.C., Gollapalli, T., Lu, N., Beucher, R., 2025. Underworld3: Mathematically Self-Describing Modelling in Python for Desktop, HPC and Cloud. JOSS 10, 7831. https://doi.org/10.21105/joss.07831

### UW Diffusion
- Knight, B.S., Clark, C., 2025. Modelling diffusion, decay and ingrowth of U–Pb isotopes in zircon. EGUsphere 1–24. https://doi.org/10.5194/egusphere-2025-2278
- Clark, C., Brown, M., Knight, B., Johnson, T.E., Mitchell, R.J., Gupta, S., 2024. Ultraslow cooling of an ultrahot orogen. Geology. https://doi.org/10.1130/G52442.1


