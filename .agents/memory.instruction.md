---
applyTo: '**'
---

# Coding Preferences
- Prefer beginner-friendly docstrings that explain purpose, when to use a class,
  and a short recommended workflow.

# Project Architecture
- Core models live in src/UWDiffusion/Models.py with three primary classes:
  - DiffusionModel
  - DiffusionDecayIngrowthModel
  - MulticomponentDiffusionModel
- Utilities and meshing helpers live in src/UWDiffusion/utilities.py and src/UWDiffusion/meshing.py.
