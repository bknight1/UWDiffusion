"""Structural and synthetic checks for the sequential Glenelg Jupytext source."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from examples import glenelg_s10_1d_support as support


SOURCE = Path(__file__).parents[1] / "examples" / "glenelg_s10_1d.py"


def test_forward_is_percent_format_and_has_legacy_cell_order() -> None:
    source = SOURCE.read_text()
    assert source.count("# %%") >= 12
    assert "run_model" not in source
    assert "argparse" not in source
    assert "sys.argv" not in source
    assert "except ImportError" not in source
    positions = [source.index(marker) for marker in (
        "# #### Growth evolution CSV", "# #### Radial grid",
        "# #### Growth hook", "# #### Prograde growth",
        "# #### Final growth shell", "# #### Retrograde boundary",
        "# #### Profiles and output",
    )]
    assert positions == sorted(positions)


def test_forward_has_only_state_specific_helpers() -> None:
    tree = ast.parse(SOURCE.read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"build_ptt_path", "generate_growth_history_magemin", "make_diffusivity_callable", "apply_growth_step", "update_retrograde_boundary", "make_plots_output"} <= functions
    assert "run_model" not in functions
    assert "_run_radial_diffusion" not in functions


def test_synthetic_support_evaluation_is_fresh_and_closed() -> None:
    history = support.generate_growth_history_synthetic(rmin=30.0, rmax=60.0)
    model, diagnostics = support.run_forward_evaluation(
        history, fixed_bc=True, quick=True, dt_myr=2.0, geometry="cylindrical"
    )
    assert model.time / support.sec_per_myr == 90.0
    assert diagnostics["profile_final"].shape[1] == 4
    assert np.max(np.abs(diagnostics["profile_final"].sum(axis=1) - 1.0)) < 1e-8
    second, _ = support.run_forward_evaluation(
        history, fixed_bc=True, quick=True, dt_myr=2.0, geometry="cylindrical"
    )
    assert second is not model


def test_support_arrhenius_has_expected_ca_scaling() -> None:
    values = support.tracer_diffusivities_m2s(998.15, 12200.0)
    assert values.shape == (4,)
    assert np.all(values > 0.0)
    assert np.isclose(values[3], 0.05 * values[2])
