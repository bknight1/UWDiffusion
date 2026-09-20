"""Focused tests for examples/glenelg_s10_1d_inversion.py.

Exercises the simplified public functions — interval/age parameter mapping,
cached-history re-timing, EPMA mapping, misfit with Ca-closure reporting, and
a small synthetic inversion — without MAGEMin/Julia and without any CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import examples.glenelg_s10_1d_inversion as inv


def make_options(tmp_path: Path, **overrides):
    """Synthetic, quick run options rooted in a temp output directory."""
    defaults = dict(synthetic=True, quick=True, output_dir=tmp_path / "out_inv", make_plots=False)
    defaults.update(overrides)
    return inv.make_options(**defaults)


# ==============================================================================
# Parameter mapping
# ==============================================================================


def test_parameter_mapping_and_strict_ordering() -> None:
    """Nominal intervals map to ages 910/810/710 Ma with 1000 > t1 > t2 > t3."""
    intervals = np.array([90.0, 100.0, 100.0])
    ages = inv.intervals_to_ages(intervals)
    assert np.allclose(ages, [910.0, 810.0, 710.0])
    assert 1000.0 > ages[0] > ages[1] > ages[2]

    assert np.allclose(inv.ages_to_intervals(ages), intervals)

    nodes = inv.ages_to_model_time_nodes(ages)
    assert np.allclose(nodes, [0.0, 60.0, 150.0, 250.0, 350.0])
    assert np.all(np.diff(nodes) > 0)


def test_arbitrary_positive_intervals_guarantee_strict_ordering() -> None:
    """Any positive (dt1, dt2, dt3) guarantees 1000 > t1 > t2 > t3 and round-trips."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        dt = rng.uniform(1.0, 200.0, size=3)
        ages = inv.intervals_to_ages(dt)
        assert 1000.0 > ages[0] > ages[1] > ages[2]
        assert np.allclose(inv.ages_to_intervals(ages), dt)


def test_malformed_interval_or_age_vectors_rejected() -> None:
    with pytest.raises(ValueError):
        inv.intervals_to_ages([90.0, 100.0])
    with pytest.raises(ValueError):
        inv.ages_to_intervals([910.0, 810.0, 710.0, 600.0])


def test_make_options_bridge() -> None:
    """make_options assembles the forward-model namespace from the settings block."""
    options = inv.make_options()
    assert options.garnet_no == 0
    assert options.rmin == 50.0 and options.rmax == 380.0
    assert options.quick is False
    assert options.geometry == "cylindrical"
    assert options.fixed_bc is True
    assert options.synthetic is False
    assert options.source_label == "magemin"
    assert options.fit_components == ["Fe", "Mg", "Mn"]

    # Overrides apply, and the source label follows an overridden synthetic flag
    synthetic_options = inv.make_options(synthetic=True, quick=True, rmax=300.0)
    assert synthetic_options.synthetic is True
    assert synthetic_options.source_label == "synthetic"
    assert synthetic_options.rmax == 300.0


# ==============================================================================
# Cached history and candidate re-timing
# ==============================================================================


def test_cached_history_satisfies_prograde_assumptions(tmp_path: Path) -> None:
    """The synthetic cache is prograde-only: strictly increasing growth steps
    within [0, 60] Myr, with arc coordinates increasing to s = 4."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)

    growth = cached["tr"][cached["tr"] > cached["tr"].min()]
    _, first_idx = np.unique(growth, return_index=True)
    growth_steps = growth[np.sort(first_idx)]
    assert np.all(np.diff(growth_steps) > 0.0)
    assert growth_steps.min() >= 0.0
    assert growth_steps.max() <= inv.growth_window[1] + 1e-9

    inv.check_prograde_history(cached)

    s = np.asarray(cached["retrograde_s"])
    assert np.all(np.diff(s) > 0.0)
    assert s[-1] == 4.0


def test_validate_growth_history_rejects_growth_past_fixed_vertex(tmp_path: Path) -> None:
    """A history still growing past the fixed 1000 Ma vertex (60 Myr) fails clearly."""
    options = make_options(tmp_path)
    history = inv.cache_growth_history(options)

    bad = dict(history)
    tr = history["tr"].copy()
    growth_idx = np.where(tr > tr.min())[0]
    tr[growth_idx] = np.linspace(10.0, 80.0, growth_idx.size)  # last step at 80 Myr > 60
    bad["tr"] = tr
    with pytest.raises(RuntimeError, match="1000 Ma vertex"):
        inv.check_prograde_history(bad)


def test_validate_growth_history_rejects_decreasing_growth(tmp_path: Path) -> None:
    """A history with a genuinely decreasing growth time fails clearly; benign
    strictly-equal shell times are de-duplicated exactly as run_1d_model does."""
    options = make_options(tmp_path)
    history = inv.cache_growth_history(options)

    # Several shells growing at the same path time: accepted (flat duplicates)
    flat = dict(history)
    tr = history["tr"].copy()
    growth_idx = np.where(tr > tr.min())[0]
    tr[growth_idx[3]] = tr[growth_idx[2]]  # duplicate growth time
    flat["tr"] = tr
    inv.check_prograde_history(flat)  # no raise

    # A genuinely decreasing growth time: rejected
    bad = dict(history)
    tr = history["tr"].copy()
    tr[growth_idx[3]] = tr[growth_idx[1]] - 1.0
    bad["tr"] = tr
    with pytest.raises(RuntimeError, match="strictly increasing"):
        inv.check_prograde_history(bad)


def test_validate_growth_history_rejects_empty_growth(tmp_path: Path) -> None:
    options = make_options(tmp_path)
    history = inv.cache_growth_history(options)

    bad = dict(history)
    bad["tr"] = np.array([10.0])  # single row: no growth steps
    with pytest.raises(RuntimeError, match="no growth steps"):
        inv.check_prograde_history(bad)


def test_update_history_timing_updates_path_but_not_zoning(tmp_path: Path) -> None:
    """Re-timing updates the PTt path and retrograde schedule while the cached
    growth zoning (radii, times, compositions) is preserved untouched."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)

    dt_new = np.array([60.0, 80.0, 120.0])  # -> ages [940, 860, 740], nodes end 320 Myr
    updated = inv.update_history_timing(cached, dt_new)

    assert np.array_equal(updated["Rr"], cached["Rr"])
    assert np.array_equal(updated["tr"], cached["tr"])
    assert np.array_equal(updated["Fer"], cached["Fer"])

    node4 = float(inv.ages_to_model_time_nodes(inv.intervals_to_ages(dt_new))[-1])
    assert node4 == 320.0
    assert float(updated["PTt_path"][:, 2].max()) == node4
    assert float(updated["retrograde"]["t"][-1]) == node4  # exact endpoint
    assert np.isclose(float(updated["T_of_t"](node4)), 510.0 + 273.15, rtol=1e-6)
    assert np.all(np.diff(updated["retrograde"]["t"]) > 0.0)


def test_candidate_schedule_ends_exactly_at_its_own_t3_node(tmp_path: Path) -> None:
    """Every candidate's retrograde schedule ends EXACTLY at its own t3 node
    (no +eps overshoot); the nominal 710 Ma imposes no constraint."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)

    candidates = [
        [90.0, 100.0, 100.0],  # nominal: t3 = 710 Ma
        [90.0, 100.0, 60.0],   # t3 = 750 Ma
        [120.0, 80.0, 200.0],  # t3 = 600 Ma
        [200.0, 40.0, 10.0],   # t3 = 850 Ma
    ]
    endpoints = set()
    for dt in candidates:
        node4 = float(inv.ages_to_model_time_nodes(inv.intervals_to_ages(dt))[-1])
        updated = inv.update_history_timing(cached, dt)

        assert float(updated["retrograde"]["t"][-1]) == node4
        assert np.isclose(float(updated["T_of_t"](node4)), 510.0 + 273.15, rtol=1e-6)
        assert np.all(np.diff(updated["retrograde"]["t"]) > 0.0)
        t_growth_final = float(cached["tr"].max())
        expected_trim = int(np.sum(updated["retrograde"]["t"] < t_growth_final))
        assert updated["retrograde_trimmed"] == expected_trim

        assert np.array_equal(updated["Rr"], cached["Rr"])
        endpoints.add(node4)

    # Candidates with different total durations end at different model times
    assert endpoints == {60.0 + float(np.sum(dt)) for dt in candidates}


def test_update_history_timing_rejects_nonmonotonic_mapping(tmp_path: Path) -> None:
    """A nonmonotonic arc-coordinate mapping is rejected, not silently repaired."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)
    bad = dict(cached)
    bad["retrograde_s"] = np.array([0.8, 4.0, 3.0, 2.0, 2.5, 3.5, 3.9])
    with pytest.raises(ValueError, match="not strictly increasing"):
        inv.update_history_timing(bad, [90.0, 100.0, 100.0])


def test_update_history_timing_reports_pre_growth_points(tmp_path: Path) -> None:
    """Retrograde points mapped before the final growth step are reported via
    'retrograde_trimmed' while the schedule stays monotone and ends exactly
    at the candidate node."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)
    tampered = dict(cached)
    n_retro = len(cached["retrograde"]["t"])
    tampered["retrograde_s"] = np.linspace(-0.5, 4.0, n_retro)  # starts before t = 0

    updated = inv.update_history_timing(tampered, [90.0, 100.0, 100.0])
    assert updated["retrograde_trimmed"] >= 1
    assert np.all(np.diff(updated["retrograde"]["t"]) > 0.0)
    node4 = float(inv.ages_to_model_time_nodes([910.0, 810.0, 710.0])[-1])
    assert float(updated["retrograde"]["t"][-1]) == node4


def test_check_candidate_intervals_rejects_inadmissible(tmp_path: Path) -> None:
    """Nonpositive/non-finite intervals and candidates ending too early are rejected."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)

    for bad in ([0.0, 100.0, 100.0], [-10.0, 100.0, 100.0], [90.0, 100.0], [np.nan, 100.0, 100.0]):
        with pytest.raises(ValueError):
            inv.check_candidate_intervals(bad, cached)

    # Ends at 60 + 30 = 90 Myr > 50 Myr final growth step: admissible, t3 = 970 Ma
    ages = inv.check_candidate_intervals([10.0, 10.0, 10.0], cached)
    assert np.allclose(ages, [990.0, 980.0, 970.0])

    # A history whose final growth step is later than the candidate end is rejected
    late = dict(cached)
    late["tr"] = cached["tr"].copy()
    late["tr"][-1] = 95.0
    with pytest.raises(ValueError, match="final growth"):
        inv.check_candidate_intervals([10.0, 10.0, 10.0], late)


# ==============================================================================
# EPMA mapping, loading and out-of-domain handling
# ==============================================================================


def test_map_epma_folded_radius_matches_legacy_convention() -> None:
    """The folded mapping centres at the transect midpoint (legacy 2D
    convention) then folds with abs(): r(0) == r(max) == r_half, midpoint -> 0."""
    r_mm = np.linspace(0.0, 0.7558, 25)  # S10-gt1-like diameter traverse
    folded = inv.map_epma_folded_radius(r_mm)

    r_half_mm = r_mm.max() / 2.0
    assert np.isclose(folded.max(), r_half_mm * 1e3)  # both ends fold onto the rim
    assert np.isclose(folded.min(), 0.0)  # the centre maps to the core
    assert np.all(folded >= 0.0)
    assert np.isclose(folded[0], folded[-1])  # symmetric points fold together


def test_load_epma_data_structure() -> None:
    """The real S10-gt1 EPMA transect loads with sorted folded radii inside rmax."""
    if not inv.epma_csv.exists():
        pytest.skip(f"EPMA file not found at {inv.epma_csv}")

    obs = inv.load_epma_data(inv.epma_csv, rmax_um=380.0)
    assert obs["r"].size == 140
    assert np.all(obs["r"] >= 0.0)
    assert np.all(np.diff(obs["r"]) >= 0.0)
    assert obs["r"].max() <= 380.0
    for comp in ("Fe", "Mg", "Mn", "Ca"):
        assert np.all(obs[comp] >= 0.0)
        assert np.all(obs[comp] <= 1.0)


def test_load_epma_data_rejects_out_of_domain() -> None:
    """Points folding beyond rmax are rejected rather than silently clipped."""
    if not inv.epma_csv.exists():
        pytest.skip(f"EPMA file not found at {inv.epma_csv}")

    with pytest.raises(ValueError, match="outside the model domain"):
        inv.load_epma_data(inv.epma_csv, rmax_um=100.0)


def test_evaluate_model_at_radii_rejects_out_of_domain() -> None:
    """Model interpolation refuses to silently clip observations outside the grid."""
    r_grid = np.linspace(0.0, 380.0, 21)
    prof = np.tile(np.array([0.70, 0.15, 0.05, 0.10]), (21, 1))

    ok = inv.evaluate_model_at_radii(r_grid, prof, np.array([0.0, 190.0, 380.0]))
    assert np.allclose(ok["Fe"], 0.70)

    with pytest.raises(ValueError, match="outside the model grid"):
        inv.evaluate_model_at_radii(r_grid, prof, np.array([380.5]))


def test_synthetic_observations(tmp_path: Path) -> None:
    """Synthetic observations sample 40 core-to-rim points inside [0, 1]."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)
    obs = inv.make_synthetic_observations(cached, options)

    assert obs["source_path"] == "synthetic"
    assert obs["r"].size == 40
    assert np.all(obs["r"] >= 0.0)
    for comp in inv.all_components:
        assert obs[comp].size == 40
        assert np.all(obs[comp] >= 0.0)
        assert np.all(obs[comp] <= 1.0)


# ==============================================================================
# Misfit: Fe/Mg/Mn default, Ca closure reported
# ==============================================================================


def test_compute_misfit_zero_on_identical_profile() -> None:
    r_grid = np.linspace(0.0, 380.0, 15)
    prof = np.tile(np.array([0.70, 0.15, 0.05, 0.10]), (15, 1))
    obs = {
        "r": np.linspace(0.0, 380.0, 20),
        "Fe": np.full(20, 0.70),
        "Mg": np.full(20, 0.15),
        "Mn": np.full(20, 0.05),
        "Ca": np.full(20, 0.10),
    }

    total_rmse, comp_rmse, residuals = inv.compute_misfit(r_grid, prof, obs)
    assert np.isclose(total_rmse, 0.0, atol=1e-12)
    for comp in inv.all_components:
        assert np.isclose(comp_rmse[comp], 0.0, atol=1e-12)
        assert np.allclose(residuals[comp], 0.0, atol=1e-12)


def test_compute_misfit_positive_on_difference() -> None:
    r_grid = np.linspace(0.0, 380.0, 10)
    prof = np.tile(np.array([0.72, 0.0, 0.0, 0.0]), (10, 1))
    obs = {
        "r": np.linspace(0.0, 380.0, 10),
        "Fe": np.full(10, 0.70),
        "Mg": np.zeros(10),
        "Mn": np.zeros(10),
        "Ca": np.zeros(10),
    }

    total_rmse, comp_rmse, _ = inv.compute_misfit(r_grid, prof, obs)
    assert np.isclose(comp_rmse["Fe"], 0.02, atol=1e-10)
    assert total_rmse > 0.0


def test_default_fit_components_exclude_ca_but_ca_is_reported() -> None:
    """A residual that lives ONLY in the Ca closure must leave the default total
    RMSE untouched, while Ca is still reported; opting into Ca moves the total."""
    r = np.linspace(0.0, 100.0, 10)
    prof = np.tile(np.array([0.70, 0.15, 0.05, 0.10]), (10, 1))
    obs = {
        "r": r,
        "Fe": np.full(10, 0.70),
        "Mg": np.full(10, 0.15),
        "Mn": np.full(10, 0.05),
        "Ca": np.full(10, 0.20),  # model says 0.10 -> residual only in Ca
    }

    total_default, comp_default, residuals_default = inv.compute_misfit(r, prof, obs)
    assert np.isclose(total_default, 0.0, atol=1e-12)  # Ca excluded by default
    assert comp_default["Ca"] > 0.05  # ... but still reported
    assert np.all(np.abs(residuals_default["Ca"]) > 0.05)

    total_with_ca, _, _ = inv.compute_misfit(r, prof, obs, components=["Fe", "Mg", "Mn", "Ca"])
    assert total_with_ca > 0.05


def test_compute_misfit_validates_fit_components() -> None:
    r = np.linspace(0.0, 100.0, 5)
    prof = np.zeros((5, 4))
    obs = {"r": r, "Fe": np.zeros(5), "Mg": np.zeros(5), "Mn": np.zeros(5), "Ca": np.zeros(5)}
    with pytest.raises(ValueError, match="unique subset"):
        inv.compute_misfit(r, prof, obs, components=["Na"])
    with pytest.raises(ValueError, match="unique subset"):
        inv.compute_misfit(r, prof, obs, components=["Fe", "Fe"])


# ==============================================================================
# Objective: fresh forward evaluations, free t3
# ==============================================================================


def test_objective_responds_to_t3_alone(tmp_path: Path) -> None:
    """Differing only in dt3 (t3) must give distinct finite objectives: t3 is free."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)
    obs = inv.make_synthetic_observations(cached, options)

    losses = [inv.objective([90.0, 100.0, dt3], cached, obs, options) for dt3 in (40.0, 100.0, 220.0)]
    assert all(np.isfinite(losses))
    assert len(set(losses)) == 3


def test_objective_penalises_infeasible_candidates_without_a_forward_run(tmp_path: Path) -> None:
    """Nonpositive/non-finite intervals get a barrier penalty, not a solver call."""
    options = make_options(tmp_path)
    cached = inv.cache_growth_history(options)
    obs = inv.make_synthetic_observations(cached, options)

    state = {"count": 0, "best_rmse": np.inf}
    penalty = inv.objective([-5.0, 100.0, 100.0], cached, obs, options, state)
    assert penalty >= 1e5
    assert state["count"] == 0  # no forward run was spent on the infeasible candidate


# ==============================================================================
# Small synthetic inversion (no CLI)
# ==============================================================================


def test_synthetic_inversion_smoke(tmp_path: Path) -> None:
    """A 2-evaluation bounded synthetic inversion writes all output artifacts."""
    options = make_options(tmp_path, max_nfev=2)
    out_dir, summary = inv.run_inversion(options)

    assert out_dir.is_dir()
    assert summary["nfev"] == 2
    assert summary["success"] is False  # stopped at max_nfev
    assert summary["fit_components"] == ["Fe", "Mg", "Mn"]

    ages = summary["best_fit_ages_ma"]
    assert len(ages) == 3
    assert 1000.0 > ages[0] > ages[1] > ages[2]

    for name in ("inversion_summary.json", "best_fit_profiles.csv", "epma_comparison.csv", "best_fit_results.npz"):
        assert (out_dir / name).is_file(), f"missing output: {out_dir / name}"

    data = json.loads((out_dir / "inversion_summary.json").read_text())
    assert data["best_fit_rmse"] >= 0.0
    assert set(data["best_fit_component_rmse"]) == {"Fe", "Mg", "Mn", "Ca"}

    npz = np.load(out_dir / "best_fit_results.npz")
    for key in ("r_model_um", "profile_final", "best_fit_ages_ma", "ptt_path", "epma_Ca"):
        assert key in npz


def test_synthetic_inversion_saves_fit_figures(tmp_path: Path) -> None:
    """With make_plots=True the comparison figure is written as PDF and PNG."""
    pytest.importorskip("matplotlib")
    options = make_options(tmp_path, max_nfev=2, make_plots=True)
    out_dir, _ = inv.run_inversion(options)

    for name in ("inversion_fit.pdf", "inversion_fit.png"):
        fig_path = out_dir / name
        assert fig_path.is_file(), f"missing figure: {fig_path}"
        assert fig_path.stat().st_size > 0, f"empty figure: {fig_path}"


def test_run_never_overwrites_previous_output(tmp_path: Path) -> None:
    """Repeated runs create suffixed output directories instead of clobbering."""
    options = make_options(tmp_path, max_nfev=2)
    out_dir_1, _ = inv.run_inversion(options)
    out_dir_2, _ = inv.run_inversion(options)
    assert out_dir_1 != out_dir_2
    assert out_dir_2.is_dir()


# ==============================================================================
# Refactor guards: no CLI, no import fallbacks
# ==============================================================================


def test_script_has_no_cli_or_fallback_imports() -> None:
    """The research script parses no arguments and uses direct imports only."""
    assert not hasattr(inv, "parse_args")
    assert not hasattr(inv, "main")
    source = Path(inv.__file__).read_text()
    assert "argparse" not in source
    assert "sys.argv" not in source
    assert "except ImportError" not in source
