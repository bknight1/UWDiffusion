"""Diffusion-matrix construction from component names.

Guards ``lasaga_matrix_from_components`` and the ``set_diffusion_values``
component-name alias.  The point of both is that the ``xi_1 .. xi_n`` naming
contract is otherwise invisible at the call site -- get it wrong and the
composition symbols are silently left unsubstituted.

Run:
    python -m pytest tests/test_diffusion_matrix.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import sympy as sp

    from UWDiffusion.diffusion_matrix import (
        lasaga_matrix,
        lasaga_matrix_from_components,
    )
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"UWDiffusion/SymPy unavailable: {exc}", allow_module_level=True)

UW3_AVAILABLE = True
try:
    import underworld3 as uw

    from UWDiffusion import MulticomponentDiffusionModel
except Exception:  # pragma: no cover - environment dependent
    UW3_AVAILABLE = False

COMPONENTS = ["Fe", "Mg", "Mn", "Ca"]
VALUES = {"Fe": 1.0e-20, "Mg": 5.0e-21, "Mn": 1.0e-19, "Ca": 5.0e-21}


# ==============================================================================
# 1. The helper reproduces the hand-built matrix exactly
# ==============================================================================


def test_from_components_matches_manual_construction():
    """One call must equal the five lines of boilerplate it replaces."""
    manual_d = list(sp.symbols("D_Fe D_Mg D_Mn D_Ca"))
    manual_x = [sp.Symbol(f"xi_{k}") for k in range(1, 5)]
    expected = lasaga_matrix(manual_d, manual_x)

    got = lasaga_matrix_from_components(COMPONENTS)

    assert got.shape == (3, 3)
    assert sp.simplify(sp.Matrix(got) - sp.Matrix(expected)) == sp.zeros(3, 3)


def test_symbol_naming_contract_is_respected():
    """Composition symbols must be xi_1..xi_n, tracer symbols D_<component>.

    This is the contract ``MulticomponentDiffusionModel`` resolves by regex;
    violating it is silent, so assert it here.
    """
    matrix = lasaga_matrix_from_components(COMPONENTS)
    names = {s.name for s in matrix.free_symbols}

    expected_x = {f"xi_{k}" for k in range(1, len(COMPONENTS) + 1)}
    expected_d = {f"D_{c}" for c in COMPONENTS}
    assert expected_x <= names, f"missing composition symbols: {expected_x - names}"
    assert expected_d <= names, f"missing diffusivity symbols: {expected_d - names}"


def test_closes_on_the_last_component():
    """The last component is dependent, so the matrix is (n-1)x(n-1)."""
    for n in (2, 3, 4, 5):
        names = [f"C{k}" for k in range(n)]
        assert lasaga_matrix_from_components(names).shape == (n - 1, n - 1)


# ==============================================================================
# 2. Validation
# ==============================================================================


def test_rejects_too_few_components():
    with pytest.raises(ValueError, match="at least 2"):
        lasaga_matrix_from_components(["Fe"])


def test_rejects_empty_or_non_string_names():
    for bad in (["Fe", ""], ["Fe", "   "], ["Fe", None], ["Fe", 3]):
        with pytest.raises(ValueError, match="non-empty strings"):
            lasaga_matrix_from_components(bad)


def test_single_component_name_is_not_special_cased():
    """A one-element list is still too few components, not a 0x0 matrix."""
    with pytest.raises(ValueError):
        lasaga_matrix_from_components(["Fe"])


# ==============================================================================
# 3. set_diffusion_values accepts bare component names
# ==============================================================================


def _model():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    return MulticomponentDiffusionModel(
        COMPONENTS, mesh, diffusion_matrix=lasaga_matrix_from_components(COMPONENTS)
    )


@pytest.mark.skipif(not UW3_AVAILABLE, reason="Underworld3 not available")
def test_set_diffusion_values_accepts_component_names():
    """{"Fe": ...} must work as well as {"D_Fe": ...}."""
    model = _model()
    model.set_diffusion_values({c: VALUES[c] for c in COMPONENTS})

    resolved = {s.name: v for s, v in model.diffusion_values.items()}
    for c in COMPONENTS:
        assert f"D_{c}" in resolved, f"D_{c} not bound; got {sorted(resolved)}"
        assert resolved[f"D_{c}"] == pytest.approx(VALUES[c])


@pytest.mark.skipif(not UW3_AVAILABLE, reason="Underworld3 not available")
def test_set_diffusion_values_accepts_prefixed_names():
    """The existing {"D_Fe": ...} spelling must keep working."""
    model = _model()
    model.set_diffusion_values({f"D_{c}": VALUES[c] for c in COMPONENTS})

    resolved = {s.name: v for s, v in model.diffusion_values.items()}
    for c in COMPONENTS:
        assert resolved[f"D_{c}"] == pytest.approx(VALUES[c])


@pytest.mark.skipif(not UW3_AVAILABLE, reason="Underworld3 not available")
def test_set_diffusion_values_rejects_unknown_key():
    """A typo must raise, not be silently ignored."""
    model = _model()
    with pytest.raises(KeyError):
        model.set_diffusion_values({"Fe": 1.0, "Fx": 2.0})


@pytest.mark.skipif(not UW3_AVAILABLE, reason="Underworld3 not available")
def test_component_alias_does_not_shadow_a_real_symbol_name():
    """If a matrix genuinely uses a bare name, that must still win over D_<name>."""
    # A hand-rolled matrix whose tracer symbols are literally "Fe"/"Mg"/"Dc",
    # not "D_Fe" etc.  Asking for "Fe" must bind the real symbol rather than
    # fall through to a nonexistent "D_Fe" and raise.
    xi = [sp.Symbol(f"xi_{k}") for k in range(1, 4)]
    bare = [sp.Symbol("Fe"), sp.Symbol("Mg"), sp.Symbol("Dc")]
    matrix = lasaga_matrix(bare, xi)  # -> 2x2, matching closure=True on 3 names

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    model = MulticomponentDiffusionModel(["A", "B", "C"], mesh, diffusion_matrix=matrix)
    model.set_diffusion_values({"Fe": 3.0, "Mg": 4.0})

    resolved = {s.name: v for s, v in model.diffusion_values.items()}
    assert resolved == {"Fe": pytest.approx(3.0), "Mg": pytest.approx(4.0)}
