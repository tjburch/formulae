"""Registered formula semantics are separate from ordinary function arithmetic."""

import numpy as np
import pandas as pd
import pytest

from formulae import design_matrices, model_description
from formulae.terms.structured import StructuredTerm

WRAPPERS = {
    "ar1": "coordinate",
    "ou": "coordinate",
    "us": "coefficient",
    "diag": "coefficient",
    "cs": "coefficient",
}


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [2.0, 1.0, 4.0, 3.0],
            "x": [0.0, 2.0, 1.0, 3.0],
            "visit": pd.Categorical(["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True),
            "g": ["a", "b", "a", "b"],
            "h": ["a", "a", "b", "b"],
            "odd name": ["a", "b", "a", "b"],
        }
    )


def structured(formula, data):
    dm = design_matrices("y ~ " + formula, data, formula_wrappers=WRAPPERS)
    return [t for t in dm.common.terms.values() if isinstance(t, StructuredTerm)]


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("group", ["g", '"g"', "[g, h]", '["g", "h"]'])
def test_grouping_forms(data, wrapper, group):
    coordinate = WRAPPERS[wrapper] == "coordinate"
    inner = "x" if coordinate else "1 + x"
    factor = "g:h" if "[" in group else "g"
    left = structured(f"{wrapper}({inner}, by={factor})", data)[0]
    right = structured(f"{wrapper}({inner}, by={group})", data)[0]
    assert left.name == right.name
    assert left.var_names == right.var_names
    pd.testing.assert_frame_equal(left.group_values, right.group_values)
    if coordinate:
        np.testing.assert_array_equal(left.coordinate, right.coordinate)
    else:
        np.testing.assert_array_equal(left.design.design_matrix, right.design.design_matrix)


def test_inner_design_is_formula(data):
    for inner in ["x", "0 + C(visit)", "x * C(visit)", "1 + center(x)"]:
        term = structured(f"us({inner}, by=g)", data)[0]
        reference = design_matrices(inner, data)
        np.testing.assert_array_equal(term.design.design_matrix, reference.common.design_matrix)
        prediction = data.iloc[[2, 0]]
        np.testing.assert_array_equal(
            term.evaluate_new_data(prediction)[0],
            reference.common.evaluate_new_data(prediction).design_matrix,
        )


def test_declared_levels_and_list_options(data):
    term = structured('us(0 + C(visit, levels=["a", "b", "c"]), by=g)', data)[0]
    assert term.design.design_matrix.shape == (4, 3)
    assert np.all(term.design.design_matrix[:, 2] == 0)


def test_nesting_duplicates_and_discovery(data):
    terms = structured("ar1(x, by=g/h/`odd name`)", data)
    assert [t.groups for t in terms] == [("g",), ("g", "h"), ("g", "h", "odd name")]
    terms = structured('ar1(x, by=g) + ar1(x, by="g")', data)
    assert len(terms) == 1
    description = model_description('y ~ ar1(x, by=["g", "odd name"])', formula_wrappers=WRAPPERS)
    assert description.var_names == {"y", "x", "g", "odd name"}


def test_group_columns_missing_values(data):
    data.loc[0, "odd name"] = None
    dm = design_matrices('y ~ ar1(x, by="odd name")', data, formula_wrappers=WRAPPERS)
    assert len(dm.data) == 3
    with pytest.raises(ValueError, match="incomplete"):
        design_matrices(
            'y ~ ar1(x, by="odd name")', data, na_action="error", formula_wrappers=WRAPPERS
        )
    with pytest.raises(ValueError, match="Missing grouping columns"):
        structured('ar1(x, by="absent")', data)


@pytest.mark.parametrize(
    "formula, message",
    [
        ("ar1(1 + x, by=g)", "coordinate"),
        ("us(x, by=[])", "empty"),
        ("us(x, by=g, by=h)", "Duplicate"),
        ("us(x, x, by=g)", "exactly one"),
        ("ar1(x, g)", "exactly one"),
        ("us(x, option=x)", "literal"),
    ],
)
def test_useful_errors(data, formula, message):
    with pytest.raises(ValueError, match=message):
        structured(formula, data)


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("options", ["", ", by=h"])
def test_inner_group_terms_are_rejected(data, wrapper, options):
    with pytest.raises(ValueError, match="Use 'by='"):
        structured(f"{wrapper}(0 + x | g{options})", data)


def test_ordinary_arithmetic_namespace_and_groups(data):
    ordinary = design_matrices(
        "y ~ identity(1 + x) + (x | g/h)", data, extra_namespace={"identity": lambda x: x}
    )
    np.testing.assert_array_equal(ordinary.common["identity(1 + x)"].ravel(), 1 + data.x)
    assert set(ordinary.group.terms) == {"1|g", "x|g", "1|g:h", "x|g:h"}
    dm = design_matrices("y ~ us(1 + x)", data, extra_namespace={"us": lambda x: x})
    np.testing.assert_array_equal(dm.common["us(1 + x)"].ravel(), 1 + data.x)


def test_group_name_cannot_collide_with_composite(data):
    column = "['g', 'h']"
    data[column] = data.g
    terms = structured(f'ar1(x, by=[g,h]) + ar1(x, by=["{column}"])', data)
    assert len(terms) == 2
    assert terms[0].name != terms[1].name


@pytest.mark.parametrize("column", ["True", "False", "None"])
def test_group_name_matching_python_literal_round_trips(data, column):
    data[column] = data.g
    term = structured(f'ar1(x, by="{column}")', data)[0]
    restored = structured(term.name, data)[0]
    assert restored.name == term.name
    assert restored.groups == (column,)
    pd.testing.assert_frame_equal(restored.group_values, term.group_values)


@pytest.mark.parametrize("expression", ["x + z", "x:z", "center(x)"])
def test_quoted_predictor_cannot_collide_with_formula(data, expression):
    data["z"] = data.x**2
    data[expression] = data.x**3
    terms = structured(f"us(`{expression}`, by=g) + us({expression}, by=g)", data)
    assert len(terms) == 2
    assert terms[0].name != terms[1].name
    for term, inner in zip(terms, [f"`{expression}`", expression]):
        reference = design_matrices(inner, data)
        np.testing.assert_array_equal(term.design.design_matrix, reference.common.design_matrix)
        assert structured(term.name, data)[0].name == term.name


@pytest.mark.parametrize(
    "left,right", [("center(`x + z`)", "center(x + z)"), ("I(x * (z + x))", "I(x * z + x)")]
)
def test_transform_identity_preserves_quoted_names_and_grouping(data, left, right):
    data["z"] = data.x**2
    data["x + z"] = data.x**3
    terms = structured(f"us({left}, by=g) + us({right}, by=g)", data)
    assert len(terms) == 2
    assert terms[0].name != terms[1].name
    for term, inner in zip(terms, [left, right]):
        reference = design_matrices(inner, data)
        np.testing.assert_array_equal(term.design.design_matrix, reference.common.design_matrix)
        assert structured(term.name, data)[0].name == term.name


@pytest.mark.parametrize("operator", ["*", ":", "/", "**"])
@pytest.mark.parametrize("reverse", [False, True])
def test_outer_interactions_are_rejected(data, operator, reverse):
    left, right = ("us(x, by=g)", "x")
    if reverse:
        left, right = right, left
    with pytest.raises(ValueError, match="additive"):
        structured(f"{left} {operator} {right}", data)


def test_grouping_subset_is_rejected(data):
    with pytest.raises(ValueError, match="column names"):
        structured('ar1(x, by=g["a"])', data)


def test_categorical_declaration_validation(data):
    for levels in [["a", "a", "b"], ["a", "c"]]:
        with pytest.raises(ValueError, match="Declared categorical levels"):
            structured(f"us(0 + C(visit, levels={levels!r}), by=g)", data)
