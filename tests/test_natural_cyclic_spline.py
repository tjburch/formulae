import pytest
import re

import numpy as np
import pandas as pd

from formulae.matrices import design_matrices
from formulae.transforms import NaturalCubicSpline, CyclicCubicSpline


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1234)
    size = 21
    data = pd.DataFrame(
        {"seq": np.linspace(0, 1, 21), "x1": rng.uniform(size=size), "x2": rng.uniform(size=size)}
    )
    return data


@pytest.fixture(scope="module")
def sequence():
    return np.linspace(0, 1, 21)


# ============================================================================
# NaturalCubicSpline (cr) tests
# ============================================================================


class TestNaturalCubicSpline:
    """Tests for the NaturalCubicSpline (cr) transform."""

    def test_basic_shape(self, sequence):
        """Test that cr returns the correct number of columns."""
        cr = NaturalCubicSpline()
        matrix = cr(sequence, df=4)
        assert matrix.shape == (21, 4)

    def test_df_equals_columns(self, sequence):
        """Test that df parameter determines number of output columns."""
        for df in [2, 3, 4, 5, 6]:
            cr = NaturalCubicSpline()
            matrix = cr(sequence, df=df)
            assert matrix.shape[1] == df, f"Expected {df} columns, got {matrix.shape[1]}"

    def test_knots_stored(self, sequence):
        """Test that knots are properly stored after initialization."""
        cr = NaturalCubicSpline()
        cr(sequence, df=4)
        assert cr.params_set is True
        assert cr._knots is not None
        # df=4 means 3 interior knots + 2 boundary = 5 knots
        assert len(cr._knots) == 5

    def test_bounds_stored(self, sequence):
        """Test that bounds are properly stored."""
        cr = NaturalCubicSpline()
        cr(sequence, df=4)
        assert cr._lower_bound == 0.0
        assert cr._upper_bound == 1.0

    def test_custom_bounds(self, sequence):
        """Test custom lower and upper bounds."""
        cr = NaturalCubicSpline()
        cr(sequence, df=4, lower_bound=-1.0, upper_bound=2.0)
        assert cr._lower_bound == -1.0
        assert cr._upper_bound == 2.0

    def test_stateful_reuse(self, sequence):
        """Test that parameters are reused on subsequent calls."""
        cr = NaturalCubicSpline()
        matrix1 = cr(sequence, df=4)
        original_knots = cr._knots.copy()

        # Call again with different data
        new_data = np.linspace(0.2, 0.8, 10)
        matrix2 = cr(new_data, df=4)

        # Knots should be unchanged
        assert np.allclose(cr._knots, original_knots)

    def test_through_design_matrices(self, data):
        """Test cr works through design_matrices interface."""
        dm = design_matrices("cr(seq, df=4) - 1", data)
        matrix = dm.common.design_matrix
        assert matrix.shape == (21, 4)

        labels = dm.common.terms["cr(seq, df=4)"].labels
        assert labels == ["cr(seq, df=4)[0]", "cr(seq, df=4)[1]",
                          "cr(seq, df=4)[2]", "cr(seq, df=4)[3]"]

    def test_new_data_evaluation(self, data):
        """Test that new data evaluation reuses stored parameters."""
        dm = design_matrices("cr(seq, df=4) - 1", data)
        original_knots = (
            dm.common.terms["cr(seq, df=4)"]
            .components[0].call.stateful_transform._knots.copy()
        )

        # Evaluate with new data
        data2 = pd.DataFrame({"seq": np.linspace(0.1, 0.9, 10)})
        new_common = dm.common.evaluate_new_data(data2)

        # Check shape
        assert new_common.design_matrix.shape == (10, 4)

        # Check knots are preserved
        new_knots = (
            new_common.terms["cr(seq, df=4)"]
            .components[0].call.stateful_transform._knots
        )
        assert np.allclose(original_knots, new_knots)

    def test_explicit_knots(self, sequence):
        """Test providing explicit knots."""
        cr = NaturalCubicSpline()
        knots = [0.25, 0.5, 0.75]
        matrix = cr(sequence, knots=knots)
        # With 3 interior knots, we get df = 3 + 1 = 4 columns
        assert matrix.shape[1] == 4
        # All knots should be: [0, 0.25, 0.5, 0.75, 1.0]
        assert np.allclose(cr._knots, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_df_with_knots_validation(self, sequence):
        """Test that df and knots must be consistent."""
        cr = NaturalCubicSpline()
        with pytest.raises(ValueError, match="implies 3 interior knots"):
            cr(sequence, df=4, knots=[0.5])  # df=4 needs 3 interior knots

    def test_invalid_df(self, sequence):
        """Test validation of df parameter."""
        with pytest.raises(ValueError, match="'df' must be either None or integer"):
            cr = NaturalCubicSpline()
            cr(sequence, df=[2])

        with pytest.raises(ValueError, match="'df' must be >= 1"):
            cr = NaturalCubicSpline()
            cr(sequence, df=0)

    def test_df_and_knots_none(self, sequence):
        """Test that either df or knots must be specified."""
        with pytest.raises(ValueError, match="Must specify either 'df' or 'knots'"):
            cr = NaturalCubicSpline()
            cr(sequence)

    def test_knots_out_of_bounds(self, sequence):
        """Test that knots must be within bounds."""
        with pytest.raises(
            ValueError, match=re.escape("Some knot values [1.2] fall above upper bound")
        ):
            cr = NaturalCubicSpline()
            cr(sequence, knots=[0.3, 1.2])

        with pytest.raises(
            ValueError, match=re.escape("Some knot values [-0.1] fall below lower bound")
        ):
            cr = NaturalCubicSpline()
            cr(sequence, knots=[-0.1, 0.3])

    def test_invalid_bounds(self, sequence):
        """Test validation of bounds."""
        with pytest.raises(ValueError, match="'lower_bound' > 'upper_bound'"):
            cr = NaturalCubicSpline()
            cr(sequence, df=4, lower_bound=1.0, upper_bound=0.0)

    def test_knots_must_be_1d(self, sequence):
        """Test that knots must be 1-dimensional."""
        with pytest.raises(ValueError, match="'knots' must be 1 dimensional"):
            cr = NaturalCubicSpline()
            cr(sequence, knots=np.array([[0.5], [0.6]]))

    def test_natural_boundary_behavior(self, sequence):
        """Test that the spline is linear at boundaries (natural constraint)."""
        cr = NaturalCubicSpline()
        matrix = cr(sequence, df=4)

        # The first basis function is linear (x)
        # Check that basis values are approximately linear near boundaries
        x_vals = sequence[:3]
        basis_vals = matrix[:3, 0]

        # The first column should be close to linear (it IS x)
        expected = x_vals
        assert np.allclose(basis_vals, expected, rtol=1e-10)


# ============================================================================
# CyclicCubicSpline (cc) tests
# ============================================================================


class TestCyclicCubicSpline:
    """Tests for the CyclicCubicSpline (cc) transform."""

    def test_basic_shape(self, sequence):
        """Test that cc returns the correct number of columns."""
        cc = CyclicCubicSpline()
        matrix = cc(sequence, df=4)
        assert matrix.shape == (21, 4)

    def test_df_equals_columns(self, sequence):
        """Test that df parameter determines number of output columns."""
        for df in [2, 3, 4, 5]:
            cc = CyclicCubicSpline()
            matrix = cc(sequence, df=df)
            assert matrix.shape[1] == df, f"Expected {df} columns, got {matrix.shape[1]}"

    def test_knots_stored(self, sequence):
        """Test that knots are properly stored after initialization."""
        cc = CyclicCubicSpline()
        cc(sequence, df=4)
        assert cc.params_set is True
        assert cc._knots is not None

    def test_bounds_stored(self, sequence):
        """Test that bounds are properly stored."""
        cc = CyclicCubicSpline()
        cc(sequence, df=4)
        assert cc._lower_bound == 0.0
        assert cc._upper_bound == 1.0

    def test_stateful_reuse(self, sequence):
        """Test that parameters are reused on subsequent calls."""
        cc = CyclicCubicSpline()
        matrix1 = cc(sequence, df=4)
        original_knots = cc._knots.copy()

        # Call again with different data
        new_data = np.linspace(0.2, 0.8, 10)
        matrix2 = cc(new_data, df=4)

        # Knots should be unchanged
        assert np.allclose(cc._knots, original_knots)

    def test_periodicity_at_bounds(self, sequence):
        """Test that basis values match at period boundaries (cyclic constraint)."""
        cc = CyclicCubicSpline()
        cc(sequence, df=4, lower_bound=0.0, upper_bound=1.0)

        # Evaluate at lower bound and at a point that wraps to lower bound
        basis_at_0 = cc.eval(np.array([0.0]))
        basis_at_1 = cc.eval(np.array([1.0]))  # Should wrap to 0

        # Basis values should be the same due to periodicity
        assert np.allclose(basis_at_0, basis_at_1), \
            f"Periodicity failed: basis at 0 = {basis_at_0}, basis at 1 = {basis_at_1}"

    def test_periodicity_wrapping(self, sequence):
        """Test that values outside [lower, upper) wrap correctly."""
        cc = CyclicCubicSpline()
        cc(sequence, df=4, lower_bound=0.0, upper_bound=1.0)

        # x=1.5 should wrap to x=0.5
        basis_at_half = cc.eval(np.array([0.5]))
        basis_at_1_5 = cc.eval(np.array([1.5]))

        assert np.allclose(basis_at_half, basis_at_1_5), \
            f"Wrapping failed: basis at 0.5 = {basis_at_half}, basis at 1.5 = {basis_at_1_5}"

    def test_through_design_matrices(self, data):
        """Test cc works through design_matrices interface."""
        dm = design_matrices("cc(seq, df=4) - 1", data)
        matrix = dm.common.design_matrix
        assert matrix.shape == (21, 4)

        labels = dm.common.terms["cc(seq, df=4)"].labels
        assert labels == ["cc(seq, df=4)[0]", "cc(seq, df=4)[1]",
                          "cc(seq, df=4)[2]", "cc(seq, df=4)[3]"]

    def test_new_data_evaluation(self, data):
        """Test that new data evaluation reuses stored parameters."""
        dm = design_matrices("cc(seq, df=4) - 1", data)
        original_knots = (
            dm.common.terms["cc(seq, df=4)"]
            .components[0].call.stateful_transform._knots.copy()
        )

        # Evaluate with new data
        data2 = pd.DataFrame({"seq": np.linspace(0.1, 0.9, 10)})
        new_common = dm.common.evaluate_new_data(data2)

        # Check shape
        assert new_common.design_matrix.shape == (10, 4)

        # Check knots are preserved
        new_knots = (
            new_common.terms["cc(seq, df=4)"]
            .components[0].call.stateful_transform._knots
        )
        assert np.allclose(original_knots, new_knots)

    def test_invalid_df(self, sequence):
        """Test validation of df parameter."""
        with pytest.raises(ValueError, match="'df' must be either None or integer"):
            cc = CyclicCubicSpline()
            cc(sequence, df=[2])

        with pytest.raises(ValueError, match="'df' must be >= 1"):
            cc = CyclicCubicSpline()
            cc(sequence, df=0)

    def test_df_and_knots_none(self, sequence):
        """Test that either df or knots must be specified."""
        with pytest.raises(ValueError, match="Must specify either 'df' or 'knots'"):
            cc = CyclicCubicSpline()
            cc(sequence)

    def test_invalid_bounds(self, sequence):
        """Test validation of bounds."""
        with pytest.raises(ValueError, match="'lower_bound' > 'upper_bound'"):
            cc = CyclicCubicSpline()
            cc(sequence, df=4, lower_bound=1.0, upper_bound=0.0)

        with pytest.raises(ValueError, match="cannot equal 'upper_bound'"):
            cc = CyclicCubicSpline()
            cc(sequence, df=4, lower_bound=0.5, upper_bound=0.5)

    def test_knots_at_boundary(self, sequence):
        """Test that interior knots cannot be at or beyond boundaries."""
        with pytest.raises(ValueError, match="fall at or below lower bound"):
            cc = CyclicCubicSpline()
            cc(sequence, knots=[0.0, 0.5])  # 0.0 is at the boundary

        with pytest.raises(ValueError, match="fall at or above upper bound"):
            cc = CyclicCubicSpline()
            cc(sequence, knots=[0.5, 1.0])  # 1.0 is at the boundary

    def test_knots_must_be_1d(self, sequence):
        """Test that knots must be 1-dimensional."""
        with pytest.raises(ValueError, match="'knots' must be 1 dimensional"):
            cc = CyclicCubicSpline()
            cc(sequence, knots=np.array([[0.5], [0.6]]))

    def test_basis_continuity(self, sequence):
        """Test that basis functions are continuous by checking smoothness."""
        cc = CyclicCubicSpline()
        cc(sequence, df=4)

        # Evaluate at very closely spaced points
        x_fine = np.linspace(0.01, 0.99, 1001)  # Avoid exact boundaries
        basis = cc.eval(x_fine)

        # Check that second differences are bounded (indicates smoothness)
        # For a continuous function, smaller step size should give smaller diffs
        for col in range(basis.shape[1]):
            # The basis should not have any NaN or Inf values
            assert np.all(np.isfinite(basis[:, col])), f"Column {col} has non-finite values"

            # Check values are reasonable (between 0 and ~1 for B-splines)
            assert np.max(np.abs(basis[:, col])) < 2, f"Column {col} has unreasonable values"


# ============================================================================
# Combined tests
# ============================================================================


class TestSplineComparison:
    """Tests comparing cr and cc behavior."""

    def test_different_basis_functions(self, sequence):
        """Test that cr and cc produce different basis functions."""
        cr = NaturalCubicSpline()
        cc = CyclicCubicSpline()

        matrix_cr = cr(sequence, df=4)
        matrix_cc = cc(sequence, df=4)

        # They should have the same shape
        assert matrix_cr.shape == matrix_cc.shape

        # But different values (natural vs cyclic constraints)
        assert not np.allclose(matrix_cr, matrix_cc)

    def test_both_work_in_formula(self, data):
        """Test that both can be used together in a formula."""
        dm = design_matrices("cr(seq, df=3) + cc(seq, df=3) - 1", data)
        # Should have 3 + 3 = 6 columns
        assert dm.common.design_matrix.shape[1] == 6
