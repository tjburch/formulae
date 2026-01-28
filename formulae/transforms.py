# pylint: disable=invalid-name
import inspect

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from scipy.interpolate import splev

from formulae.categorical import CategoricalBox, Sum, Treatment

TRANSFORMS = {}


def is_class_callable(cls):
    members = (member[0] for member in inspect.getmembers(cls))
    return "__call__" in members


# Stateful transformations.
# These transformations have memory about the state of parameters that are
# required to compute the transformation and are obtained as a subproduct of the
# data that is used to compute the transform.
def register_stateful_transform(cls):
    assert isinstance(cls, type), "Can only decorate classes"
    assert is_class_callable(cls), "The class must implement a __call__ method"
    key = cls.__transform_name__ if hasattr(cls, "__transform_name__") else cls.__name__
    cls.__stateful_transform__ = True
    TRANSFORMS[key] = cls
    return cls


@register_stateful_transform
class Center:
    __transform_name__ = "center"

    def __init__(self):
        self.params_set = False
        self.mean = None

    def __call__(self, x):
        if not self.params_set:
            self.mean = np.mean(x)
            self.params_set = True
        return x - self.mean


@register_stateful_transform
class Scale:
    __transform_name__ = "scale"

    def __init__(self):
        self.params_set = False
        self.mean = None
        self.std = None

    def __call__(self, x):
        if not self.params_set:
            self.mean = np.mean(x)
            self.std = np.std(x)
            self.params_set = True
        return (x - self.mean) / self.std


# The following are just regular functions that are made available
# in the environment where the formula is evaluated.
def I(x):
    """Identity function. Returns its argument as it is.

    This allows to call Python code within the formula interface.
    This is an allias for ``{x}``, which does exactly the same, but in a more concise manner.

    Examples
    ----------

    >>> x + I(x**2)
    >>> x + {x**2}
    >>> {(x + y) / z}
    """
    return x


def C(data, contrast=None, levels=None):
    if isinstance(data, CategoricalBox):
        if contrast is None:
            contrast = data.contrast
        if levels is None:
            levels = data.levels
        data = data.data
    return CategoricalBox(data, contrast, levels)


def S(data, omit=None, levels=None):
    """Convert to categorical using Treatment encoding

    It is a shorthand for C(x, Sum)
    """
    return CategoricalBox(data, Sum(omit), levels)


def T(data, ref=None, levels=None):
    """Convert to categorical using Treatment encoding

    It is a shorthand for C(x, Treatment)
    """
    return CategoricalBox(data, Treatment(ref), levels)


def binary(x, success=None):
    """Make a variable binary

    Parameters
    ----------
    x: pd.Series
        The object containing the variable to be converted to binary.
    success: str, numeric or None
        The success level. When the variable is equal to this level, the binary variable is 1.
        All the rest are 0. Defaults to ``None`` which means formulae is going to sort all the
        values in the variable and pick the first one as success.

    Returns
    -------
    x: np.array
        A 0-1 numpy array with shape ``(n, 1)`` where ``n`` is the number of observations.
    """
    if success is None:
        categories = sorted(x.unique().tolist())
        success = categories[0]
    booleans = x == success
    if not sum(booleans):
        raise ValueError(f"No value in 'x' is equal to \"{success}\"")
    return np.where(booleans, 1, 0)


class Proportion:
    """Representation of a proportion term.

    Parameters
    ----------
    successes: ndarray
        1D array containing data with ``int`` dtype.
    trials: ndarray
        1D array containing data with ``int`` dtype. Its values must be equal or larger than the
        values in ``successes``
    trials_type: str
        Indicates whether ``trials`` is a constant value or not. It can be either ``"constant"``
        or ``"variable"``.
    """

    def __init__(self, successes, trials, trials_type):
        if not (np.mod(successes, 1) == 0).all():
            raise ValueError("'successes' must be a collection of integer numbers")

        if not (np.mod(trials, 1) == 0).all():
            raise ValueError("'trials' must be a collection of integer numbers")

        if not (np.less_equal(successes, trials)).all():
            raise ValueError("'successes' cannot be greater than 'trials'")

        self.successes = successes
        self.trials = trials
        self.trials_type = trials_type

    def eval(self):
        return np.vstack([self.successes, self.trials]).T


def proportion(successes, trials):
    """Create a term that represents the proportion ``successes/trials``.

    This function is actually a wrapper of class ``Proportion`` that checks its arguments.

    Parameters
    ----------
    successes: pd.Series
        The number of successes for each observation unit.
    trials: pd.Series or int
        The number of trials for each observation unit. If ``int``, this function internally
        generates an array of the same length than ``successes``.
    """
    # If this function does not receive a pd.Series, it means the user didn't pass a name in the
    # formula interface

    if not isinstance(successes, pd.Series):
        raise ValueError("'successes' must be a variable name.")
    successes = successes.values

    if isinstance(trials, pd.Series):
        trials = trials.values
        trials_type = "variable"
    elif isinstance(trials, int):
        trials = np.ones(len(successes), dtype=int) * trials
        trials_type = "constant"
    else:
        raise ValueError("'trials' must be a variable name or an integer.")

    return Proportion(successes, trials, trials_type)


class Offset:
    def __init__(self, x):
        self.size = None
        if not (is_numeric_dtype(x) or isinstance(x, (int, float))):
            raise ValueError("offset() can only be used with numeric variables.")

        if isinstance(x, pd.Series):
            self.x = x.values
            self.kind = "variable"
        elif isinstance(x, (int, float)):
            self.x = x
            self.kind = "constant"
        else:
            raise ValueError("'x' must be a variable name or a number.")

    def eval(self):
        if self.kind == "variable":
            return self.x.flatten()
        else:
            return np.ones((self.size, 1)) * self.x

    def set_size(self, size):
        self.size = size


def offset(x):
    return Offset(x)


@register_stateful_transform
class BSpline:
    """B-Spline representation

    Generates a B-spline basis for ``x``, allowing non-linear fits. The usual
    usage is something like::

        y ~ 1 + bs(x, 4)

    to fit ``y`` as a smooth function of ``x``, with 4 degrees of freedom
    given to the smooth.

    Parameters
    ----------
    x: 1D array-like
        The data.
    df: The number of degrees of freedom to use for this spline. The return value will have this
        many columns. You must specify at least one of ``df`` and ``knots``.
    knots: 1D array-like or None
        The interior knots to use for the spline. If unspecified, then equally spaced quantiles of
        the input data are used. You must specify at least one of ``df`` and ``knots`
    degree: int
        Degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept: bool
        If ``True``, an intercept is included in the basis. Default is ``False``.
    lower_bound:
        The lower exterior knot location.
    upper_bound:
        The upper exterior knot location.
    """

    __transform_name__ = "bs"

    def __init__(self):
        self.params_set = False
        self._intercept = None
        self._degree = None
        self._knots = None

    def __call__(
        self, x, df=None, knots=None, degree=3, intercept=False, lower_bound=None, upper_bound=None
    ):
        if not self.params_set:
            self._initialize(x, df, knots, degree, intercept, lower_bound, upper_bound)
        return self.eval(x)

    def _initialize(self, x, df, knots, degree, intercept, lower_bound, upper_bound):

        if not isinstance(degree, int):
            raise ValueError(f"'degree' must be an integer, not {type(degree)}")

        if degree < 0:
            raise ValueError(f"'degree' must be greater than 0, not {degree}")

        if df is None and knots is None:
            raise ValueError("Must specify either 'df' or 'knots'")

        if df and not isinstance(df, int):
            raise ValueError("'df' must be either None or integer")
        # XTODO: Check the type of knots.

        order = degree + 1

        if df is not None:
            n_inner_knots = df - order
            if not intercept:
                n_inner_knots += 1
            if n_inner_knots < 0:
                # We know that n_inner_knots is negative;
                # If df were that much larger, it would have been zero, and things would work.
                raise ValueError(
                    f"df={df} is too small for degree={degree} and intercept={intercept}; "
                    f"it must be >= {df - n_inner_knots}"
                )

            # User specified 'df' AND 'knots'
            if knots is not None:
                if len(knots) != n_inner_knots:
                    raise ValueError(
                        f"df={df} with degree={degree} implies {n_inner_knots} knots; "
                        f"but {len(knots)} were provided"
                    )
            # User specified 'df' but NOT 'knots'
            else:
                knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
                inner_knots = np.percentile(x, 100 * np.asarray(knot_quantiles))

        if knots is not None:
            inner_knots = knots

        if lower_bound is None:
            lower_bound = np.min(x)

        if upper_bound is None:
            upper_bound = np.max(x)

        if lower_bound > upper_bound:
            raise ValueError(f"'lower_bound' > 'upper_bound' ({lower_bound} > {upper_bound})")

        # NOTE: We need to clean the logic that creates 'inner_knots'.
        inner_knots = np.asarray(inner_knots)  # pylint: disable=used-before-assignment
        if inner_knots.ndim > 1:
            raise ValueError("'knots' must be 1 dimensional")

        if np.any(inner_knots < lower_bound):
            raise ValueError(
                f"Some knot values {inner_knots[inner_knots < lower_bound]} "
                f"fall below lower bound {lower_bound}"
            )

        if np.any(inner_knots > upper_bound):
            raise ValueError(
                f"Some knot values {inner_knots[inner_knots > upper_bound]} "
                f"fall above upper bound {upper_bound}"
            )

        all_knots = np.concatenate(([lower_bound, upper_bound] * order, inner_knots))
        all_knots.sort()

        self._intercept = intercept
        self._degree = degree
        self._knots = all_knots
        self.params_set = True

    def eval(self, x):
        n_bases = len(self._knots) - (self._degree + 1)
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        for i in range(n_bases):
            coefs = np.zeros((n_bases,))
            coefs[i] = 1
            basis[:, i] = splev(x, (self._knots, coefs, self._degree))

        if not self._intercept:
            basis = basis[:, 1:]
        return basis


@register_stateful_transform
class NaturalCubicSpline:
    """Natural Cubic Regression Spline (cr)

    Generates a natural cubic spline basis for ``x``. Natural cubic splines have the
    constraint that the second derivative is zero at the boundaries, which provides
    more stable behavior for extrapolation.

    The usual usage is something like::

        y ~ 1 + cr(x, df=4)

    to fit ``y`` as a smooth function of ``x``, with 4 degrees of freedom
    given to the smooth.

    Parameters
    ----------
    x: 1D array-like
        The data.
    df: int or None
        The number of degrees of freedom to use for this spline. The return value will have this
        many columns. You must specify at least one of ``df`` and ``knots``.
    knots: 1D array-like or None
        The interior knots to use for the spline. If unspecified, then equally spaced quantiles of
        the input data are used. You must specify at least one of ``df`` and ``knots``.
    lower_bound: float or None
        The lower exterior knot location. Defaults to the minimum of ``x``.
    upper_bound: float or None
        The upper exterior knot location. Defaults to the maximum of ``x``.
    constraints: str or None
        Type of constraints. Currently only ``None`` is supported (natural spline constraints
        are always applied).

    Notes
    -----
    This is a stateful transform. Parameters computed from the first dataset (knots, bounds)
    are stored and reused when evaluating new data.

    Unlike mgcv's s() function, this provides only the basis matrix without automatic
    smoothing penalty estimation.
    """

    __transform_name__ = "cr"

    def __init__(self):
        self.params_set = False
        self._knots = None
        self._lower_bound = None
        self._upper_bound = None
        self._df = None

    def __call__(
        self, x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None
    ):
        if not self.params_set:
            self._initialize(x, df, knots, lower_bound, upper_bound, constraints)
        return self.eval(x)

    def _initialize(self, x, df, knots, lower_bound, upper_bound, constraints):
        if df is None and knots is None:
            raise ValueError("Must specify either 'df' or 'knots'")

        if df is not None and not isinstance(df, int):
            raise ValueError("'df' must be either None or integer")

        if df is not None and df < 1:
            raise ValueError(f"'df' must be >= 1, not {df}")

        if lower_bound is None:
            lower_bound = np.min(x)

        if upper_bound is None:
            upper_bound = np.max(x)

        if lower_bound > upper_bound:
            raise ValueError(f"'lower_bound' > 'upper_bound' ({lower_bound} > {upper_bound})")

        if df is not None:
            # For natural cubic splines, df = number of output columns (without intercept)
            # We need df + 1 total knots (2 boundary + df - 1 interior)
            n_inner_knots = df - 1
            if n_inner_knots < 0:
                n_inner_knots = 0

            if knots is not None:
                if len(knots) != n_inner_knots:
                    raise ValueError(
                        f"df={df} implies {n_inner_knots} interior knots; "
                        f"but {len(knots)} were provided"
                    )
            else:
                if n_inner_knots > 0:
                    knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
                    knots = np.percentile(x, 100 * np.asarray(knot_quantiles))
                else:
                    knots = np.array([])
            self._df = df
        else:
            # df not specified, compute from knots
            knots = np.asarray(knots)
            self._df = len(knots) + 1  # df = n_interior + 1

        if knots is not None:
            knots = np.asarray(knots)
            if knots.ndim > 1:
                raise ValueError("'knots' must be 1 dimensional")

            if len(knots) > 0:
                if np.any(knots < lower_bound):
                    raise ValueError(
                        f"Some knot values {knots[knots < lower_bound]} "
                        f"fall below lower bound {lower_bound}"
                    )

                if np.any(knots > upper_bound):
                    raise ValueError(
                        f"Some knot values {knots[knots > upper_bound]} "
                        f"fall above upper bound {upper_bound}"
                    )

        # All knots including boundaries
        all_knots = np.concatenate([[lower_bound], knots, [upper_bound]])
        all_knots.sort()

        self._knots = all_knots
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.params_set = True

    def eval(self, x):
        """Evaluate the natural cubic spline basis at values x."""
        knots = self._knots
        n_knots = len(knots)

        if n_knots < 2:
            raise ValueError("Need at least 2 knots for natural cubic spline")

        x = np.asarray(x).flatten()
        n = len(x)

        # Natural cubic spline basis construction using the ESL formulation
        # For K knots, we get K basis functions: N_1(x) = 1, N_2(x) = x, N_{k+2}(x) = d_k - d_{K-1}
        # We drop the intercept (N_1) to return K-1 columns

        def d_func(x, k, knots):
            """Compute d_k(x) for natural cubic spline basis."""
            t_k = knots[k]
            t_K = knots[-1]

            term1 = np.maximum(x - t_k, 0) ** 3
            term2 = np.maximum(x - t_K, 0) ** 3

            denom = t_K - t_k
            if denom == 0:
                return np.zeros_like(x)
            return (term1 - term2) / denom

        # Basis without intercept: x, d_0 - d_{K-2}, d_1 - d_{K-2}, ..., d_{K-3} - d_{K-2}
        # Total columns: 1 + (K-2) = K-1

        basis = np.empty((n, n_knots - 1), dtype=float)
        basis[:, 0] = x

        if n_knots > 2:
            d_Km1 = d_func(x, n_knots - 2, knots)
            for k in range(n_knots - 2):
                d_k = d_func(x, k, knots)
                basis[:, k + 1] = d_k - d_Km1

        return basis


@register_stateful_transform
class CyclicCubicSpline:
    """Cyclic Cubic Regression Spline (cc)

    Generates a cyclic cubic spline basis for ``x``. Cyclic splines enforce periodicity
    by requiring that the function value and its first two derivatives match at the
    boundaries.

    The usual usage is something like::

        y ~ 1 + cc(x, df=4)

    to fit ``y`` as a smooth periodic function of ``x``, with 4 degrees of freedom
    given to the smooth.

    Parameters
    ----------
    x: 1D array-like
        The data.
    df: int or None
        The number of degrees of freedom to use for this spline. The return value will have this
        many columns. You must specify at least one of ``df`` and ``knots``.
    knots: 1D array-like or None
        The interior knots to use for the spline. If unspecified, then equally spaced quantiles of
        the input data are used. You must specify at least one of ``df`` and ``knots``.
    lower_bound: float or None
        The lower exterior knot location (period start). Defaults to the minimum of ``x``.
    upper_bound: float or None
        The upper exterior knot location (period end). Defaults to the maximum of ``x``.
    constraints: str or None
        Type of constraints. Currently only ``None`` is supported.

    Notes
    -----
    This is a stateful transform. Parameters computed from the first dataset (knots, bounds)
    are stored and reused when evaluating new data.

    Unlike mgcv's s() function, this provides only the basis matrix without automatic
    smoothing penalty estimation.

    The cyclic constraint ensures that f(lower_bound) = f(upper_bound) and that the
    first two derivatives also match at these boundaries.
    """

    __transform_name__ = "cc"

    def __init__(self):
        self.params_set = False
        self._knots = None
        self._lower_bound = None
        self._upper_bound = None
        self._df = None

    def __call__(
        self, x, df=None, knots=None, lower_bound=None, upper_bound=None, constraints=None
    ):
        if not self.params_set:
            self._initialize(x, df, knots, lower_bound, upper_bound, constraints)
        return self.eval(x)

    def _initialize(self, x, df, knots, lower_bound, upper_bound, constraints):
        if df is None and knots is None:
            raise ValueError("Must specify either 'df' or 'knots'")

        if df is not None and not isinstance(df, int):
            raise ValueError("'df' must be either None or integer")

        if df is not None and df < 1:
            raise ValueError(f"'df' must be >= 1, not {df}")

        if lower_bound is None:
            lower_bound = np.min(x)

        if upper_bound is None:
            upper_bound = np.max(x)

        if lower_bound > upper_bound:
            raise ValueError(f"'lower_bound' > 'upper_bound' ({lower_bound} > {upper_bound})")

        if lower_bound == upper_bound:
            raise ValueError("'lower_bound' cannot equal 'upper_bound' for cyclic splines")

        if df is not None:
            # For cyclic cubic splines, df = number of output columns
            # We use df knots total (including boundaries), so df - 2 interior knots
            n_inner_knots = df

            if knots is not None:
                if len(knots) != n_inner_knots:
                    raise ValueError(
                        f"df={df} implies {n_inner_knots} interior knots; "
                        f"but {len(knots)} were provided"
                    )
            else:
                # Place knots at equally spaced quantiles
                if n_inner_knots > 0:
                    knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
                    knots = np.percentile(x, 100 * np.asarray(knot_quantiles))
                else:
                    knots = np.array([])
            self._df = df
        else:
            knots = np.asarray(knots)
            self._df = len(knots)

        if knots is not None:
            knots = np.asarray(knots)
            if knots.ndim > 1:
                raise ValueError("'knots' must be 1 dimensional")

            if len(knots) > 0:
                if np.any(knots <= lower_bound):
                    raise ValueError(
                        f"Some knot values {knots[knots <= lower_bound]} "
                        f"fall at or below lower bound {lower_bound}"
                    )

                if np.any(knots >= upper_bound):
                    raise ValueError(
                        f"Some knot values {knots[knots >= upper_bound]} "
                        f"fall at or above upper bound {upper_bound}"
                    )

        # All knots including boundaries for cyclic spline
        all_knots = np.concatenate([[lower_bound], knots, [upper_bound]])
        all_knots.sort()

        self._knots = all_knots
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.params_set = True

    def eval(self, x):
        """Evaluate the cyclic cubic spline basis at values x."""
        knots = self._knots
        df = self._df

        if df < 1:
            raise ValueError("Need at least df=1 for cyclic cubic spline")

        x = np.asarray(x).flatten()
        n = len(x)

        # Period
        period = self._upper_bound - self._lower_bound

        # Wrap x to the period [lower_bound, upper_bound)
        x_wrapped = self._lower_bound + np.mod(x - self._lower_bound, period)

        # Build cyclic cubic spline basis using cyclic B-splines
        # For a cyclic spline with K knots (including endpoints), we have K-1 basis functions
        # But due to periodicity, we treat the boundary knots as coincident

        # Create extended knot sequence for periodic B-splines
        n_knots = len(knots)
        all_knots = np.array(knots)

        # For cyclic B-splines, we extend the knot sequence periodically
        extended_knots = np.concatenate([
            all_knots[:-1] - period,
            all_knots,
            all_knots[1:] + period
        ])

        # Compute B-spline basis of degree 3
        degree = 3
        basis = np.zeros((n, df), dtype=float)

        # The cyclic basis has df functions, corresponding to df interior knots
        for j in range(df):
            # Index in the extended knot sequence
            # Interior knot j is at position j+1 in all_knots
            # In extended_knots, it's at position (n_knots - 1) + (j + 1)
            idx = (n_knots - 1) + (j + 1) - degree
            basis[:, j] = self._eval_bspline_basis(x_wrapped, extended_knots, idx, degree)

        return basis

    def _eval_bspline_basis(self, x, knots, i, degree):
        """Evaluate B-spline basis function B_{i,degree} using de Boor's recursion."""
        n = len(x)
        result = np.zeros(n, dtype=float)

        # Use iterative de Boor algorithm for better numerical stability
        # Start with degree 0 B-splines
        basis_prev = {}

        for k in range(i, i + degree + 2):
            if k + 1 < len(knots):
                basis_prev[k] = np.where(
                    (x >= knots[k]) & (x < knots[k + 1]),
                    1.0,
                    0.0
                ).astype(float)
            else:
                basis_prev[k] = np.zeros(n, dtype=float)

        # Build up to desired degree
        for d in range(1, degree + 1):
            basis_curr = {}
            for k in range(i, i + degree + 1 - d + 1):
                left = np.zeros(n, dtype=float)
                right = np.zeros(n, dtype=float)

                if k + d < len(knots):
                    denom_left = knots[k + d] - knots[k]
                    if denom_left > 0 and k in basis_prev:
                        left = (x - knots[k]) / denom_left * basis_prev[k]

                if k + d + 1 < len(knots):
                    denom_right = knots[k + d + 1] - knots[k + 1]
                    if denom_right > 0 and (k + 1) in basis_prev:
                        right = (knots[k + d + 1] - x) / denom_right * basis_prev[k + 1]

                basis_curr[k] = left + right

            basis_prev = basis_curr

        if i in basis_prev:
            result = basis_prev[i]

        return result


@register_stateful_transform
class Polynomial:
    """Polynomial transformation

    The computation of this transformation is borrowed from the implementation in the
    Formulaic library written by Matthew Wardrop.

    The original implementation and more documentation can be found here:
    https://github.com/matthewwardrop/formulaic/blob/main/formulaic/transforms/poly.py

    Parameters
    ----------
    x: 1d array-like
        The data.
    degree: int
        The degree of the polynomial terms to compute. If degree is k, with k > 1, this
        transformation computes the polinomials x^1, x^2, ...x^k.
    raw: bool
        Whether to use raw polynomials or orthonormal ones. Defaults to False.
    """

    __transform_name__ = "poly"

    def __init__(self):
        self.params_set = False
        self.degree = 1
        self.raw = False
        self.alpha = {}
        self.norms2 = {}

    def __call__(self, x, degree=1, raw=False):
        if not self.params_set:
            self.degree = degree
            self.raw = raw
        return self.eval(x)

    def eval(self, x):
        if self.raw:
            return np.column_stack([np.power(x, k) for k in range(1, self.degree + 1)])

        def get_alpha(k):
            if k not in self.alpha:
                self.alpha[k] = np.sum(x * P[:, k] ** 2) / np.sum(P[:, k] ** 2)
            return self.alpha[k]

        def get_norm(k):
            if k not in self.norms2:
                self.norms2[k] = np.sum(P[:, k] ** 2)
            return self.norms2[k]

        def get_beta(k):
            return get_norm(k) / get_norm(k - 1)

        P = np.empty((x.shape[0], self.degree + 1))
        P[:, 0] = 1

        for i in range(1, self.degree + 1):
            P[:, i] = (x - get_alpha(i - 1)) * P[:, i - 1]
            if i >= 2:
                P[:, i] -= get_beta(i - 1) * P[:, i - 2]

        P /= np.array([np.sqrt(get_norm(k)) for k in range(0, self.degree + 1)])
        return P[:, 1:]


TRANSFORMS.update(
    {
        "B": binary,
        "binary": binary,
        "C": C,
        "I": I,
        "offset": offset,
        "p": proportion,
        "prop": proportion,
        "proportion": proportion,
        "S": S,
        "standardize": Scale,
        "T": T,
    }
)
