"""Opt-in formula expressions for Gaussian and other structured effects.

This module describes formulas and fitted designs. It does not define probability
models or covariance functions.
"""

from copy import deepcopy
from functools import reduce

import numpy as np

from formulae import expr as ast
from formulae.categorical import ENCODINGS
from formulae.environment import Environment
from formulae.token import Token
from formulae.terms.call_resolver import LazyCall, LazyList, LazyOperator, LazyVariable
from formulae.terms.terms import Term, Model, Intercept
from formulae.terms.variable import Variable
from formulae.transforms import TRANSFORMS

_LITERAL_NAMES = {"True", "False", "None"}


def _variable_name(name):
    return name if name.isidentifier() and name not in _LITERAL_NAMES else f"`{name}`"


def _call_name(value):
    if isinstance(value, LazyVariable):
        return _variable_name(value.name)
    if isinstance(value, LazyList):
        return "[" + ", ".join(_call_name(item) for item in value.args) + "]"
    if isinstance(value, LazyOperator):
        args = [_call_name(item) for item in value.args]
        if len(args) == 1:
            return f"({value.symbol}{args[0]})"
        return f"({args[0]} {value.symbol} {args[1]})"
    if isinstance(value, LazyCall):
        args = [_call_name(item) for item in value.args]
        args.extend(f"{key}={_call_name(item)}" for key, item in value.kwargs.items())
        return f"{value.callee}({', '.join(args)})"
    return str(value)


def _term_name(term):
    if isinstance(term, Intercept):
        return "1"
    return ":".join(
        (
            _variable_name(component.name)
            if isinstance(component, Variable)
            else _call_name(component.call)
        )
        for component in term.components
    )


class StructuredTerm(Term):  # pylint: disable=too-many-instance-attributes
    """A registered wrapper with an inner design and optional grouping columns.

    ``design`` retains ordinary Formulae encodings and stateful transformations.
    Coordinate wrappers expose raw ``coordinate`` values instead. ``group_values``
    stores the original columns, without combining or coercing their identifiers.
    The term occupies no columns of the ordinary common-effects matrix.
    """

    def __init__(self, wrapper, mode, inner, groups, options):
        super().__init__()
        self.wrapper = wrapper
        self.mode = mode
        self.inner = inner
        self.groups = tuple(groups)
        self.options = options
        self.kind = "structured"
        self.design = None
        self.coordinate = None
        self.group_values = None
        self.env = None
        names = [_term_name(term) for term in inner.common_terms]
        if mode == "coefficient" and "1" not in names:
            names.insert(0, "0")
        args = [" + ".join(names)]
        if groups:
            group = (
                groups[0]
                if len(groups) == 1 and groups[0].isidentifier() and groups[0] not in _LITERAL_NAMES
                else repr(list(groups))
            )
            args.append(f"by={group}")
        args.extend(f"{key}={value!r}" for key, value in sorted(options.items()))
        self.name = f"{wrapper}({', '.join(args)})"

    def __eq__(self, other):
        return isinstance(other, StructuredTerm) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        return Model(self) + other

    def __sub__(self, other):
        return Model(self) - other

    def __mul__(self, other):
        raise ValueError("A structured wrapper must be an additive term.")

    __matmul__ = __mul__
    __or__ = __mul__
    __truediv__ = __mul__
    __pow__ = __mul__

    @property
    def var_names(self):
        return self.inner.var_names | set(self.groups)

    @property
    def labels(self):
        return []

    def set_type(self, data, env):
        from formulae.matrices import CommonEffectsMatrix  # pylint: disable=import-outside-toplevel

        missing = set(self.groups) - set(data.columns)
        if missing:
            raise ValueError(f"Missing grouping columns: {sorted(missing)}")
        self.env = Environment([{**TRANSFORMS, **ENCODINGS}]).with_outer_namespace(env.namespace)
        self.group_values = data[list(self.groups)].copy()
        if self.mode == "coordinate":
            self.coordinate = self._coordinate_values(data)
        else:
            data = data.copy(deep=False)
            for name in data:
                if hasattr(data[name].dtype, "categories"):
                    data[name] = data[name].cat.as_ordered()
            self.inner.eval(data, env)
            self.design = CommonEffectsMatrix(self.inner.common_terms)
            self.design.evaluate(data, env)
        self.data = np.empty((len(data), 0))

    def _coordinate_values(self, data):
        term = self.inner.common_terms[0]
        component = term.components[0]
        if isinstance(component, Variable):
            return data[component.name]
        return component.call.eval(data, self.env)

    def set_data(self, _spans_intercept):
        """The fitted design and metadata are evaluated together in ``set_type``."""

    def eval_new_data(self, data):
        return np.empty((len(data), 0))

    def evaluate_new_data(self, data):
        """Return raw coordinates or the fitted coefficient basis, and group columns."""
        values = (
            self._coordinate_values(data)
            if self.mode == "coordinate"
            else (self.design.evaluate_new_data(data).design_matrix)
        )
        return values, data[list(self.groups)].copy()


def _unwrap(expression):
    while isinstance(expression, ast.Grouping):
        expression = expression.expression
    return expression


def _group_expression(expression, resolver):
    expression = _unwrap(expression)
    if isinstance(expression, ast.ListExpr):
        if not expression.items:
            raise ValueError("A grouping list cannot be empty.")
        return reduce(
            lambda left, right: left @ right,
            [_group_expression(item, resolver) for item in expression.items],
        )
    if isinstance(expression, ast.Literal) and isinstance(expression.value, str):
        return Term(Variable(expression.value))
    return expression.accept(resolver)


def _with_intercept(expression):
    expression = _unwrap(expression)
    if isinstance(expression, ast.Binary) and expression.operator.kind in ("PLUS", "MINUS"):
        return ast.Binary(_with_intercept(expression.left), expression.operator, expression.right)
    return ast.Binary(ast.Literal(1), Token("PLUS", "+"), expression)


def resolve_wrapper(expression, resolver, mode):
    """Resolve an opt-in call through the ordinary formula algebra."""
    if mode not in ("coordinate", "coefficient"):
        raise ValueError("Wrapper mode must be 'coordinate' or 'coefficient'.")
    positional, options = [], {}
    by = None
    for argument in expression.args:
        if isinstance(argument, ast.Assign):
            key = argument.name.name.lexeme
            if key in options or (key == "by" and by is not None):
                raise ValueError(f"Duplicate wrapper option '{key}'.")
            if key == "by":
                by = argument.value
            elif isinstance(argument.value, ast.Literal):
                options[key] = argument.value.value
            else:
                raise ValueError(f"Wrapper option '{key}' must be a literal.")
        else:
            positional.append(argument)
    if len(positional) != 1:
        raise ValueError("A formula wrapper requires exactly one formula expression.")
    inner_expression = _unwrap(positional[0])
    if mode == "coefficient":
        inner_expression = _with_intercept(inner_expression)
    inner = inner_expression.accept(resolver)
    inner = inner if isinstance(inner, Model) else Model(inner)
    if inner.group_terms:
        raise ValueError(
            "Use 'by=' for grouping a formula wrapper; inner group effects are unsupported."
        )
    if any(isinstance(t, StructuredTerm) for t in inner.common_terms):
        raise ValueError("Nested formula wrappers are not supported.")
    if mode == "coordinate":
        terms = inner.common_terms
        if len(terms) != 1 or not isinstance(terms[0], Term) or len(terms[0].components) != 1:
            raise ValueError("A coordinate wrapper requires one coordinate.")
    elif not inner.common_terms:
        raise ValueError("The coefficient design cannot be empty.")
    factors = [None]
    if by is not None:
        factor = _group_expression(by, resolver)
        factors = factor.common_terms if isinstance(factor, Model) else [factor]
    result = Model()
    for factor in factors:
        if factor is not None and (
            not isinstance(factor, Term)
            or not all(
                isinstance(c, Variable) and isinstance(c.name, str) and c.reference is None
                for c in factor.components
            )
        ):
            raise ValueError("Grouping requires column names or their interactions/nesting.")
        groups = [] if factor is None else [c.name for c in factor.components]
        result.add_term(
            StructuredTerm(expression.callee.name.lexeme, mode, deepcopy(inner), groups, options)
        )
    return result
