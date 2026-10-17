from formulae.terms.terms import Model

from formulae.scanner import Scanner
from formulae.parser import Parser
from formulae.resolver import Resolver


def model_description(formula, formula_wrappers=None):
    """Interpret model formula and obtain a model description.

    This function receives a string with a formula describing a statistical
    model and returns an object of class `Model` that describes the
    model interpreted from the formula.

    Parameters
    ----------
    formula : str
        A string with a model description in formula language.
    formula_wrappers : dict, optional
        Explicitly registered wrapper names, mapped to ``"coordinate"`` or ``"coefficient"``.
        Only these calls interpret their first argument as a formula expression.

    Returns
    -------
    An object of class `Model` with an internal description of the model.
    """

    description = Resolver(
        Parser(Scanner(formula).scan()).parse(), formula_wrappers=formula_wrappers
    ).resolve()

    if isinstance(description, Model):
        return description

    return Model(description)
