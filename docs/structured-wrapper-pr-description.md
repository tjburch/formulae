# Add opt-in formula expressions inside registered wrappers

Registered formula wrappers can retain an inner coordinate expression or coefficient design instead of treating `1+x` as elementwise arithmetic. The bar and `by=` forms produce equivalent metadata, including scalar/composite groups and standard nesting expansion. Registration is explicit and local to `model_description` or `design_matrices`.

The change adds a list AST node and lazy call lists, structured metadata with fitted design evaluation, and variable discovery for quoted grouping columns. It preserves ordinary formula and namespace behavior. Declared categorical domains may include unobserved levels; incomplete declarations and duplicates are rejected.

The paired Bambi feature consumes this interface. A Formulae release containing this change is required before Bambi's release dependency minimum can be set. No covariance functions or inference algorithms are implemented here.

Validation commands and current results are recorded in the paired Bambi development validation report. The 162 focused parser/resolver/design/new-data regressions pass. The full Formulae suite is configured in the paired Bambi PR workflow against the exact Formulae feature commit; it has not yet run. Independent review identified and prompted regression coverage for canonical-name collisions, outer interactions, and grouping subset expressions.
