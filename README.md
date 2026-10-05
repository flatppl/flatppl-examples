# FlatPPL Examples

Examples of FlatPPL, the Flat Portable Probabilistic Language.

## Model comparisons

[Two-level binary HGF](examples/hgf-binary-2level.md) compares a scan-based FlatPPL
model with the original HierarchicalGaussianFiltering.jl package and an independent score oracle.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Writing examples

Doc-comments render in the viewer and in the math view, so they follow one
style:

- A one-line `%` doc-comment is a caption shown beside its binding: sentence
  case, no trailing period — `% Observed data`, `% Forward kernel (observation
  model)`. A caption that is an expression stays as written.
- A `%%%` block is prose: full sentences, with capitals and periods.

## Funding

This work was supported by Germany's Federal Ministry of Research, Technology
and Space (BMFTR) within the ErUM-Data programme under grant FKZ 05D25PC1
(DEMOS consortium).

## License

[MIT](LICENSE)
