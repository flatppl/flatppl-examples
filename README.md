# FlatPPL Examples

Examples of FlatPPL, the Flat Portable Probabilistic Language.

- [`examples/`](examples): introductory and applied models.
- [`posteriordb/`](posteriordb): ten posteriorDB models with separate data files
  and independent Stan density checks.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Writing examples

The playground publishes the directories listed in `collections.json`, in list
order. Each record has a top-level directory `path` and a display `title`.
An optional `exclude` list hides collection-relative file paths from the sidebar.
The files remain available for imports and existing links.
Keep existing paths stable because playground links use the full file path.

Add a collection by creating a peer directory and adding its record to the list:

```json
{ "path": "posteriordb", "title": "PosteriorDB" }
```

Use letters, digits, underscores, and hyphens for collection paths, starting
with a letter or digit. Paths must be unique without regard to case. Keep the
`examples` entry. The names `demo`, `test-cases`, `new`, `user`, `vendor`, and
`theme` belong to the playground and cannot name collections.

The sidebar mirrors nested folders. The build copies support files alongside
models, and lists `.flatppl`, `.md`, `.markdown`, `.hs3.json`, `.pyhf.json`,
`.json`, `.csv`, and `.wsv` files. Unlisted repository directories are not published.
New content appears after the next playground build and deployment.

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
