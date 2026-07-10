# SENTRY manuscript

Springer Nature journal-format manuscript for the SENTRY monitor, using the
official `sn-jnl` class (`sn-mathphys-num` numbered-reference style).

## Build

```bash
pdflatex sentry
bibtex   sentry
pdflatex sentry
pdflatex sentry
```

Produces `sentry.pdf` (24 pages). Requires a TeX Live installation with
`sn-jnl.cls` (included here), `algorithm`/`algpseudocode`, `booktabs`,
`tikz`, and `amsmath` (all standard).

## Files

| File | Purpose |
|---|---|
| `sentry.tex` | manuscript source |
| `sentry.pdf` | compiled output |
| `references.bib` | bibliography (29 refs; every entry has a DOI) |
| `sn-jnl.cls`, `sn-mathphys-num.bst`, `bst/` | Springer Nature class + styles |
| `figures/*.pdf` | figures |
| `make_figures.py` | regenerates every figure from the real results |

## Figures

`make_figures.py` regenerates all figures from the collected trajectories
(`../real_data/`) and the committed multi-seed results
(`../real_data/results.json`) — nothing is hand-drawn or invented:

```bash
source ../.venv/bin/activate
PYTHONPATH=.. python make_figures.py
```

## References

All 29 references were verified against the arXiv API (titles, full author
lists, dates) and Crossref (DOIs for the published statistics literature).
Every entry carries a DOI: `10.48550/arXiv.*` for preprints, publisher DOIs
for journal articles. Eight references (27%) are from 2026.
