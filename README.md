# QuantMFR Chapter 11 — Figures 11.1–11.3 replication

**One-click reproduction:**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as7391746/QuantMFR-Colab/blob/main/colab.ipynb)
(runs on your own Colab runtime; ~3 minutes for the thirteen expansion solves)

A replication of the three AK-economy shock-elasticity figures of *Risk,
Uncertainty, and Value* Chapter 11: the model is declared in the chapter's
notation with the parameters of the chapter appendix (converted
annual→quarterly explicitly), and solved with the expansion engine
`uncertain_expansion` imported directly from this repository.

Model — solve — plot:

- `colab.ipynb` — the deliverable: model declaration (chapter notation,
  appendix parameters), engine solve, one cell per figure. Regenerated
  from `make_notebook.py`.
- `src/` — the expansion engine, an unmodified snapshot of
  RiskUncertaintyValue (branch `Planners_with_External`, commit `09ca5df`),
  in the same `src/` layout as the upstream repository.
- `expansion/` — a small declaration layer over the engine (`Model`,
  parameter handling, automatic steady-state starting values, elasticity
  requests), plus worked examples (`ak_example.py`, `habit_example.py`)
  and a direct-import test notebook (`expansion_test.ipynb`).
- `assets/` — the pipeline diagram in the notebook's title cell
  (`method.tex` is the TikZ source).

> **Status**: internal demonstration mirror of the QuantMFR book's
> (private) companion material, published here temporarily so the demo is
> one click while the book repository is private. Not linked from the
> published book.
