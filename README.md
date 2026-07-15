# QuantMFR Chapter 11 — Figures 11.1–11.3 replication

**One-click reproduction:**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as7391746/QuantMFR-Colab/blob/main/colab.ipynb)
(runs on your own Colab runtime; the full 13-scenario solve takes ~0.2 s)

A self-contained replication of the three AK-economy shock-elasticity
figures of *Risk, Uncertainty, and Value* Chapter 11, written from the
chapter's equations in the chapter's notation, at quarterly frequency, with
parameters following the chapter appendix and Hansen–Khorrami–Tourre (2024,
*Annu. Rev. Financ. Econ.* 16), converted annual→quarterly explicitly.

This repository serves exactly one purpose — that notebook:

- `colab.ipynb` — the deliverable: two sections, five code cells
  (model / solve / one per figure), fully self-contained, no downloads.
  Parameters are a plain `PARAMS` dictionary; edit it (or the equations)
  and re-run.
- `model.py`, `solve.py` — the same code kept as importable files;
  `make_notebook.py` splices them verbatim into the notebook's cells
  (edit the `.py`, regenerate, push — the notebook cannot drift).
- `assets/` — the pipeline diagram in the title cell
  (`method.tex` is the TikZ source).

Verification (numeric cross-check of all 27 curves against the run behind
the published figures, residual checks against the chapter's exact
recursions, and the parameter documentation) lives with the book's internal
bookshelf folder, not here.

> **Status**: internal demonstration mirror of one folder of the QuantMFR
> book's (private) companion "bookshelf", published here temporarily so the
> demo is one click while the book repository is private. Not linked from
> the published book.
