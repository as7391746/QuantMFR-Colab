# QuantMFR Chapter 11 — figure replication

**One-click reproduction:**
- Figures 11.1–11.3 (baseline AK economy):
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as7391746/QuantMFR-Colab/blob/main/colab.ipynb)
- Figures 11.4–11.9 (habit preferences):
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as7391746/QuantMFR-Colab/blob/main/colab_habit.ipynb)

(each runs on your own Colab runtime.)

A replication of the Chapter 11 shock-elasticity figures of *Risk,
Uncertainty, and Value*. Both notebooks state the model in the chapter's
notation with the quarterly parameters of the chapter appendix, and solve
it by calling the book's expansion code (`uncertain_expansion`) directly —
fetched from the RiskUncertaintyValue repository and used exactly as the
book's *Uncertainty Expansion — Computation Process* appendix uses it.

- `colab.ipynb` — Figures 11.1–11.3 (the baseline AK economy).
  Regenerated from `make_notebook.py`.
- `colab_habit.ipynb` — Figures 11.4–11.9 (habit preferences), the same
  delivery pattern. Regenerated from `make_notebook_habit.py`.
- `assets/` — the pipeline diagram in the notebook's title cell
  (`method.tex` is the TikZ source).

> **Status**: internal demonstration mirror of the QuantMFR book's
> (private) companion material, published here temporarily so the demo is
> one click while the book repository is private. Not linked from the
> published book.
