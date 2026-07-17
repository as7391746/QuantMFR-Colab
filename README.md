# QuantMFR — Colab figure replications

One-click reproductions of the numerically-computed figures in *Risk,
Uncertainty, and Value* (Hansen, Sargent). One notebook per figure group;
each opens and runs on a free Colab runtime.

Every notebook follows the same **model → solve → plot** pattern, and uses
only the book's own materials:

- the **model** is stated in the chapter's notation, with the parameters of
  the chapter appendix;
- the **solve** step calls the book's expansion code (`uncertain_expansion`)
  directly — fetched from the
  [RiskUncertaintyValue](https://github.com/lphansen/RiskUncertaintyValue)
  repository and used exactly as the book's *Uncertainty Expansion —
  Computation Process* appendix uses it;
- the **plots** render the figures.

Nothing outside the chapter and the book's own code is used.

## Figures

| Chapter | Figures | Notebook | Open in Colab |
|---|---|---|---|
| 11 | 11.1–11.3 — AK economy, shock elasticities | [`colab.ipynb`](colab.ipynb) | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as7391746/QuantMFR-Colab/blob/main/colab.ipynb) |
| 11 | 11.4–11.9 — habit preferences | [`ch11_habit.ipynb`](ch11_habit.ipynb) | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as7391746/QuantMFR-Colab/blob/main/ch11_habit.ipynb) |

More chapters follow, one notebook per figure group, on the same pattern.

## Layout

```
colab.ipynb          # Chapter 11, Figures 11.1–11.3  (stable link — do not rename)
ch11_habit.ipynb     # Chapter 11, Figures 11.4–11.9
assets/              # title-cell pipeline diagram
generators/          # the scripts that emit the notebooks (not needed to run them)
```

Each notebook is self-contained: opening the Colab link and running all
cells fetches the expansion code, solves, and draws the figures — no local
setup. To add a chapter, add a notebook (and a generator, by convention)
and a row to the table above.

> **Status**: internal demonstration mirror of the QuantMFR book's
> (private) companion material, published here temporarily so the demos are
> one click while the book repository is private. Not linked from the
> published book.
