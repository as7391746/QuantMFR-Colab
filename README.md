# QuantMFR — Colab figure replications

One-click reproductions of the numerically-computed figures in *Risk,
Uncertainty, and Value* (Hansen, Sargent), plus an **automatic
initial-guess extension** of the book's expansion code (see below). One
notebook per figure group; each opens and runs on a free Colab runtime.

Every figure notebook follows the same **model → solve → plot** pattern,
and uses only the book's own materials:

- the **model** is stated in the chapter's notation, with the parameters of
  the chapter appendix;
- the **solve** step calls the book's expansion code (`uncertain_expansion`)
  directly — fetched from the
  [RiskUncertaintyValue](https://github.com/lphansen/RiskUncertaintyValue)
  repository and used exactly as the book's *Uncertainty Expansion —
  Computation Process* appendix uses it;
- the **plots** render the figures.

The figure notebooks use nothing outside the chapter and the book's own
code. The `v2/` extension additionally solves four production economies
from the published literature, each declared from its original paper.

## The automatic initial guess (`v2/`)

The book's expansion engine requires a hand-built starting vector for its
steady-state solve. [`v2/`](v2/) removes that requirement: the starting
vector is constructed from the model declaration itself, with the solver
mathematics untouched. Six published production economies are the test
bed; five solve from the constructed guess alone.

- [`v2/README.md`](v2/README.md) — what changed, in one page
- [`v2/AUTO_GUESS.md`](v2/AUTO_GUESS.md) — the construction, its
  mathematics, six 3D loss landscapes, and the numerical validation
- [`v2_demo.ipynb`](v2_demo.ipynb) — the executable record
  ([![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as7391746/QuantMFR-Colab/blob/main/v2_demo.ipynb))

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
v2_demo.ipynb        # six published economies solved via the automatic initial guess
v2/                  # the extension: engine copy + auto guess + docs + landscapes
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
