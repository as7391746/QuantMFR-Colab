# `v2/` — an automatic starting-point and verification layer for `uncertain_expansion`

**Purpose.** Let a user state a production economy — the four pieces the
chapter itself uses ($\kappa$, capital growth, state dynamics, resource
constraint) plus a parameter dictionary — and solve it with the book's
expansion engine **without hand-tuning a starting point**. The solver
mathematics is untouched; the layer decides where to start, how far to
escalate, and whether to accept.

**Upstream baseline.** `lphansen/RiskUncertaintyValue`, branch
`Planners_with_External`, commit `09ca5df` (engine file
`uncertain_expansion_faisal_feb26.py`, the robust steady-state variant).
Byte-level differences from that baseline: `PROVENANCE.md`. Frozen file
hashes: `support_material/HASHES.txt`.

**Conclusions, in brief.**

1. Six economies — the book's AK and habit models plus
   Kaltenbrunner–Lochstoer (RFS 2010), Ai–Croce–Li (RFS 2013, two
   capitals), Croce (JME 2014, fixed-labor WP version), and Tallarini
   (JME 2000, χ=100) — solve at their published preference targets from
   automatically constructed starting points, each checked against
   steady-state restrictions from the papers themselves.
2. The constructed guess is a **feasible, model-consistent starting
   vector, not a solution**; multi-start handles the ratio states only the
   Euler equation can pin; optional paper seeds act as hints, and results
   obtained with them are labeled consistency checks.
3. The acceptance gates certify **convergence, not specification**; the v2
   model class requires the resource constraint in output-share form. The
   Lagrangian generalization that removes this restriction (and unlocks
   labor choice) is a **proposal**, prototyped and validated separately —
   it is not part of v2.
4. Extreme uncertainty aversion is not a mathematical obstacle for these
   calibrations; historical "extreme-ξ failures" trace to starting points
   and solve-time budgets.

**How to read this folder.**

| file | contents |
|---|---|
| [`MODIFICATIONS.md`](MODIFICATIONS.md) | the audit record: every change relative to upstream, its mathematical effect, validation, and status (implemented vs proposed); what is unchanged; known limitations |
| [`AUTO_GUESS.md`](AUTO_GUESS.md) | the mathematical definition of the automatic starting point, the algorithm, and the three classes of coordinates (derived / unpinned / normalized) |
| [`NUMERICAL_VALIDATION.md`](NUMERICAL_VALIDATION.md) | the merit-function geometry figure (KL), the six-model anchor record, and the ablation table separating each stage's contribution |
| [`PROVENANCE.md`](PROVENANCE.md) | byte-level differences from the upstream engine copy |
| `../v2_demo.ipynb` | the executable record: model declarations with notation glossary and calibrated/derived tags on every parameter, the solves, the robustness battery, elasticity paths |
| `support_material/` | the figure script, grid data, ablation results, file hashes |

**The one figure to look at first** —
`support_material/kl_basin_found.png`: the merit-function geometry of the
Kaltenbrunner–Lochstoer model. A plateau where the unseeded guess lands, a
narrow Euler-compatible valley, domain gaps, and the lowest sampled point
one grid cell from the paper's closed-form steady state — obtained by grid
evaluation and a linear multiplier least-squares step, with no nonlinear
root solve. It shows the numerical geometry of the problem — why
initialization and globalization matter. It is not a convexity claim.
