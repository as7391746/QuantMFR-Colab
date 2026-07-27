# `v2/` — an automatic initial guess for `uncertain_expansion`

**What changed, in one paragraph.** The book's expansion engine requires
the user to hand-build the starting vector for its steady-state solve —
in the solver's internal ordering, by trial and error, once per model.
v2 removes that requirement: an **automatic initial guess** constructs a
feasible, model-consistent starting vector from the model declaration
itself, and the solver mathematics is untouched. That construction is the
one substantive change. Everything else in this folder is auxiliary
machinery around it — a multi-start search for the few ratio states no
one-dimensional equation can pin, optional paper-derived seeds (used as
labeled comparators, never silently), a continuation path for hard
preference targets, and acceptance gates that verify every returned
steady state before it is reported.

**What it demonstrates.** Six published production economies — the book's
AK and habit models, Kaltenbrunner–Lochstoer (RFS 2010), Ai–Croce–Li
(RFS 2013, two capitals), Croce (JME 2014, fixed-labor 2008 WP version),
Tallarini (JME 2000, χ=100) — declared from their papers and solved at
their published preference targets. **Five of the six solve cold**, from
the constructed guess alone; the two-capital ACL model is the honest
exception — it needs the paper's closed-form seed, and all its results
are labeled consistency checks. Extreme risk aversion is not a
mathematical obstacle for these calibrations; historical failures trace
to starting points and solve budgets.

**How to read this folder.**

| file | contents |
|---|---|
| [`AUTO_GUESS.md`](AUTO_GUESS.md) | the whole story: the mathematical definition of the construction, the driver around it, the loss-landscape geometry of all six models, and the numerical validation (cold vs seeded, anchors, ablation) |
| [`PROVENANCE.md`](PROVENANCE.md) | technical traceability: byte-level differences from the upstream engine copy (`lphansen/RiskUncertaintyValue`, branch `Planners_with_External`, commit `09ca5df`); frozen hashes in `support_material/HASHES.txt` |
| `../v2_demo.ipynb` | the executable record: each economy with its own variable dictionary, equations in the paper's notation, calibration table with per-number provenance, cold and seed-assisted solves reported separately, the correlation battery, elasticity paths |
| `support_material/` | `landscape_*.png` + `landscapes_overview.png` (the six loss landscapes), `make_landscapes.py` + `landscape_*.npz` (fully reproducible), `ablation.json`, `HASHES.txt` |

**The one figure to look at first** —
`support_material/landscapes_overview.png`: the loss landscape of each
economy on its reconstruction manifold, with the unseeded guess (●), the
optional seed (◇), the lowest sampled residual (★), the verified steady
state (+), and — on the floor — actual engine solves started from each
region, classified by where they converge. It shows what the construction
has to find and where it starts. It is not a convexity claim.

*A Lagrangian generalization of the engine's first-order-condition
assembly (removing the single-constraint restriction and unlocking labor
choice) exists as a separately validated prototype; it is a proposal under
review and is deliberately **not** part of v2.*
