# Modifications relative to upstream — audit record

**Upstream baseline**: `lphansen/RiskUncertaintyValue`, branch
`Planners_with_External`, commit `09ca5df`; engine file
`src/uncertain_expansion_faisal_feb26.py` (the branch's robust
steady-state variant). **Candidate under review: v2 (this folder).**
The Lagrangian first-order-condition generalization ("v3") is a
**prototype in a separate sandbox — proposed, not part of v2**; it is
recorded in category 3 below so the two are never conflated. File hashes
for this frozen v2 are listed at the end.

## 1. Entry layer (implemented — v2)

| ID | Upstream behavior | Modification | Mathematical effect | Files | Validation | Status |
|---|---|---|---|---|---|---|
| E1 | `initial_guess` is a mandatory hand-built vector in the solver's internal ordering | `initial_guess=None` derives the vector from the model (see `AUTO_GUESS.md`) | none on solve mathematics; adds a default | `auto_guess.py` (new) + 9-line entry patch in the engine file | book AK solves cold at γ=8.001, D2=0.019016 | implemented |
| E2 | user assembles variable lists / ordered args by convention | `spec_pack` assembles engine arguments from the four model pieces; name guards | none | `models_sourced.py` | six economies declared and solved | implemented |

## 2. Initialization and globalization (implemented — v2, `autosolve.py`)

| ID | Upstream behavior | Modification | Mathematical effect | Files | Validation | Status |
|---|---|---|---|---|---|---|
| G1 | none (hand guess only) | coordinate-wise constructed starting vector + feasibility growth ladder | selects the starting point; solve unchanged | `autosolve.py::derive_guess` | ablation table, `NUMERICAL_VALIDATION.md` | implemented |
| G2 | none | multi-start over unpinned states (grid restarts, 10-min budget) | selects starting points | `autosolve.py::_multistart` | KL/ACL cases | implemented |
| G3 | none | optional paper closed-form seeds (`state_seeds`) | starting hint only; results = consistency checks vs the paper | `autosolve.py` | seeded anchors ~1e-10..1e-11 | implemented |
| G4 | manual warm-start chains | coordinate-wise continuation with bisection + warm starts | path to hard targets; each solve unchanged | `autosolve.py` | habit γ=8, λ=.67, τ=.01 fully auto | implemented |
| G5 | fallback may return sub-tolerance roots | gray-zone re-polish: re-solve warm-started at the returned root | precision only | `autosolve.py::_solve_checked` | two-capital cells | implemented |
| G6 | `signal.SIGALRM` timeout (crashes on Windows) | portability guard (no-op where SIGALRM missing) | none | `autosolve.py::_time_limit` | AK regression after patch | implemented |

## 3. Equation construction (**prototype — proposed, NOT in v2**)

| ID | Upstream behavior | Modification | Mathematical effect | Files | Validation | Status |
|---|---|---|---|---|---|---|
| C1 | first-order conditions assembled from a single-multiplier template (assumes one static constraint, all controls with unit coefficient, state-free) | one multiplier per constraint × the constraints' actual symbolic derivatives; three dimension fixes (`n_J`, `n_C`, H/L padding) | **changes the compiled equation system** for models outside the template's class; book-class models reproduce the template exactly | sandbox copy only | book models bit-identical; KL stated naively solves to its true root; Tallarini with labor choice: χ=1 ⇒ N=0.2304 (his anchor 0.2305); Croce (JME 2014) baseline with elastic labor and two constraints: N=0.180000 exactly | **proposed** (awaiting review) |

## 4. Outputs and diagnostics

| ID | Upstream behavior | Modification | Mathematical effect | Files | Validation | Status |
|---|---|---|---|---|---|---|
| D1 | solver's own convergence report | dual residual gates: the model's deterministic equations AND the engine's complete steady-state system, both to 1e-6 | acceptance only | `autosolve.py` | all reported solutions | implemented |
| D2 | none | multi-start uniqueness census (converged-root clustering) | diagnostic only | sandbox | KL demonstration | proposed |
| D3 | per-notebook hand assembly of SDF / increments | generic `response(sol, of=expr, kind=...)` | none (wraps existing elasticity code) | sandbox | smoke tests | proposed |
| D4 | silent or opaque failures | explicit diagnostics (TIMEOUT vs ill-behaved; continuation stall reports with the parameter interval) | none | `autosolve.py` | transversality boundary case (ρ ≤ 0.65) | implemented |

## What is unchanged

The second-order expansion, the change-of-measure (μ₀) iteration, the
Schur decomposition, the elasticity computations, and every line of solver
numerics are the upstream engine's. In v2 the engine file differs from
upstream only by the 9-line entry patch (E1) and removed unreachable code;
`PROVENANCE.md` lists the byte-level differences.

## Known limitations

1. The growth target $\bar g$ and its ladder are heuristics; the
   equal-investment closing rule for extra capitals is a heuristic.
2. Unpinned ratio states require multi-start or a paper seed; nothing in
   the one-dimensional construction can pin their scale.
3. **The residual gates certify convergence, not specification**: both
   gates evaluate the same compiled system the solver just solved. A
   mis-assembled system (a model outside the v2 constraint class) can
   converge to a point that passes both. What prevents that in v2 is
   stating the model in output shares — a modeling requirement, not a
   check. The proposed C1 removes the class restriction at the source.
4. Results seeded by a paper's closed forms are consistency checks against
   that paper, not independent validation. Independent anchors are flagged
   as such in `NUMERICAL_VALIDATION.md`.
5. Solve-time budgets must scale with model size: with the default 90 s
   budget, large models (two-capital ACL at high risk aversion) time out
   even though the iteration converges — an artifact once misread as an
   "extreme uncertainty aversion failure" (see `NUMERICAL_VALIDATION.md`).

## File hashes (SHA-256) of frozen v2

(generated at freeze time; see `support_material/HASHES.txt`)
