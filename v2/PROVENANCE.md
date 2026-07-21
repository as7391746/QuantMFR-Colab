# v2/ — provenance

Base: `lphansen/RiskUncertaintyValue`, branch `Planners_with_External`, commit `09ca5df`.
Engine file: `src/uncertain_expansion_faisal_feb26.py` (the branch's robust steady-state variant).
All solver mathematics is unchanged.

Changes relative to the branch copy:

1. `auto_guess.py` — NEW (165 lines). Derives a feasible initial guess from the model's
   own equations: deterministic fixed points of the state equations, inversion of the
   investment margin to a target growth rate (with a feasibility ladder so consumption
   stays positive), absorption of the resource constraints, and computed utility entries.
2. `uncertain_expansion_faisal_feb26.py` — 9-line entry patch: calling
   `uncertain_expansion` with `initial_guess=None` now derives the guess via
   `auto_guess`. Passing an explicit guess behaves exactly as before.
3. `autosolve.py` — NEW (driver; no engine modification). Solves from the derived
   guess and accepts a solution only if it passes TWO residual gates: the model's
   own deterministic equations AND the engine's complete steady-state system
   (Euler/costate rows included), with a single re-polish pass when the residual
   is close to tolerance. When a solve fails it grid-restarts over states whose
   scale the pre-solve cannot pin (10-minute budget), accepts model-derived
   closed-form steady-state hints (`state_seeds`), and when a parameter target
   is not directly solvable walks one parameter at a time from the model's
   default values, bisecting the step on failure.
3a. `models_sourced.py` — NEW. Six test economies: the book's AK and habit
   models plus four written from published papers (Kaltenbrunner–Lochstoer,
   RFS 2010; Ai–Croce–Li, RFS 2013, via the Borovička–Hansen companion Dynare
   implementation; Croce, 2008 IGIER WP; Tallarini, JME 2000). Every
   calibration number traces to a printed table; frequency conversions and
   per-paper steady-state anchors are documented in the notebook. Includes
   `with_loadings` (shock-correlation mixing with unit row norms, used by the
   robustness battery).
3b. Two structural facts about the engine's model class, established during
   testing and stated in the notebook's Model section: (i) the first-order-
   condition assembly assumes ONE static constraint of the form
   constant − sum of controls (state-free, every control included); production
   economies with capital in the resource constraint must therefore be declared
   in output shares, else the compiled system can admit roots that are not the
   model's steady state; (ii) an endogenous labor choice (a second, time
   constraint) lies outside the current class.
4. Dead code removed from the engine copy (call-graph verified, regression-tested):
   `first_order_expansion_approach_2`, `compute_adj_approach_2`,
   `second_order_expansion_approach_2` (220 lines) and the 4-line dispatch branch that
   called them — unreachable because `approach` is fixed to `'1'` at entry — plus
   unused imports (`sympy.lambdify`, `seaborn`, `scipy.optimize.fsolve`).
5. `derivatives.py` — one import retargeted (`split_variables` now imported from the
   engine file in this folder instead of `src/uncertain_expansion.py`) so the folder
   is self-contained.
6. The folder ships only the engine's import closure (8 files). Not included because
   the engine never imports them: `BY_example_sol.py`, `elasticity_test.py`, `plot.py`,
   `shockElasDecomposition.py`, `shockElasModules.py`, `stationaryDensityModules.py`,
   `utils_pde_shock_elasticity.py`, and `src/uncertain_expansion.py`.

Everything else is byte-identical to the branch copy.
