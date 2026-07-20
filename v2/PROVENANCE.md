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
   guess, checks the returned steady state independently against the model's own
   deterministic equations (with a single re-polish pass when the residual is close
   to tolerance), and when a parameter target is not directly solvable walks one
   parameter at a time from the model's default values, bisecting the step on failure.
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
