#!/usr/bin/env python3
"""Generate colab.ipynb — Chapter 11, Figures 11.1-11.3.

Layout: two sections. Codes (setup, model, solve, one cell per figure) and
Analysis. Model - solve - plot: the model is declared in the chapter's
notation, the solve step imports the expansion engine (uncertain_expansion)
directly from this repository's src/, and the figure cells plot.
"""

import json

md = lambda s: {"cell_type": "markdown", "metadata": {},
                "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": s.splitlines(keepends=True)}

TITLE = r"""# Chapter 11: Figures 11.1–11.3

The AK production economy of Chapter 11, declared in the chapter's notation
and solved with the expansion engine (`uncertain_expansion`, imported
directly from this repository's `src/`). Parameters are the quarterly
values of the chapter appendix.

![method](https://raw.githubusercontent.com/as7391746/QuantMFR-Colab/main/assets/method.png)

*The pipeline and where each step lives in this notebook. Risk aversion is
rescaled with the perturbation ($\gamma-1=(\gamma_o-1)/\mathsf q$), so the
recursive-utility change of measure $N^0$ already matters at first order: it
shifts the shock mean to $\mu^0$, which is what generates the shock-price
elasticities in Figures 11.1 and 11.3.*

**Runtime → Run all** (~5 minutes on a standard Colab runtime: a shallow
clone, then thirteen expansion solves of ~10–45 s each). To experiment,
edit `PARAMS` or `SCENARIOS` and re-run."""

SETUP = """# ================================ SETUP =================================
# Fetch this repository and put two folders on the Python path:
#   src/       - the expansion engine (uncertain_expansion and its modules),
#                an unmodified snapshot of RiskUncertaintyValue
#                (branch Planners_with_External, commit 09ca5df)
#   expansion/ - a thin declaration layer: it lets the model be written in
#                the chapter's notation and builds the engine's inputs
#                (variable lists, ordered arguments, steady-state starting
#                vector) automatically
%pip -q install autograd
import os, sys, io, time, contextlib, warnings

if not os.path.isdir("QuantMFR-Colab"):
    !git clone -q --depth 1 https://github.com/as7391746/QuantMFR-Colab
sys.path.insert(0, os.path.abspath("QuantMFR-Colab/src"))
sys.path.insert(0, os.path.abspath("QuantMFR-Colab/expansion"))

# The engine snapshot is used as-is, so silence two harmless warnings it
# emits on newer Python/numba: docstring escape sequences (SyntaxWarning)
# and a numba array-layout performance note.
warnings.filterwarnings("ignore", category=SyntaxWarning)
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except Exception:
    pass

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from uncertain_expansion import uncertain_expansion   # the engine, imported directly
from expansion import Model                            # the declaration layer
import uncertain_expansion as _engine
print("engine:", _engine.__file__)"""

MODEL = r"""# ================================ MODEL =================================
# The Chapter 11 AK economy, written exactly as the chapter prints it:
#   - two exogenous states {eq}`equation2`: Z1 = long-run growth state,
#     Z2 = log stochastic-volatility state (exp(Z2) scales all shock
#     variances);
#   - capital evolution with adjustment costs (display after
#     {eq}`equation3`): installation function (1/zeta) log(1 + zeta I/K);
#   - resource constraint {eq}`equation3`: C/K + I/K = alpha;
#   - recursive preferences {eq}`value_recur5`/{eq}`value_risk6` (handled
#     inside the engine; only beta, rho, gamma appear here).
#
# All parameter values are quarterly, as in the chapter appendix.
SQRT3 = np.sqrt(3.0)
PARAMS = {
    "beta": np.exp(-0.0025),          # quarterly discount factor
    "zeta": 32.0,                     # adjustment-cost curvature
    "iota_k": 0.01,                   # capital drift constant
    "nu_k": 0.01,                     # capital loading on Z1
    "nu_1": 0.014,                    # Z1 mean reversion (autocorrelation 0.986)
    "nu_2": 0.0485,                   # Z2 mean reversion
    "mu_2": 6.3e-6,                   # central tendency of exp(Z2)
    "sigma_k": SQRT3 * np.array([0.92, 0.40, 0.0]),     # capital shock loadings
    "sigma_1": SQRT3 * np.array([0.0, 5.70, 0.0]),      # growth-state loadings
    "sigma_2": SQRT3 * np.array([0.0, 0.0, 0.00031]),   # volatility-state loadings
}
# Productivity alpha is paired with rho so that all specifications share
# the same steady-state growth. The base case uses 0.02305.
ALPHA = {0.67: 0.0205, 1.001: 0.023, 1.5: 0.027}
ALPHA_BASE = 0.02305

# Declare the model. m["X"] gives the symbol for a variable; m.p.name gives
# a parameter symbol; m.W are next-period shocks; m.q is the noise scale;
# m.dot("sigma") = sigma . W and m.norm2("sigma") = |sigma|^2.
m = Model(states=["Z1", "Z2"], controls=["D1", "D2"], shocks=3)
Z1, Z2, D1, D2 = m["Z1"], m["Z2"], m["D1"], m["D2"]
p, q = m.p, m.q

# Exogenous states {eq}`equation2`. Shock loadings carry no noise scale;
# the Ito-correction terms carry an explicit q^2 - both exactly as printed
# in the chapter (the displayed system IS the small-noise family).
m.state["Z1"] = Z1 - p.nu_1 * Z1 + sp.exp(Z2 / 2) * m.dot("sigma_1")
m.state["Z2"] = (Z2 - p.nu_2 * (1 - p.mu_2 * sp.exp(-Z2))
                 - q**2 / 2 * m.norm2("sigma_2") * sp.exp(-Z2)
                 + sp.exp(-Z2 / 2) * m.dot("sigma_2"))

# Capital evolution (display after {eq}`equation3`): installation of new
# capital, the long-run-risk channel nu_k * Z1, depreciation, and the
# capital-quality shock with stochastic volatility.
m.growth = ((1 / p.zeta) * sp.log(1 + p.zeta * D2) + p.nu_k * Z1 - p.iota_k
            - q**2 / 2 * m.norm2("sigma_k") * sp.exp(Z2)
            + sp.exp(Z2 / 2) * m.dot("sigma_k"))

# log consumption relative to capital, and the resource constraint
# {eq}`equation3`: the only real decision is how to split output alpha*K
# between consumption (D1) and investment (D2).
m.consumption = sp.log(D1)
m.constraint = p.alpha - D1 - D2

# The 13 solves behind the three figures:
#   base                        -> Fig 11.3 (price elasticities at three
#                                  volatility quantiles, both shocks)
#   invk_r*   (gamma = 8)       -> Fig 11.2 (investment-capital exposure)
#   g*_r*     (3x3 gamma x rho) -> Fig 11.1 (consumption, 6-panel grid)
# gamma/rho carry small offsets from the degenerate values (8.001 for 8,
# 1.001 for 1) to stay clear of the limiting formulas.
SCENARIOS = (
    [{"id": "base", "gamma": 8.001, "rho": 1.001,
      "alpha": ALPHA_BASE, "figure": "price_quantiles"}]
    + [{"id": f"invk_r{r}", "gamma": 8.0, "rho": r,
        "alpha": ALPHA[r], "figure": "invk_expo"}
       for r in [0.67, 1.001, 1.5]]
    + [{"id": f"g{g}_r{r}", "gamma": g, "rho": r,
        "alpha": ALPHA[r], "figure": "six_panel"}
       for g in [1.001, 4.001, 8.001] for r in [0.67, 1.001, 1.5]]
)"""

SOLVE = r"""# ================================ SOLVE =================================
# m.solve(...) hands the model to uncertain_expansion, which computes the
# second-order small-noise expansion: the deterministic steady state
# (order 0), the linear responses (order 1, where the recursive-utility
# change of measure N^0 first appears), and the quadratic corrections
# (order 2). From the solution we then request Borovicka-Hansen shock
# elasticities:
#   - exposure elasticity: how much a bit of extra date-1 exposure to one
#     shock moves the expected quantity at each horizon (a nonlinear
#     impulse response);
#   - price elasticity: the expected-return compensation per unit of that
#     exposure (a term structure of risk prices).
# Horizon: 160 quarters = 40 years. Shocks (0-indexed): W2 = growth-rate
# shock, W1 = direct capital shock.
T, GROWTH_SHOCK, CAPITAL_SHOCK = 160, 1, 0

def solve_scenario(sc):
    # per-scenario parameters: preferences and the rho-paired alpha
    alpha_q = sc["alpha"]
    params = dict(PARAMS)
    params.update({"gamma": float(sc["gamma"]), "rho": float(sc["rho"]),
                   "alpha": alpha_q})
    # Rough starting values for the steady-state search, stated in the
    # model's own terms: all rho share the same steady-state growth by
    # construction, so investment starts near 0.019 and consumption
    # absorbs the rest of output; volatility starts at its central
    # tendency log(mu_2).
    start = {"D2": 0.019, "D1": max(alpha_q - 0.019, 1e-3),
             "Z2": float(np.log(6.3e-6)), "growth": 0.005}
    return m.solve(params, start=start)

results = {}
t0 = time.time()
for sc in SCENARIOS:
    t1 = time.time()
    # the engine prints its iteration log; keep the notebook output to one
    # line per scenario (remove redirect_stdout to see the full log)
    with contextlib.redirect_stdout(io.StringIO()):
        sol = solve_scenario(sc)
    if sc["figure"] == "price_quantiles":
        # Fig 11.3: the elasticity depends on the volatility state; report
        # its 0.1 / 0.5 / 0.9 quantiles under the stationary distribution
        for quantile in (0.1, 0.5, 0.9):
            for name, shock in (("growth", GROWTH_SHOCK),
                                ("capital", CAPITAL_SHOCK)):
                results[f"{sc['id']}/price_{name}_q{quantile}"] = \
                    sol.price_elasticity(shock, T, quantile)
    elif sc["figure"] == "invk_expo":
        # Fig 11.2: elasticity of the investment-capital ratio; the
        # increment of ANY expression of the model's variables can be
        # priced this way
        invk = sol.increment(sp.log(m["D2"]))
        results[f"{sc['id']}/expo_invk_growth_q0.5"] = \
            sol.exposure_elasticity(GROWTH_SHOCK, T, process=invk)
    else:
        # Fig 11.1: consumption exposure and price elasticities (the
        # default priced process is consumption growth), median state
        results[f"{sc['id']}/expo_growth_q0.5"] = \
            sol.exposure_elasticity(GROWTH_SHOCK, T)
        results[f"{sc['id']}/price_growth_q0.5"] = \
            sol.price_elasticity(GROWTH_SHOCK, T)
    print(f"{sc['id']}: solved in {time.time() - t1:.1f}s", flush=True)
print(f"{len(SCENARIOS)} scenarios in {time.time() - t0:.0f}s")"""

FIG1 = r"""# ============================ FIGURE 11.1 ===============================
# Consumption exposure (left) and price (right) elasticities for the
# growth-rate shock; rows vary gamma, curves vary rho (alpha paired to rho).
# What to look for: rho (inverse IES) moves the LEFT column (quantities),
# gamma (risk aversion) moves the RIGHT column (prices) - the two
# preference parameters act on different objects.
sns.set_style("darkgrid")
yrs = np.arange(1, 161) / 4
gammas, rhos = [1.001, 4.001, 8.001], [0.67, 1.001, 1.5]
fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
for i, g in enumerate(gammas):
    for r in rhos:
        axes[i, 0].plot(yrs, results[f"g{g}_r{r}/expo_growth_q0.5"], label=fr"$\rho={r}$")
        axes[i, 1].plot(yrs, results[f"g{g}_r{r}/price_growth_q0.5"])
    axes[i, 0].set_ylabel(fr"$\gamma={g:.0f}$")
axes[0, 0].set_title("exposure elasticity"); axes[0, 1].set_title("price elasticity")
axes[0, 0].legend(frameon=False)
for ax in axes[-1]: ax.set_xlabel("years")
plt.tight_layout(); plt.show()"""

FIG2 = r"""# ============================ FIGURE 11.2 ===============================
# Investment-capital exposure elasticity, growth-rate shock, gamma = 8.
# What to look for: the sign of the initial response flips as rho crosses
# 1. With IES > 1 (rho = 2/3) the planner invests more on good growth
# news (substitution effect wins); with IES < 1 (rho = 3/2) consumption
# absorbs the news (income effect wins); at rho = 1 they cancel exactly.
plt.figure(figsize=(6, 4))
for r in rhos:
    plt.plot(yrs, results[f"invk_r{r}/expo_invk_growth_q0.5"], label=fr"$\rho={r}$")
plt.axhline(0, color="k", lw=0.6)
plt.xlabel("years"); plt.legend(frameon=False); plt.tight_layout(); plt.show()"""

FIG3 = r"""# ============================ FIGURE 11.3 ===============================
# Price elasticities at the 0.1 / 0.5 / 0.9 stochastic-volatility quantiles,
# growth and capital shocks; gamma = 8, rho = 1.
# What to look for: the growth-rate shock carries the large, upward-sloping
# prices (the forward-looking continuation value amplifies persistent
# risks); the spread across quantiles is the stochastic-volatility effect.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for name, ax in [("growth", axes[0]), ("capital", axes[1])]:
    for quantile in [0.1, 0.5, 0.9]:
        ax.plot(yrs, results[f"base/price_{name}_q{quantile}"], label=f"q = {quantile}")
    ax.set_title(f"{name} shock"); ax.set_xlabel("years")
axes[0].legend(frameon=False)
plt.tight_layout(); plt.show()"""

ANALYSIS = r"""# Analysis

**What the figures show.** $\rho$ (the inverse IES) moves the *quantity*
elasticities and $\gamma$ moves the *price* elasticities. In Figure 11.2 the
sign of the initial investment response flips as $\rho$ crosses one: with
IES $>1$ ($\rho=2/3$) the planner invests more on good growth news, while
with IES $<1$ ($\rho=3/2$) consumption absorbs it. In Figure 11.3
stochastic volatility does little to the average price but generates the
state dependence visible in the quantile spread, and the forward-looking
continuation value makes the growth-rate shock, rather than the direct
capital shock, carry the large prices.

**Calibration.** When $\rho$ varies, the productivity $\alpha$ is re-paired
(0.0205 / 0.023 / 0.027) so all specifications share the same steady-state
growth. All parameter values are quarterly, as in the chapter appendix.

**Checks.** The solver is the expansion engine itself, imported unmodified
from `src/` (RiskUncertaintyValue, branch `Planners_with_External`, commit
`09ca5df`), and the resulting figures line up with the chapter's.
More detail: [`README.md`](https://github.com/as7391746/QuantMFR-Colab)."""


def main():
    cells = [md(TITLE), md("# Codes"),
             code(SETUP), code(MODEL), code(SOLVE),
             code(FIG1), code(FIG2), code(FIG3),
             md(ANALYSIS)]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"name": "ak-elasticities-chapter.ipynb"},
                       "kernelspec": {"name": "python3",
                                      "display_name": "Python 3"}},
          "cells": cells}
    json.dump(nb, open("colab.ipynb", "w"), indent=1)
    print(f"colab.ipynb: {len(cells)} cells (6 code)")


if __name__ == "__main__":
    main()
