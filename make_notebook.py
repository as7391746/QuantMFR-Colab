#!/usr/bin/env python3
"""Generate colab.ipynb — Chapter 11, Figures 11.1-11.3.

Model - solve - plot. The model is stated in the chapter's notation and the
solve step calls the book's expansion code (uncertain_expansion) directly,
fetched from the RiskUncertaintyValue repository — no layer in between.
"""

import json

md = lambda s: {"cell_type": "markdown", "metadata": {},
                "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": s.splitlines(keepends=True)}

TITLE = r"""# Chapter 11: Figures 11.1–11.3

The AK production economy of Chapter 11, solved with the book's expansion
code (`uncertain_expansion`), used exactly as the book's *Uncertainty
Expansion — Computation Process* appendix uses it. Parameters are the
quarterly values of the chapter appendix.

![method](https://raw.githubusercontent.com/as7391746/QuantMFR-Colab/main/assets/method.png)

*The pipeline and where each step lives in this notebook. Risk aversion is
rescaled with the perturbation ($\gamma-1=(\gamma_o-1)/\mathsf q$), so the
recursive-utility change of measure $N^0$ already matters at first order: it
shifts the shock mean to $\mu^0$, which is what generates the shock-price
elasticities in Figures 11.1 and 11.3.*

**Runtime → Run all** (~5 minutes on a standard Colab runtime: a shallow
clone, then thirteen expansion solves of ~10–45 s each). To experiment,
edit the parameters or `COMPARISONS` and re-run."""

SETUP = """# ================================ SETUP =================================
# Fetch the book's expansion code (the RiskUncertaintyValue repository) and
# put its src/ on the path, exactly as the book's computation appendix does.
%pip -q install autograd
import os, sys, io, time, contextlib, warnings

if not os.path.isdir("RiskUncertaintyValue"):
    !git clone -q --depth 1 -b Planners_with_External https://github.com/lphansen/RiskUncertaintyValue
sys.path.insert(0, os.path.abspath("RiskUncertaintyValue/src"))

# Silence two harmless warnings the code emits on newer Python/numba:
# docstring escape sequences (SyntaxWarning) and a numba array-layout note.
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

from uncertain_expansion import (uncertain_expansion,
                                 generate_symbols_and_args, approximate_fun)
from elasticity import exposure_elasticity, price_elasticity
from lin_quad_util import next_period
import uncertain_expansion as _code
print("expansion code:", _code.__file__)"""

MODEL = r"""# ================================ MODEL =================================
# The Chapter 11 AK economy, written exactly as the chapter prints it:
#   - two exogenous states {eq}`equation2`: Z1 = long-run growth state,
#     Z2 = log stochastic-volatility state (exp(Z2) scales all shock
#     variances);
#   - capital evolution with adjustment costs (display after
#     {eq}`equation3`): installation function (1/zeta) log(1 + zeta I/K);
#   - resource constraint {eq}`equation3`: C/K + I/K = alpha;
#   - recursive preferences {eq}`value_recur5`/{eq}`value_risk6` (handled
#     inside the expansion code; only beta, rho, gamma appear here).
#
# Variable-name conventions of the expansion code (as in the book's
# computation appendix): a trailing _t marks a date-t variable and _tp1 the
# next period; the growth variable is log_gk_t; q_t is the noise scale;
# W1..W3 are the shocks (W1 = capital shock, W2 = growth shock).
control_variables     = ["D1_t", "D2_t"]     # C/K and I/K
state_variables       = ["Z1_t", "Z2_t"]     # growth state, log volatility
growth_variables      = ["log_gk_t"]
perturbation_variable = ["q_t"]
shock_variables       = ["W1_t", "W2_t", "W3_t"]
variables = (control_variables + state_variables + growth_variables
             + perturbation_variable + shock_variables)
variables_tp1 = [v + "p1" for v in variables]


def create_args():
    '''The quarterly parameter values of the chapter appendix.'''
    beta  = float(np.exp(-0.0025))   # discount factor
    rho   = 1.001                    # inverse IES        (set per comparison)
    gamma = 8.001                    # risk aversion      (set per comparison)
    alpha = 0.02305                  # productivity       (set per comparison)
    zeta  = 32.0                     # adjustment-cost curvature
    iota_k = 0.01                    # capital drift constant
    nu_k   = 0.01                    # capital loading on Z1
    nu_1   = 0.014                   # Z1 mean reversion (autocorrelation 0.986)
    nu_2   = 0.0485                  # Z2 mean reversion
    mu_2   = 6.3e-6                  # central tendency of exp(Z2)
    s = float(np.sqrt(3.0))
    sigma_k1, sigma_k2, sigma_k3 = 0.92 * s, 0.40 * s, 0.0   # capital loadings (chapter's sigma_k)
    sigma_11, sigma_12, sigma_13 = 0.0, 5.70 * s, 0.0        # Z1 loadings      (chapter's sigma_1)
    sigma_21, sigma_22, sigma_23 = 0.0, 0.0, 0.00031 * s     # Z2 loadings      (chapter's sigma_2)
    del s
    return locals()

parameter_names, args = generate_symbols_and_args(create_args)
globals().update(parameter_names)                       # parameter symbols
globals().update({v: sp.Symbol(v) for v in variables + variables_tp1})

# Exogenous states {eq}`equation2`. Shock loadings carry no noise scale;
# the Ito-correction terms carry an explicit q_t^2 - both exactly as
# printed in the chapter (the displayed system IS the small-noise family).
Z1_next = (Z1_t - nu_1 * Z1_t
           + sp.exp(Z2_t / 2) * (sigma_11 * W1_tp1 + sigma_12 * W2_tp1
                                 + sigma_13 * W3_tp1))
Z2_next = (Z2_t - nu_2 * (1 - mu_2 * sp.exp(-Z2_t))
           - q_t**2 / 2 * (sigma_21**2 + sigma_22**2 + sigma_23**2)
           * sp.exp(-Z2_t)
           + sp.exp(-Z2_t / 2) * (sigma_21 * W1_tp1 + sigma_22 * W2_tp1
                                  + sigma_23 * W3_tp1))

# Capital evolution (display after {eq}`equation3`): installation of new
# capital, the long-run-risk channel nu_k * Z1, depreciation, and the
# capital-quality shock with stochastic volatility.
capital_growth = ((1 / zeta) * sp.log(1 + zeta * D2_t)
                  + nu_k * Z1_t - iota_k
                  - q_t**2 / 2 * (sigma_k1**2 + sigma_k2**2 + sigma_k3**2)
                  * sp.exp(Z2_t)
                  + sp.exp(Z2_t / 2) * (sigma_k1 * W1_tp1 + sigma_k2 * W2_tp1
                                        + sigma_k3 * W3_tp1))

# log consumption relative to capital, and the resource constraint
# {eq}`equation3`: the only real decision is how to split output alpha*K
# between consumption (D1) and investment (D2).
output_constraint = sp.log(D1_t)
static_constraints = [alpha - D1_t - D2_t]
state_equations = [Z1_next, Z2_next]

# Productivity alpha is paired with rho so that all specifications share
# the same steady-state growth; the base case uses 0.02305.
ALPHA = {0.67: 0.0205, 1.001: 0.023, 1.5: 0.027}
ALPHA_BASE = 0.02305

# The 13 comparisons behind the three figures:
#   base                        -> Fig 11.3 (price elasticities at three
#                                  volatility quantiles, both shocks)
#   invk_r*   (gamma = 8)       -> Fig 11.2 (investment-capital exposure)
#   g*_r*     (3x3 gamma x rho) -> Fig 11.1 (consumption, 6-panel grid)
# gamma/rho carry small offsets from the degenerate values (8.001 for 8,
# 1.001 for 1) to stay clear of the limiting formulas.
COMPARISONS = (
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
# uncertain_expansion computes the second-order small-noise expansion: the
# deterministic steady state (order 0), the linear responses (order 1,
# where the recursive-utility change of measure N^0 first appears), and
# the quadratic corrections (order 2). From the solution we then compute
# Borovicka-Hansen shock elasticities:
#   - exposure elasticity: how much a bit of extra date-1 exposure to one
#     shock moves the expected quantity at each horizon (a nonlinear
#     impulse response);
#   - price elasticity: the expected-return compensation per unit of that
#     exposure (a term structure of risk prices).
# Horizon: 160 quarters = 40 years. Shocks (0-indexed): W2 = growth-rate
# shock, W1 = direct capital shock.
T, GROWTH_SHOCK, CAPITAL_SHOCK = 160, 1, 0
BETA, QUANTILES = float(np.exp(-0.0025)), (0.1, 0.5, 0.9)


def args_for(gamma_v, rho_v, alpha_v):
    '''The parameter list with this comparison's preferences and alpha.'''
    a = dict(zip(parameter_names.keys(), args))
    a.update({"gamma": gamma_v, "rho": rho_v, "alpha": alpha_v})
    return list(a.values())


def initial_guess(alpha_v):
    '''Starting values for the steady-state search, in the order the code
    solves them:
      [Vhat-Khat, log(C/K), MS, D1, D2, MX1, MX2, MG, growth, Z1, Z2]
    All rho share the same steady-state growth by construction, so
    investment starts near 0.019 and consumption absorbs the rest of
    output; the volatility state starts at its central tendency log(mu_2).'''
    return np.array([-2.0, -4.0, 0.0,
                     max(alpha_v - 0.019, 1e-3), 0.019,
                     0.0, 0.0, 1.0, 0.005, 0.0, float(np.log(6.3e-6))])


def log_sdf(sol, rho_v):
    '''log SDF increment, as in the book's computation appendix:
    log(beta) - rho * consumption growth + (rho-1)(vmr1 + vmr2/2) + log N-tilde.'''
    vmr = sol["vmr1_tp1"] + 0.5 * sol["vmr2_tp1"]
    return (np.log(BETA) - rho_v * sol["gc_tp1"] + (rho_v - 1) * vmr
            + sol["log_N_tilde"])


def invk_increment(sol):
    '''log(I/K) growth: second-order approximation of log(D2), pushed one
    period ahead — any function of the model's variables can be priced
    this way.'''
    level, _, _, _ = approximate_fun(
        fun=sp.log(sp.Symbol("D2_t")), ss=sol["ss"],
        ss_variables=sol["ss_variables"],
        ss_variables_tp1=sol["ss_variables_tp1"],
        parameter_names=sol["parameter_names"], args=sol["args"],
        var_shape=sol["var_shape"], JX1_t=sol["JX1_t"], JX2_t=sol["JX2_t"],
        X1_tp1=sol["X1_tp1"], X2_tp1=sol["X2_tp1"],
        recursive_ss=sol["recursive_ss"], second_order=True)
    return next_period(level, sol["X1_tp1"], sol["X2_tp1"]) - level


results = {}
t0 = time.time()
for cmp in COMPARISONS:
    t1 = time.time()
    # one call to the book's expansion code per comparison (its iteration
    # log is captured; remove redirect_stdout to see it in full)
    with contextlib.redirect_stdout(io.StringIO()):
        sol = uncertain_expansion(
            control_variables, state_variables, shock_variables,
            variables, variables_tp1, output_constraint, capital_growth,
            state_equations, static_constraints,
            initial_guess(cmp["alpha"]), parameter_names,
            args_for(cmp["gamma"], cmp["rho"], cmp["alpha"]),
            approach="1", iter_tol=1e-10, max_iter=500)
    if cmp["figure"] == "price_quantiles":
        # Fig 11.3: the elasticity depends on the volatility state; report
        # its 0.1 / 0.5 / 0.9 quantiles under the stationary distribution
        sdf = log_sdf(sol, cmp["rho"])
        for quantile in QUANTILES:
            for name, shock in (("growth", GROWTH_SHOCK),
                                ("capital", CAPITAL_SHOCK)):
                results[f"{cmp['id']}/price_{name}_q{quantile}"] = \
                    price_elasticity(sol["gc_tp1"], sdf, sol["X1_tp1"],
                                     sol["X2_tp1"], T, shock=shock,
                                     percentile=quantile).flatten()
    elif cmp["figure"] == "invk_expo":
        # Fig 11.2: exposure elasticity of the investment-capital ratio
        results[f"{cmp['id']}/expo_invk_growth_q0.5"] = \
            exposure_elasticity(invk_increment(sol), sol["X1_tp1"],
                                sol["X2_tp1"], T, shock=GROWTH_SHOCK,
                                percentile=0.5).flatten()
    else:
        # Fig 11.1: consumption exposure and price elasticities (the
        # priced process is consumption growth), median state
        results[f"{cmp['id']}/expo_growth_q0.5"] = \
            exposure_elasticity(sol["gc_tp1"], sol["X1_tp1"],
                                sol["X2_tp1"], T, shock=GROWTH_SHOCK,
                                percentile=0.5).flatten()
        results[f"{cmp['id']}/price_growth_q0.5"] = \
            price_elasticity(sol["gc_tp1"], log_sdf(sol, cmp["rho"]),
                             sol["X1_tp1"], sol["X2_tp1"], T,
                             shock=GROWTH_SHOCK, percentile=0.5).flatten()
    print(f"{cmp['id']}: solved in {time.time() - t1:.1f}s", flush=True)
print(f"{len(COMPARISONS)} comparisons in {time.time() - t0:.0f}s")"""

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

**Checks.** The solver is the book's expansion code, used as-is and called
exactly as the book's computation appendix calls it, and the resulting
figures line up with the chapter's. More detail:
[`README.md`](https://github.com/as7391746/QuantMFR-Colab)."""


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
