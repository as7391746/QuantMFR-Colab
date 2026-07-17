#!/usr/bin/env python3
"""Generate colab_habit.ipynb — Chapter 11, Figures 11.4-11.9.

Model - solve - plot, same delivery standard as colab.ipynb: the
habit-preference model is stated as the book's computation appendix states
it, solved by calling uncertain_expansion directly (fetched from the
RiskUncertaintyValue repository), and the six habit figures are plotted.
"""

import json

md = lambda s: {"cell_type": "markdown", "metadata": {},
                "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": s.splitlines(keepends=True)}

TITLE = r"""# Chapter 11: Figures 11.4–11.9 (habit preferences)

Model — solve — plot. The **model** adds habit preferences to the Chapter
11 production economy: a habit stock $X=\log(H/K)$, and consumption
services that aggregate the consumption input and the habit stock with
weight $\lambda$ and curvature $\tau$ — the same model the book's
*Uncertainty Expansion — Computation Process* appendix uses as its worked
example, with its quarterly parameters. The **solve** step is the book's
expansion code (`uncertain_expansion`), called exactly as that appendix
calls it — including its warm-start rule, where each solve starts from the
previous solution's steady state. The **plots** are the six habit figures:
investment-capital exposure elasticities and consumption price
elasticities, internal versus external habit, $\gamma = 4$ and $8$.

**Runtime → Run all** (~45 minutes on a standard Colab runtime: 40
expansion solves, warm-started along a chain). To reproduce a single
figure, keep only its (habit, $\gamma$) block in `COMPARISONS` and
re-run."""

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
# The habit-preference production economy, exactly as the book's
# computation appendix states it:
#   - three exogenous/endogenous states: Z = long-run growth state, Y = log
#     stochastic-volatility state, X = log(H/K), the habit stock relative
#     to capital (it depreciates at nu_h and is replenished by the
#     consumption input I_h/K);
#   - controls: imh = I_h/K (consumption input) and imk = I_k/K
#     (investment); resource constraint imh + imk = a;
#   - consumption services aggregate imh and the habit stock with weight
#     llambda and curvature tau:
#         kappa = 1/(1-tau) * log((1-llambda) imh^(1-tau)
#                                  + llambda exp((1-tau) X));
#   - "internal" habit: the planner internalizes how imh feeds the habit
#     stock; "external": the habit stock is taken as given (the code's
#     ExternalHabit flag).
#
# Time step: epsilon = 0.25 years (a quarter); rates are annual and enter
# the equations multiplied by epsilon, as in the appendix.
control_variables     = ["imh_t", "imk_t"]
state_variables       = ["Z_t", "Y_t", "X_t"]
growth_variables      = ["log_gk_t"]
perturbation_variable = ["q_t"]
shock_variables       = ["W1_t", "W2_t", "W3_t"]
variables = (control_variables + state_variables + growth_variables
             + perturbation_variable + shock_variables)
variables_tp1 = [v + "p1" for v in variables]


def create_args():
    '''The parameter values of the book's computation appendix.'''
    sigma_k1 = 0.92 * np.sqrt(12.0)      # capital shock loadings
    sigma_k2 = 0.40 * np.sqrt(12.0)
    sigma_k3 = 0.0
    sigma_z1 = 0.0                       # growth-state loadings
    sigma_z2 = 5.7 * np.sqrt(12.0)
    sigma_z3 = 0.0
    sigma_y1 = 0.0                       # volatility-state loadings
    sigma_y2 = 0.0
    sigma_y3 = 0.00031 * np.sqrt(12.0)
    epsilon = 0.25                       # years per model step
    delta = 0.01                         # subjective discount rate
    beta = float(np.exp(-delta * epsilon))
    gamma = 1.001                        # risk aversion   (set per comparison)
    rho = 1.001                          # inverse IES
    a = 0.0922                           # productivity
    zeta = 8.0                           # adjustment-cost curvature
    phi_1 = 1.0 / zeta
    phi_2 = zeta
    alpha_k = 0.04                       # capital drift
    beta_k = 0.04                        # capital loading on Z
    beta_z = 0.056                       # Z mean reversion
    beta_2 = 0.194                       # Y mean reversion
    mu_2 = 6.3e-6                        # central tendency of exp(Y)
    nu_h = 0.1 * epsilon                 # habit depreciation (quarterly 0.025)
    tau = 1.01                           # services curvature (set per comparison)
    llambda = -0.0                       # habit weight       (set per comparison)
    return locals()

parameter_names, args = generate_symbols_and_args(create_args)
globals().update(parameter_names)                       # parameter symbols
globals().update({v: sp.Symbol(v) for v in variables + variables_tp1})

# Long-run growth state
technology_growth = (Z_t - epsilon * beta_z * Z_t
                     + sp.sqrt(epsilon) * sp.exp(0.5 * Y_t)
                     * (sigma_z1 * W1_tp1 + sigma_z2 * W2_tp1
                        + sigma_z3 * W3_tp1))

# Log stochastic-volatility state (with its q^2 Ito-correction term)
volatility_growth = (Y_t - epsilon * beta_2 * (1 - mu_2 * sp.exp(-Y_t))
                     - q_t**2 * 0.5 * (sigma_y1**2 + sigma_y2**2
                                       + sigma_y3**2)
                     * sp.exp(-Y_t) * epsilon
                     + sp.exp(-0.5 * Y_t)
                     * (sigma_y1 * W1_tp1 + sigma_y2 * W2_tp1
                        + sigma_y3 * W3_tp1) * sp.sqrt(epsilon))

# Capital growth: installation, the long-run-risk channel beta_k * Z,
# depreciation, and the capital-quality shock with stochastic volatility
capital_growth = (epsilon * (phi_1 * sp.log(1. + phi_2 * imk_t)
                             - alpha_k + beta_k * Z_t
                             - q_t**2 * 0.5 * (sigma_k1**2 + sigma_k2**2
                                               + sigma_k3**2)
                             * sp.exp(Y_t))
                  + sp.sqrt(epsilon) * sp.exp(0.5 * Y_t)
                  * (sigma_k1 * W1_tp1 + sigma_k2 * W2_tp1
                     + sigma_k3 * W3_tp1))

# Habit stock relative to capital: accumulate, then subtract capital growth
habit_growth = (sp.log(sp.exp(-nu_h + X_t)
                       + (1 - sp.exp(-nu_h)) * imh_t)
                - capital_growth)

# Consumption services and the resource constraint
kappa = (1 / (1 - tau)
         * sp.log((1 - llambda) * imh_t ** (1 - tau)
                  + llambda * sp.exp((1 - tau) * X_t)))
output_constraint = kappa
static_constraints = [a - imh_t - imk_t]
state_equations = [technology_growth, volatility_growth, habit_growth]

# The comparisons behind the six figures. Each figure overlays
# lambda in {0.67, 0, -2} across three tau panels {1.001, 0.6, 0.01},
# for internal/external habit and gamma in {4, 8}. Solves are chained:
# every comparison warm-starts from its parent's steady state (the
# appendix's own loop pattern); tau descends within each lambda, and
# lambda = 0.5 is a stepping stone toward 0.67 (solved, not plotted).
GAMMAS = [4.0, 8.0]

def _block(habit, gamma_v):
    def cid(lam, tau_v):
        return f"{habit}_g{gamma_v:g}_lam{lam:g}_tau{tau_v:g}"
    def c(lam, tau_v, warm_from, stone=False):
        return {"id": cid(lam, tau_v), "habit": habit, "gamma": gamma_v,
                "llambda": lam, "tau": tau_v, "warm_from": warm_from,
                "stone": stone}
    root = cid(0.0, 1.001)
    return ([c(0.0, 1.001, None),
             c(0.0, 0.6, root), c(0.0, 0.01, cid(0.0, 0.6)),
             c(-2.0, 1.001, root), c(-2.0, 0.6, cid(-2.0, 1.001)),
             c(-2.0, 0.01, cid(-2.0, 0.6)),
             c(0.5, 1.001, root, stone=True),
             c(0.67, 1.001, cid(0.5, 1.001)),
             c(0.67, 0.6, cid(0.67, 1.001)),
             c(0.67, 0.01, cid(0.67, 0.6))])

COMPARISONS = [c for habit in ["ext", "int"] for g in GAMMAS
               for c in _block(habit, g)]
print(len(COMPARISONS), "comparisons")"""

SOLVE = r"""# ================================ SOLVE =================================
# One call to uncertain_expansion per comparison. Two series are computed
# from each solution:
#   - the exposure elasticity of the investment-capital ratio log(I_k/K)
#     to the growth shock (a nonlinear impulse response);
#   - the consumption price elasticity for the growth shock (the
#     expected-return compensation per unit of exposure).
# Horizon 160 quarters, median volatility state. W2 is the growth shock.
T, GROWTH_SHOCK, PCT = 160, 1, 0.5
BETA, RHO = float(np.exp(-0.01 * 0.25)), 1.001

# Starting values for the steady-state search, as in the appendix; chained
# comparisons instead start from [WARM_HEAD] + the parent's steady state.
BASELINE_GUESS = np.array([-3.0599204508041717, -3.1433984583550463,
                           (1 - BETA) / 0.0257233, 0.0257233,
                           0.0922 - 0.0257233, 0.02039991, 0.0, 0.0, 1.0,
                           0.01330655, 0.0, -11.97496092, -3.51858229])
WARM_HEAD = -10.40379947


def args_for(gamma_v, lam_v, tau_v):
    '''The parameter list with this comparison's gamma, lambda, tau.'''
    a_ = dict(zip(parameter_names.keys(), args))
    a_.update({"gamma": gamma_v, "llambda": lam_v, "tau": tau_v})
    return list(a_.values())


def invk_exposure(sol):
    '''Exposure elasticity of log(I_k/K) growth: second-order approximation
    of log(imk), pushed one period ahead.'''
    level, _, _, _ = approximate_fun(
        fun=sp.log(sp.Symbol("imk_t")), ss=sol["ss"],
        ss_variables=sol["ss_variables"],
        ss_variables_tp1=sol["ss_variables_tp1"],
        parameter_names=sol["parameter_names"], args=sol["args"],
        var_shape=sol["var_shape"], JX1_t=sol["JX1_t"], JX2_t=sol["JX2_t"],
        X1_tp1=sol["X1_tp1"], X2_tp1=sol["X2_tp1"],
        recursive_ss=sol["recursive_ss"], second_order=True)
    growth = next_period(level, sol["X1_tp1"], sol["X2_tp1"]) - level
    return exposure_elasticity(growth, sol["X1_tp1"], sol["X2_tp1"], T,
                               shock=GROWTH_SHOCK, percentile=PCT).flatten()


def price(sol):
    '''Consumption price elasticity, with the log SDF assembled as in the
    appendix: log(beta) - rho * consumption growth + (rho-1)(vmr1 + vmr2/2)
    + log N-tilde.'''
    vmr = sol["vmr1_tp1"] + 0.5 * sol["vmr2_tp1"]
    log_SDF = (np.log(BETA) - RHO * sol["gc_tp1"] + (RHO - 1) * vmr
               + sol["log_N_tilde"])
    return price_elasticity(sol["gc_tp1"], log_SDF, sol["X1_tp1"],
                            sol["X2_tp1"], T, shock=GROWTH_SHOCK,
                            percentile=PCT).flatten()


results, ss_by_id = {}, {}
t0 = time.time()
for cmp in COMPARISONS:
    t1 = time.time()
    guess = (BASELINE_GUESS if cmp["warm_from"] is None else
             np.concatenate([[WARM_HEAD], ss_by_id[cmp["warm_from"]]]))
    with contextlib.redirect_stdout(io.StringIO()):   # iteration log
        sol = uncertain_expansion(
            control_variables, state_variables, shock_variables,
            variables, variables_tp1, output_constraint, capital_growth,
            state_equations, static_constraints, guess, parameter_names,
            args_for(cmp["gamma"], cmp["llambda"], cmp["tau"]),
            approach="1", iter_tol=1e-5, max_iter=500,
            ExternalHabit=(cmp["habit"] == "ext"))
    ss_by_id[cmp["id"]] = np.asarray(sol["ss"], dtype=float)
    if not cmp["stone"]:                  # stepping stones are not plotted
        results[f"{cmp['id']}/expo"] = invk_exposure(sol)
        results[f"{cmp['id']}/price"] = price(sol)
    print(f"{cmp['id']}: solved in {time.time() - t1:.1f}s", flush=True)
print(f"{len(COMPARISONS)} comparisons in {(time.time() - t0) / 60:.1f} min")"""

PLOTFUN = r"""# ============================= FIGURES ==================================
# Shared skeleton of the six figures: three tau panels, three lambda curves
# per panel (as in the book: lambda = 0.67 green, 0 red, -2 blue).
sns.set_style("darkgrid")
TAUS_PLOT = [1.001, 0.60, 0.01]
LAMBDAS_PLOT = [0.67, 0.0, -2.0]
COLORS = ["green", "red", "blue"]
HABIT_NAME = {"int": "Internal", "ext": "External"}

def habit_figure(habit, gamma_v, var, ylabel, ylim, kind):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for i, tau_v in enumerate(TAUS_PLOT):
        ax = axes[i]
        for color, lam in zip(COLORS, LAMBDAS_PLOT):
            key = f"{habit}_g{gamma_v:g}_lam{lam:g}_tau{tau_v:g}/{var}"
            ax.plot(np.arange(T), results[key], color=color, lw=2.5,
                    label=rf"$\lambda={lam:.2f}$")
        tau_lbl = "1" if abs(tau_v - 1.0) < 0.01 else f"{tau_v:g}"
        ax.set_title(rf"$\tau={tau_lbl}$")
        ax.set_xlabel("quarters"); ax.set_xlim(0, T)
        if ylim: ax.set_ylim(ylim)
    axes[0].set_ylabel(ylabel)
    axes[2].legend(frameon=False, loc="upper right")
    fig.suptitle(rf"{HABIT_NAME[habit]} habit — {kind}, "
                 rf"$\gamma={gamma_v:g}$, $\nu_h=0.1$ (annually)", y=1.04)
    plt.tight_layout(); plt.show()"""

FIG45 = r"""# ====================== FIGURES 11.4 & 11.5 =============================
# Investment-capital exposure elasticity to the growth shock, gamma = 8:
# internal habit (11.4), then external (11.5).
# What to look for: with a strong habit motive (small tau, positive
# lambda), the planner smooths consumption services and lets investment
# absorb the growth shock.
for habit in ["int", "ext"]:
    habit_figure(habit, 8.0, "expo",
                 "Exposure Elasticity", (-1e-3, 5e-3),
                 "Investment–Capital Ratio Exposure Elasticity")"""

FIG67 = r"""# ====================== FIGURES 11.6 & 11.7 =============================
# Consumption price elasticity for the growth shock, gamma = 8: internal
# habit (11.6), then external (11.7).
# What to look for: how the habit weight lambda and curvature tau reshape
# the term structure of risk prices, and how the internal/external
# treatments differ.
for habit in ["int", "ext"]:
    habit_figure(habit, 8.0, "price",
                 "Price Elasticity", (0.0, 0.3),
                 "Consumption Price Elasticity")"""

FIG89 = r"""# ====================== FIGURES 11.8 & 11.9 =============================
# The same price elasticities at gamma = 4: internal habit (11.8), then
# external (11.9). Compare with the previous cell: lowering risk aversion
# scales the whole term structure down.
for habit in ["int", "ext"]:
    habit_figure(habit, 4.0, "price",
                 "Price Elasticity", (0.0, 0.3),
                 "Consumption Price Elasticity")"""

ANALYSIS = r"""# Analysis

**What the figures show.** Habit preferences separate consumption from
consumption *services*: with a positive habit weight ($\lambda = 0.67$) and
strong curvature (small $\tau$), the planner protects the service flow, so
growth-shock news is absorbed by investment (the exposure elasticities of
Figures 11.4–11.5) and the pricing of the shock changes with it (Figures
11.6–11.9). Comparing the internal and external columns isolates the habit
externality: whether the decision maker accounts for the effect of today's
consumption input on tomorrow's habit stock. Lowering $\gamma$ from 8 to 4
scales the price term structures down without changing their shape.

**Calibration.** The production side is the same economy as Figures
11.1–11.3 at $\rho = 1.001$ and $\alpha = 0.0922$; the habit block adds
$\nu_h = 0.025$ per quarter, with $(\lambda, \tau)$ varied across the
panels. All values are those of the book's computation appendix.

**Checks.** The solver is the book's expansion code, used as-is and called
exactly as the book's computation appendix calls it — including its
warm-start chain — and the resulting figures line up with the chapter's.
More detail: [`README.md`](https://github.com/as7391746/QuantMFR-Colab)."""


def main():
    cells = [md(TITLE), md("# Codes"),
             code(SETUP), code(MODEL), code(SOLVE), code(PLOTFUN),
             code(FIG45), code(FIG67), code(FIG89),
             md(ANALYSIS)]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"name": "habit-elasticities.ipynb"},
                       "kernelspec": {"name": "python3",
                                      "display_name": "Python 3"}},
          "cells": cells}
    json.dump(nb, open("colab_habit.ipynb", "w"), indent=1)
    print(f"colab_habit.ipynb: {len(cells)} cells (7 code)")


if __name__ == "__main__":
    main()
