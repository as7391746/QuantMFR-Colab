#!/usr/bin/env python3
"""Generate colab.ipynb — Chapter 11, Figures 11.1-11.3.

Layout: two sections. Codes (setup, model, solve, one cell per figure) and
Analysis. The solve step imports the expansion engine (uncertain_expansion)
directly from this repository's src/, through the declaration layer in
expansion/. The independent direct implementation (model.py / solve.py in
the repository root) is kept for cross-checking and is not used here.
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
directly from this repository's `src/`). Parameters follow the appendix
table and Hansen–Khorrami–Tourre (2024), converted to quarterly.

![method](https://raw.githubusercontent.com/as7391746/QuantMFR-Colab/main/assets/method.png)

*The pipeline and where each step lives in this notebook. Risk aversion is
rescaled with the perturbation ($\gamma-1=(\gamma_o-1)/\mathsf q$), so the
recursive-utility change of measure $N^0$ already matters at first order: it
shifts the shock mean to $\mu^0$, which is what generates the shock-price
elasticities in Figures 11.1 and 11.3.*

**Runtime → Run all** (~3 min: a shallow clone, then thirteen expansion
solves of a few seconds each). To experiment, edit `PARAMS` or `SCENARIOS`
and re-run."""

SETUP = """# ================================ SETUP =================================
# Fetch this repository and put the expansion engine (src/) and the
# declaration layer (expansion/) on the path.
%pip -q install autograd
import os, sys, io, time, contextlib
if not os.path.isdir("QuantMFR-Colab"):
    !git clone -q --depth 1 https://github.com/as7391746/QuantMFR-Colab
sys.path.insert(0, os.path.abspath("QuantMFR-Colab/src"))
sys.path.insert(0, os.path.abspath("QuantMFR-Colab/expansion"))

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from uncertain_expansion import uncertain_expansion   # the engine, direct
from expansion import Model                            # declaration layer
import uncertain_expansion as _engine
print("engine:", _engine.__file__)"""

MODEL = r"""# ================================ MODEL =================================
# Chapter 11 AK economy, quarterly, in the chapter's own notation.
# States {eq}`equation2`; capital evolution (display after {eq}`equation3`);
# resource constraint {eq}`equation3`; preferences {eq}`value_recur5`/
# {eq}`value_risk6`. Parameters: appendix table / HKT 2024 (annual drifts
# / 4, volatilities = sqrt(3) x the monthly vectors).
SQRT3 = np.sqrt(3.0)
PARAMS = {
    "beta": np.exp(-0.01 / 4),        # quarterly discount factor
    "zeta": 32.0,                     # adjustment-cost curvature (= 8 x 4)
    "iota_k": 0.01,                   # capital drift constant
    "nu_k": 0.01,                     # capital loading on Z1
    "nu_1": 0.014,                    # Z1 mean reversion
    "nu_2": 0.0485,                   # Z2 mean reversion
    "mu_2": 6.3e-6,                   # central tendency of exp(Z2)
    "sigma_k": SQRT3 * np.array([0.92, 0.40, 0.0]),
    "sigma_1": SQRT3 * np.array([0.0, 5.70, 0.0]),
    "sigma_2": SQRT3 * np.array([0.0, 0.0, 0.00031]),
}
# alpha is annual, paired with rho so all rho share the same steady-state
# growth (HKT Table 3); quarterly = /4. The base case uses 0.0922.
ALPHA_ANNUAL = {0.67: 0.082, 1.001: 0.092, 1.5: 0.108}
ALPHA_BASE = 0.0922

m = Model(states=["Z1", "Z2"], controls=["D1", "D2"], shocks=3)
Z1, Z2, D1, D2 = m["Z1"], m["Z2"], m["D1"], m["D2"]
p, q = m.p, m.q

# exogenous states {eq}`equation2`
m.state["Z1"] = Z1 - p.nu_1 * Z1 + sp.exp(Z2 / 2) * m.dot("sigma_1")
m.state["Z2"] = (Z2 - p.nu_2 * (1 - p.mu_2 * sp.exp(-Z2))
                 - q**2 / 2 * m.norm2("sigma_2") * sp.exp(-Z2)
                 + sp.exp(-Z2 / 2) * m.dot("sigma_2"))
# capital evolution (display after {eq}`equation3`)
m.growth = ((1 / p.zeta) * sp.log(1 + p.zeta * D2) + p.nu_k * Z1 - p.iota_k
            - q**2 / 2 * m.norm2("sigma_k") * sp.exp(Z2)
            + sp.exp(Z2 / 2) * m.dot("sigma_k"))
# consumption and the resource constraint {eq}`equation3`
m.consumption = sp.log(D1)
m.constraint = p.alpha - D1 - D2

# the 13 scenarios behind the three figures
SCENARIOS = (
    [{"id": "base", "gamma": 8.001, "rho": 1.001,
      "alpha_annual": ALPHA_BASE, "figure": "price_quantiles"}]
    + [{"id": f"invk_r{r}", "gamma": 8.0, "rho": r,
        "alpha_annual": ALPHA_ANNUAL[r], "figure": "invk_expo"}
       for r in [0.67, 1.001, 1.5]]
    + [{"id": f"g{g}_r{r}", "gamma": g, "rho": r,
        "alpha_annual": ALPHA_ANNUAL[r], "figure": "six_panel"}
       for g in [1.001, 4.001, 8.001] for r in [0.67, 1.001, 1.5]]
)"""

SOLVE = r"""# ================================ SOLVE =================================
# Each scenario: second-order small-noise expansion via uncertain_expansion
# (order 0/1/2 and the N^0 change of measure inside the engine), then
# Borovicka-Hansen exposure/price elasticities from the solution.
T, GROWTH_SHOCK, CAPITAL_SHOCK = 160, 1, 0

def solve_scenario(sc):
    alpha_q = sc["alpha_annual"] / 4.0
    params = dict(PARAMS)
    params.update({"gamma": float(sc["gamma"]), "rho": float(sc["rho"]),
                   "alpha": alpha_q})
    # rough steady-state starting values, in the model's own terms
    start = {"D2": 0.019, "D1": max(alpha_q - 0.019, 1e-3),
             "Z2": float(np.log(6.3e-6)), "growth": 0.005}
    return m.solve(params, start=start)

results = {}
t0 = time.time()
for sc in SCENARIOS:
    t1 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):   # engine iteration log
        sol = solve_scenario(sc)
    if sc["figure"] == "price_quantiles":
        for quantile in (0.1, 0.5, 0.9):
            for name, shock in (("growth", GROWTH_SHOCK),
                                ("capital", CAPITAL_SHOCK)):
                results[f"{sc['id']}/price_{name}_q{quantile}"] = \
                    sol.price_elasticity(shock, T, quantile)
    elif sc["figure"] == "invk_expo":
        invk = sol.increment(sp.log(m["D2"]))
        results[f"{sc['id']}/expo_invk_growth_q0.5"] = \
            sol.exposure_elasticity(GROWTH_SHOCK, T, process=invk)
    else:
        results[f"{sc['id']}/expo_growth_q0.5"] = \
            sol.exposure_elasticity(GROWTH_SHOCK, T)
        results[f"{sc['id']}/price_growth_q0.5"] = \
            sol.price_elasticity(GROWTH_SHOCK, T)
    print(f"{sc['id']}: solved in {time.time() - t1:.1f}s", flush=True)
print(f"{len(SCENARIOS)} scenarios in {time.time() - t0:.0f}s")"""

FIG1 = r"""# ============================ FIGURE 11.1 ===============================
# Consumption exposure (left) and price (right) elasticities for the
# growth-rate shock; rows vary gamma, curves vary rho (alpha paired to rho).
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
# The sign of the initial response flips with rho relative to 1 (IES = 1).
plt.figure(figsize=(6, 4))
for r in rhos:
    plt.plot(yrs, results[f"invk_r{r}/expo_invk_growth_q0.5"], label=fr"$\rho={r}$")
plt.axhline(0, color="k", lw=0.6)
plt.xlabel("years"); plt.legend(frameon=False); plt.tight_layout(); plt.show()"""

FIG3 = r"""# ============================ FIGURE 11.3 ===============================
# Price elasticities at the 0.1 / 0.5 / 0.9 stochastic-volatility quantiles,
# growth and capital shocks; gamma = 8, rho = 1.
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
(0.082 / 0.092 / 0.108 annually) so all specifications share the same
1.9%/yr steady-state growth (HKT 2024, Table 3). All parameters here are
quarterly: annual drifts ÷ 4, volatilities $=\sqrt3\,\times$ the
monthly-calibrated vectors, $\beta = e^{-0.01/4}$.

**Checks.** The solver is the expansion engine itself, imported unmodified
from `src/` (RiskUncertaintyValue, branch `Planners_with_External`, commit
`09ca5df`). The curves match the runs behind the published figures; an
independent direct implementation of the same expansion (`model.py` /
`solve.py` in this repository) agrees with the engine at tight tolerance
and is kept for cross-checking. More detail:
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
