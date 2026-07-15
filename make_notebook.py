#!/usr/bin/env python3
"""Generate colab.ipynb FROM the folder's .py sources.

Layout (per Kunjian's spec): two sections.
  Codes    — exactly five cells: MODEL, SOLVE, FIG 11.1, FIG 11.2, FIG 11.3
  Analysis — short interpretive text (economics, calibration notes,
             verification summary)

The two .py files (model.py / solve.py) remain the single source of truth:
their bodies are spliced verbatim into the MODEL and SOLVE cells at the
`# ====` section markers. Edit the .py, re-run this script, push — the
notebook cannot drift from the code of record. The notebook is fully
standalone (no git clone, no downloads).
"""

import json
import re

STRIP_LINES = re.compile(
    r"^(import numpy as np$|from scipy[^\n]*|from model import[^\n]*)", re.M)


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None, "source": src.splitlines(keepends=True)}


def strip_module(text):
    text = re.sub(r'^""".*?"""\n', "", text, flags=re.S)
    text = text.split("if __name__")[0]
    text = STRIP_LINES.sub("", text)
    text = re.sub(r"\ndef main\(\):.*", "\n", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def sections(path):
    text = strip_module(open(path).read())
    parts = re.split(r"# =+\n# (.+)\n(?:#[^\n]*\n)*?# =+\n", text)
    out = []
    if parts[0].strip():
        out.append(("", parts[0]))
    for i in range(1, len(parts), 2):
        out.append((parts[i].strip(), parts[i + 1]))
    return [(t, b.strip() + "\n") for t, b in out if b.strip()]


def get(secs, prefix):
    for t, b in secs:
        if t.startswith(prefix):
            return b
    raise KeyError(prefix)


TITLE = r"""# Chapter 11: Figures 11.1–11.3

The AK production economy of Chapter 11, solved with the second-order
small-noise expansion described in the appendix ("approach one"), and the
Borovička–Hansen (2014) shock elasticities computed from that solution.
Everything is quarterly and in the chapter's notation; parameters follow the
appendix table and Hansen–Khorrami–Tourre (2024).

![method](https://raw.githubusercontent.com/as7391746/QuantMFR-Colab/main/assets/method.png)

*The pipeline and where each step lives in this notebook. Risk aversion is
rescaled with the perturbation ($\gamma-1=(\gamma_o-1)/\mathsf q$), so the
recursive-utility change of measure $N^0$ already matters at first order: it
shifts the shock mean to $\mu^0$, which is what generates the shock-price
elasticities in Figures 11.1 and 11.3.*

**Runtime → Run all** (~10 s; the 13-scenario solve itself is ~0.2 s). To
experiment, edit `PARAMS` or `SCENARIOS` in the model cell and re-run."""

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

**Checks.** The curves match the run behind the published figures to
$10^{-12}$–$10^{-5}$ relative (the residual is that run's iteration
tolerance), and substituting the solved policy and value function back into
the chapter's exact recursions leaves residuals of third order, as a
second-order expansion should. More detail:
[`README.md`](https://github.com/as7391746/QuantMFR-Colab)."""

RUN_BLOCK = '''# ---- run the 13 scenarios behind Figures 11.1-11.3 ----
def solve_scenario(sc, params=PARAMS):
    p = model(sc, params)
    o0 = order0(p)
    o1 = order1(p, o0)
    o2 = order2(p, o0, o1)
    return p, elastic_build(p, o0, o1, o2)

t0 = time.time()
results = {}
for sc in SCENARIOS:
    p, inc = solve_scenario(sc)
    T = p["T"]
    if sc["figure"] == "price_quantiles":
        for q in p["quantiles"]:
            for name, shock in [("growth", p["growth_shock"]),
                                ("capital", p["capital_shock"])]:
                results[f"{sc['id']}/price_{name}_q{q}"] = \\
                    price_elasticity(inc, T, shock, percentile=q)
    elif sc["figure"] == "invk_expo":
        results[f"{sc['id']}/expo_invk_growth_q0.5"] = \\
            exposure_elasticity(inc, T, p["growth_shock"], process="invk")
    else:
        results[f"{sc['id']}/expo_growth_q0.5"] = \\
            exposure_elasticity(inc, T, p["growth_shock"])
        results[f"{sc['id']}/price_growth_q0.5"] = \\
            price_elasticity(inc, T, p["growth_shock"])
print(f"{len(SCENARIOS)} scenarios solved in {time.time()-t0:.2f}s")'''

FIG1 = r'''# ============================ FIGURE 11.1 ===============================
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
plt.tight_layout(); plt.show()'''

FIG2 = r'''# ============================ FIGURE 11.2 ===============================
# Investment-capital exposure elasticity, growth-rate shock, gamma = 8.
# The sign of the initial response flips with rho relative to 1 (IES = 1).
plt.figure(figsize=(6, 4))
for r in rhos:
    plt.plot(yrs, results[f"invk_r{r}/expo_invk_growth_q0.5"], label=fr"$\rho={r}$")
plt.axhline(0, color="k", lw=0.6)
plt.xlabel("years"); plt.legend(frameon=False); plt.tight_layout(); plt.show()'''

FIG3 = r'''# ============================ FIGURE 11.3 ===============================
# Price elasticities at the 0.1 / 0.5 / 0.9 stochastic-volatility quantiles,
# growth and capital shocks; gamma = 8, rho = 1.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for name, ax in [("growth", axes[0]), ("capital", axes[1])]:
    for q in [0.1, 0.5, 0.9]:
        ax.plot(yrs, results[f"base/price_{name}_q{q}"], label=f"q = {q}")
    ax.set_title(f"{name} shock"); ax.set_xlabel("years")
axes[0].legend(frameon=False)
plt.tight_layout(); plt.show()'''


def main():
    model_body = "\n\n".join(b for _, b in sections("model.py"))
    solve_secs = sections("solve.py")

    cell_model = (
        "# ================================ MODEL =================================\n"
        "# Chapter 11 AK economy, quarterly, in the chapter's own notation.\n"
        "# States {eq}`equation2`; capital evolution; resource constraint\n"
        "# {eq}`equation3`; preferences {eq}`value_recur5`/{eq}`value_risk6`;\n"
        "# planner FOC {eq}`equation4`. Parameter provenance in the comments\n"
        "# (HKT 2024 annual values, converted to quarterly).\n"
        "import time\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from scipy.optimize import brentq\n"
        "from scipy.stats import norm\n\n"
        + model_body)

    cell_solve = (
        "# ================================ SOLVE =================================\n"
        "# Second-order small-noise expansion (appendix B, approach one):\n"
        "#   order 0: deterministic steady state; rho=1 reproduces the chapter's\n"
        "#            closed form D2* = ((beta-1)+beta*alpha)/(beta+(1-beta)*zeta)\n"
        "#   order 1: 4x4 linear system for (upsilon_1, d_1); the recursive-utility\n"
        "#            change of measure N^0 shifts the shock mean by\n"
        "#            mu0 = (1-gamma)*Sigma_w   (appendix `first_recursive_update`)\n"
        "#   order 2: affine in its 8 coefficients -> solved exactly by probing\n"
        "#            (appendix `second_recursive_update`, all under N^0)\n"
        "# Pricing: every log increment is linear-quadratic in (X1, X2, W); Gaussian\n"
        "# integrals keep the class closed, so Borovicka-Hansen (2014) elasticities\n"
        "# follow from one backward recursion (no simulation). SDF as in the runs\n"
        "# behind the published figures (normalized quadratic N-tilde).\n\n"
        + get(solve_secs, "Quadratic-form container") + "\n\n"
        + get(solve_secs, "Order 0") + "\n\n"
        + get(solve_secs, "Order 1") + "\n\n"
        + get(solve_secs, "Order 2") + "\n\n"
        + get(solve_secs, "LQ") + "\n\n"
        + get(solve_secs, "Assemble") + "\n\n"
        + get(solve_secs, "Elasticities") + "\n\n"
        + RUN_BLOCK
    )

    cells = [md(TITLE), md("# Codes"),
             code(cell_model), code(cell_solve),
             code(FIG1), code(FIG2), code(FIG3),
             md(ANALYSIS)]

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"name": "ak-elasticities-chapter.ipynb"},
                       "kernelspec": {"name": "python3",
                                      "display_name": "Python 3"}},
          "cells": cells}
    json.dump(nb, open("colab.ipynb", "w"), indent=1)
    print(f"colab.ipynb: {len(cells)} cells (5 code)")


if __name__ == "__main__":
    main()
