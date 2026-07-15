"""The Chapter 11 AK economy through the user-facing expansion layer:
declare the chapter's equations in the chapter's notation, feed the
appendix's quarterly parameter table, and generate Figures 11.1-11.3.

Run:  python ak_example.py           (all 13 solves + 3 figures)
      python ak_example.py base      (the single base solve, quick check)
"""

import sys
import time

import numpy as np
import sympy as sp

from expansion import Model, annual_to_quarterly

# ---------------------------------------------------------------------------
# Parameters: the appendix table (quarterly), preferences per scenario.
# The annual -> quarterly conversions, stated explicitly:
# ---------------------------------------------------------------------------

CONVERTED = annual_to_quarterly(
    rates={"iota_k": 0.04, "nu_k": 0.04, "nu_1": 0.056, "nu_2": 0.194},
    volatilities_monthly={"sigma_k": [0.92, 0.40, 0.0],
                          "sigma_1": [0.0, 5.70, 0.0],
                          "sigma_2": [0.0, 0.0, 0.00031]},
    ratios={"zeta": 8.0},
    verbose=False,
)

PARAMS = {
    "beta": np.exp(-0.01 / 4),      # exp(-delta/4), annual delta = 0.01
    "mu_2": 6.3e-6,                 # mean of exp(Z2): a level, no conversion
    **CONVERTED,
}

# productivity alpha is paired with rho so that all rho share the same
# steady-state growth (annual values; quarterly = /4)
ALPHA_ANNUAL = {0.67: 0.082, 1.001: 0.092, 1.5: 0.108}
ALPHA_BASE = 0.0922

# ---------------------------------------------------------------------------
# Model: the chapter's equations, in the chapter's notation.
# ---------------------------------------------------------------------------


def ak_model():
    m = Model(states=["Z1", "Z2"], controls=["D1", "D2"], shocks=3)
    Z1, Z2, D1, D2 = m["Z1"], m["Z2"], m["D1"], m["D2"]
    p, q = m.p, m.q

    # exogenous states {eq}`equation2`
    m.state["Z1"] = (Z1 - p.nu_1 * Z1
                     + sp.exp(Z2 / 2) * m.dot("sigma_1"))
    m.state["Z2"] = (Z2 - p.nu_2 * (1 - p.mu_2 * sp.exp(-Z2))
                     - q ** 2 / 2 * m.norm2("sigma_2") * sp.exp(-Z2)
                     + sp.exp(-Z2 / 2) * m.dot("sigma_2"))

    # capital evolution (display after {eq}`equation3`)
    m.growth = ((1 / p.zeta) * sp.log(1 + p.zeta * D2)
                + p.nu_k * Z1 - p.iota_k
                - q ** 2 / 2 * m.norm2("sigma_k") * sp.exp(Z2)
                + sp.exp(Z2 / 2) * m.dot("sigma_k"))

    # consumption and the resource constraint {eq}`equation3`
    m.consumption = sp.log(D1)
    m.constraint = p.alpha - D1 - D2
    return m


def start_values(alpha_quarterly):
    """Rough steady-state starting values, in the model's own terms. All
    rho share the same steady-state growth by construction, so investment
    starts near 0.019 and consumption absorbs the rest of output."""
    return {"D2": 0.019, "D1": max(alpha_quarterly - 0.019, 1e-3),
            "Z2": float(np.log(6.3e-6)), "growth": 0.005}

T = 160                     # horizon: 40 years of quarters
GROWTH_SHOCK, CAPITAL_SHOCK = 1, 0

SCENARIOS = (
    [{"id": "base", "gamma": 8.001, "rho": 1.001,
      "alpha_annual": ALPHA_BASE, "figure": "price_quantiles"}]
    + [{"id": f"invk_r{r}", "gamma": 8.0, "rho": r,
        "alpha_annual": ALPHA_ANNUAL[r], "figure": "invk_expo"}
       for r in [0.67, 1.001, 1.5]]
    + [{"id": f"g{g}_r{r}", "gamma": g, "rho": r,
        "alpha_annual": ALPHA_ANNUAL[r], "figure": "six_panel"}
       for g in [1.001, 4.001, 8.001] for r in [0.67, 1.001, 1.5]]
)


def solve_scenario(m, sc):
    alpha_quarterly = sc["alpha_annual"] / 4.0
    params = dict(PARAMS)
    params.update({"gamma": float(sc["gamma"]), "rho": float(sc["rho"]),
                   "alpha": alpha_quarterly})
    return m.solve(params, start=start_values(alpha_quarterly))


def run(scenario_ids=None):
    m = ak_model()
    results = {}
    for sc in SCENARIOS:
        if scenario_ids and sc["id"] not in scenario_ids:
            continue
        t0 = time.time()
        sol = solve_scenario(m, sc)
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
        print(f"{sc['id']}: solved in {time.time() - t0:.1f}s", flush=True)
    return results


def figures(results):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("darkgrid")
    yrs = np.arange(1, T + 1) / 4

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    for i, g in enumerate([1.001, 4.001, 8.001]):
        for r in [0.67, 1.001, 1.5]:
            axes[i, 0].plot(yrs, results[f"g{g}_r{r}/expo_growth_q0.5"],
                            label=fr"$\rho={r}$")
            axes[i, 1].plot(yrs, results[f"g{g}_r{r}/price_growth_q0.5"])
        axes[i, 0].set_ylabel(fr"$\gamma={g:.0f}$")
    axes[0, 0].set_title("exposure elasticity")
    axes[0, 1].set_title("price elasticity")
    axes[0, 0].legend(frameon=False)
    fig.savefig("fig_11_1.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(6, 4))
    for r in [0.67, 1.001, 1.5]:
        plt.plot(yrs, results[f"invk_r{r}/expo_invk_growth_q0.5"],
                 label=fr"$\rho={r}$")
    plt.axhline(0, color="k", lw=0.6)
    plt.xlabel("years")
    plt.legend(frameon=False)
    plt.savefig("fig_11_2.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for name, ax in [("growth", axes[0]), ("capital", axes[1])]:
        for quantile in [0.1, 0.5, 0.9]:
            ax.plot(yrs, results[f"base/price_{name}_q{quantile}"],
                    label=f"q = {quantile}")
        ax.set_title(f"{name} shock")
        ax.set_xlabel("years")
    axes[0].legend(frameon=False)
    fig.savefig("fig_11_3.png", dpi=150, bbox_inches="tight")
    print("figures: fig_11_1.png fig_11_2.png fig_11_3.png")


if __name__ == "__main__":
    ids = sys.argv[1:] or None
    res = run(ids)
    np.savez("output_solution.npz", **res)
    if ids is None:
        figures(res)
