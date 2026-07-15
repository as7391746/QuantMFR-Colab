"""Model declaration: the AK adjustment-cost economy of Chapter 11, stated at
quarterly frequency in the chapter's notation.

The model (chapter section "An economy with long-run uncertainty", anchor
(AK_model); equation labels cited per line):

  exogenous states {eq}`equation2`:
    Z1' - Z1 = -nu_1 Z1 + exp(Z2/2) sigma_1 . W'
    Z2' - Z2 = -nu_2 [1 - mu_2 exp(-Z2)] - (q^2/2)|sigma_2|^2 exp(-Z2)
               + exp(-Z2/2) sigma_2 . W'
  capital (log) evolution [unlabeled display after {eq}`equation3`]:
    Khat' - Khat = (1/zeta) log(1 + zeta D2) + nu_k Z1 - iota_k
                   - (q^2/2)|sigma_k|^2 exp(Z2) + exp(Z2/2) sigma_k . W'
  resource constraint {eq}`equation3`:  D1 + D2 = alpha,  Chat - Ghat = log D1
  planner FOC {eq}`equation4` (MS eliminated):
    (1-beta) exp[(1-rho)(Chat-Ghat)] / D1
      = beta exp[(1-rho)(Rhat-Ghat)] / (1 + zeta D2)
  preferences {eq}`value_recur5`/{eq}`value_risk6`:
    Vhat = (1/(1-rho)) log[(1-beta) e^{(1-rho)Chat} + beta e^{(1-rho)Rhat}]
    Rhat = (1/(1-gamma)) log E[ e^{(1-gamma) Vhat'} | A_t ]

(The q^2 factors on the Ito-correction terms make the displayed equations the
q = 1 member of the small-noise family X_{t+1}(q) = psi[D_t, X_t, qW_{t+1}, q]
expanded in Appendix B "approach one".)

Notation map (chapter <-> HKT 2024 Table 1) with the annual->quarterly
conversions (drifts and flows /4; volatilities = sqrt(3) x monthly):

  chapter   HKT 2024 (annual)    quarterly value used here
  -------   -----------------    -----------------------------------
  beta      exp(-delta), d=0.01  exp(-0.01/4) = 0.99750
  rho       rho                  scenario
  gamma     gamma = 1 + 1/xi     scenario
  alpha     alpha (Table 3)      annual/4: {0.0205, 0.023, 0.027}
  zeta      phi = 8 /yr          32 (= 8x4: I/K is a quarterly ratio)
  iota_k    eta_k = 0.04 /yr     0.04/4 = 0.01
  nu_k      beta_k = 0.04 /yr    0.04/4 = 0.01
  nu_1      beta_1 = 0.056 /yr   0.056/4 = 0.014
  nu_2      beta_2 = 0.194 /yr   0.194/4 = 0.0485
  mu_2      mu_2 = 6.3e-6        6.3e-6 (a level, no /4)
  sigma_*   sqrt(12)x[M]         sqrt(3)x[M]
                                 ([M] = monthly-calibrated numbers
                                  .92/.40/5.7/.00031)

  where the chapter's Z2 is the log of HKT's Feller volatility state Z^2.

alpha is paired with rho (annual 0.082/0.092/0.108 for rho = 2/3, 1, 3/2,
HKT Table 3, chosen so all rho share steady-state consumption growth of
0.019/yr).
"""

import numpy as np

SQRT3 = np.sqrt(3.0)

# ---------------------------------------------------------------------------
# Quarterly calibration in the chapter's notation (annual sources: HKT 2024
# Tables 1 and 3; drifts /4, sigmas = sqrt(3) x the monthly vectors).
# gamma/rho carry small offsets away from the degenerate values (8.001 for 8,
# 1.001 for 1), as in the runs behind the published figures.
# ---------------------------------------------------------------------------

PARAMS = {
    "beta": float(np.exp(-0.0025)),   # quarterly discount factor
    "zeta": 32.0,                     # adjustment-cost curvature, quarterly-ratio form
    "iota_k": 0.01,                   # capital drift constant
    "nu_k": 0.01,                     # capital loading on Z1
    "nu_1": 0.014,                    # Z1 mean reversion
    "nu_2": 0.0485,                   # Z2 mean reversion
    "mu_2": 6.3e-6,                   # central tendency of exp(Z2)
    "sigma_k": SQRT3 * np.array([0.92, 0.40, 0.0]),
    "sigma_1": SQRT3 * np.array([0.0, 5.70, 0.0]),
    "sigma_2": SQRT3 * np.array([0.0, 0.0, 0.00031]),
    # elasticity horizon and state quantiles (Figs 11.1-11.3)
    "T": 160,                         # quarters = 40 years
    "quantiles": [0.1, 0.5, 0.9],     # Z2 stationary quantiles (Fig 11.3)
    "growth_shock": 1,                # W2 = growth-rate shock (0-indexed)
    "capital_shock": 0,               # W1 = capital shock
}

# alpha is annual in the rho<->alpha pairing (HKT Table 3); quarterly = /4.
# The base case uses 0.0922, matching published Figure 11.3.
ALPHA_ANNUAL_BY_RHO = {0.67: 0.082, 1.001: 0.092, 1.5: 0.108}
ALPHA_ANNUAL_BASE = 0.0922


def alpha_quarterly(alpha_annual):
    return alpha_annual / 4.0


# ---------------------------------------------------------------------------
# The 13 scenarios behind the three figures (mirrors the published runs).
# ---------------------------------------------------------------------------

SCENARIOS = (
    [{"id": "base", "gamma": 8.001, "rho": 1.001,
      "alpha_annual": ALPHA_ANNUAL_BASE, "figure": "price_quantiles"}]
    + [{"id": f"invk_r{r}", "gamma": 8.0, "rho": r,
        "alpha_annual": ALPHA_ANNUAL_BY_RHO[r], "figure": "invk_expo"}
       for r in [0.67, 1.001, 1.5]]
    + [{"id": f"g{g}_r{r}", "gamma": g, "rho": r,
        "alpha_annual": ALPHA_ANNUAL_BY_RHO[r], "figure": "six_panel"}
       for g in [1.001, 4.001, 8.001] for r in [0.67, 1.001, 1.5]]
)


def model(sc, params=PARAMS):
    """Bundle one scenario's primitives (all quarterly, chapter notation)."""
    p = dict(params)
    p["gamma"] = float(sc["gamma"])
    p["rho"] = float(sc["rho"])
    p["alpha"] = alpha_quarterly(sc["alpha_annual"])
    # first-order shock loadings of the two exogenous states around
    # Z1 = 0, exp(Z2) = mu_2  (chapter's displayed first-order dynamics):
    #   Z1^1' = (1 - nu_1) Z1^1 + sqrt(mu_2)   sigma_1 . W'
    #   Z2^1' = (1 - nu_2) Z2^1 + (1/sqrt(mu_2)) sigma_2 . W'
    p["s1"] = np.sqrt(p["mu_2"]) * p["sigma_1"]
    p["s2"] = p["sigma_2"] / np.sqrt(p["mu_2"])
    p["a1"] = 1.0 - p["nu_1"]
    p["a2"] = 1.0 - p["nu_2"]
    p["sk"] = np.sqrt(p["mu_2"]) * p["sigma_k"]   # first-order capital loading
    return p
