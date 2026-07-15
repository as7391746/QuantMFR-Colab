"""The internal-habit model from the book's "Uncertainty Expansion -
Computation Process" appendix, declared through the user-facing layer and
solved with the same engine — a second model through the same API.

Equations and parameters are transcribed from the appendix notebook
(quarterly time step epsilon = 0.25, annual drifts, sqrt(12) volatilities;
habit state X = log H/K, consumption services aggregate imh and the habit
stock with weight llambda and curvature tau).

Run:  python habit_example.py
"""

import numpy as np
import sympy as sp

from expansion import Model

PARAMS = {
    "epsilon": 0.25,                       # years per model step (a quarter)
    "beta": float(np.exp(-0.01 * 0.25)),   # exp(-delta*epsilon), delta = 0.01/yr
    "gamma": 1.001,
    "rho": 1.001,
    "a": 0.0922,                           # productivity
    "phi_1": 1.0 / 8.0,                    # 1/zeta
    "phi_2": 8.0,                          # zeta (annual curvature)
    "alpha_k": 0.04,                       # capital drift (annual)
    "beta_k": 0.04,                        # capital loading on Z (annual)
    "beta_z": 0.056,                       # Z mean reversion (annual)
    "beta_2": 0.194,                       # Y mean reversion (annual)
    "mu_2": 6.3e-6,                        # level of exp(Y)
    "sigma_k": np.sqrt(12.0) * np.array([0.92, 0.40, 0.0]),
    "sigma_z": np.sqrt(12.0) * np.array([0.0, 5.70, 0.0]),
    "sigma_y": np.sqrt(12.0) * np.array([0.0, 0.0, 0.00031]),
    "nu_h": 0.1 * 0.25,                    # habit depreciation (quarterly)
    "tau": 1.01,                           # consumption-services curvature
    "llambda": -0.0,                       # habit weight (0 = no habit channel)
}


def habit_model():
    m = Model(states=["Z", "Y", "X"], controls=["imh", "imk"], shocks=3)
    Z, Y, X = m["Z"], m["Y"], m["X"]
    imh, imk = m["imh"], m["imk"]
    p, q = m.p, m.q
    seps = sp.sqrt(p.epsilon)

    # long-run growth state
    m.state["Z"] = (Z - p.epsilon * p.beta_z * Z
                    + seps * sp.exp(Y / 2) * m.dot("sigma_z"))

    # log stochastic-volatility state
    m.state["Y"] = (Y - p.epsilon * p.beta_2 * (1 - p.mu_2 * sp.exp(-Y))
                    - q ** 2 / 2 * m.norm2("sigma_y") * sp.exp(-Y) * p.epsilon
                    + seps * sp.exp(-Y / 2) * m.dot("sigma_y"))

    # capital growth (also the model's growth process)
    capital = (p.epsilon * (p.phi_1 * sp.log(1 + p.phi_2 * imk)
                            - p.alpha_k + p.beta_k * Z
                            - q ** 2 / 2 * m.norm2("sigma_k") * sp.exp(Y))
               + seps * sp.exp(Y / 2) * m.dot("sigma_k"))
    m.growth = capital

    # habit stock relative to capital: X = log(H/K)
    m.state["X"] = (sp.log(sp.exp(-p.nu_h + X) + (1 - sp.exp(-p.nu_h)) * imh)
                    - capital)

    # consumption services: CES aggregate of imh and the habit stock
    m.consumption = (1 / (1 - p.tau)
                     * sp.log((1 - p.llambda) * imh ** (1 - p.tau)
                              + p.llambda * sp.exp((1 - p.tau) * X)))

    # resource constraint
    m.constraint = p.a - imh - imk
    return m


# ModelSol.ss as printed in the appendix notebook (its stored output)
APPENDIX_SS = np.array([-4.12667104, 0.15503472, 0.01613651, 0.07606349,
                        0.60576715, 0.0, -0.0, 1.0, 0.00485334, -0.0,
                        -11.97496092, -4.30652983])


if __name__ == "__main__":
    m = habit_model()
    sol = m.solve(
        PARAMS,
        start={"imh": 0.016, "imk": 0.076, "Y": float(np.log(6.3e-6)),
               "X": -4.3, "growth": 0.005},
        tol=1e-5)
    ss = np.asarray(sol.raw["ss"], dtype=float).flatten()
    print("steady state:", np.round(ss, 8))
    n = min(len(ss), len(APPENDIX_SS))
    diff = np.max(np.abs(ss[-n:] - APPENDIX_SS[-n:]))
    print(f"max abs diff vs the appendix notebook's stored ss: {diff:.2e}")
    curve = sol.exposure_elasticity(shock=1, T=160)
    print("consumption exposure elasticity, growth shock, t=1/40/160:",
          np.round(curve[[0, 39, 159]], 6))
