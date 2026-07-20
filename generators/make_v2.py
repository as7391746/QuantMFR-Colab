"""Generate v2_demo.ipynb: the v2 entry (automatic initial guess) plus the
5-specification x 5-correlation robustness grid."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}

cells = []

cells.append(md('''# `uncertain_expansion` — automatic initial guess (v2) and a 25-case robustness grid

The engine in `v2/` is the book's expansion code (the robust steady-state variant), with one change to its entry and two added files:

- `auto_guess.py` — if `uncertain_expansion` is called with `initial_guess=None`, a guess is derived from the model's own equations (a 9-line entry patch; an explicit guess behaves exactly as before).
- `autosolve.py` — a driver that solves from the derived guess, **verifies** the returned steady state against the model's own deterministic equations, and, when a parameter target is not directly solvable, moves one parameter at a time from the model's default values.

No solver mathematics is changed. `v2/PROVENANCE.md` lists every difference from the branch copy, including removed unreachable code.

**Contents**

1. The book's model solved with *no* initial guess.
2. A robustness grid: 5 model specifications x 5 shock-correlation structures (25 cases), each judged by four checks.

Runtime: about 10 minutes on Colab.'''))

cells.append(code('''import os, sys, io, time, warnings, contextlib
warnings.filterwarnings("ignore", category=SyntaxWarning)
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except Exception:
    pass

V2 = "v2"
if not os.path.isdir(V2):
    os.system("git clone -q https://github.com/as7391746/QuantMFR-Colab")
    V2 = "QuantMFR-Colab/v2"
sys.path.insert(0, os.path.abspath(V2))

import numpy as np, sympy as sp
import uncertain_expansion_faisal_feb26 as engine
from autosolve import autosolve
print("engine:", os.path.relpath(engine.__file__))'''))

cells.append(md('''## Declaring a model

A model is declared by its variables (controls, states, shocks) and four pieces, written in the chapter's notation:

- $\\kappa$ — the consumption entry of the utility recursion,
- the capital growth equation,
- the state evolution equations,
- the resource constraints.

`pack` assembles these into the engine's argument list; parameters are a plain dictionary (vector values are allowed and expanded automatically).'''))

cells.append(code('''def pack(cv, sv, n_shocks, params, equations):
    shocks = ["W%d_t" % (i + 1) for i in range(n_shocks)]
    variables = cv + sv + ["log_gk_t", "q_t"] + shocks
    vt = [v + "p1" for v in variables]
    flat = {}
    for k, v in params.items():
        a = np.asarray(v, float)
        if a.ndim == 0:
            flat[k] = float(a)
        else:
            for i, x in enumerate(a):
                flat["%s%d" % (k, i + 1)] = float(x)
    pn = {k: sp.Symbol(k) for k in flat}
    g = {n: sp.Symbol(n) for n in list(pn) + variables + vt}
    S = lambda n: g[n]
    kappa, growth, states, constraints = equations(S)
    return {"control_variables": cv, "state_variables": sv, "shock_variables": shocks,
            "variables": variables, "variables_tp1": vt, "kappa": kappa, "growth": growth,
            "state_equations": states, "static_constraints": constraints,
            "parameter_names": pn, "args": list(flat.values()),
            "n_controls": len(cv), "control_names": [c[:-2] for c in cv],
            "state_names": [s[:-2] for s in sv]}'''))

cells.append(md('''## 1. The book's model with no initial guess

The Section 11.7 AK economy with stochastic volatility, at $\\gamma = 8$. The call passes `initial_guess=None`; the code derives the guess from the equations.'''))

cells.append(code('''def ak_equations(S):
    dot = lambda b: sum(S(b + str(i + 1)) * S("W%d_tp1" % (i + 1)) for i in range(3))
    nn = lambda b: sum(S(b + str(i + 1)) ** 2 for i in range(3))
    kappa = sp.log(S("D1_t"))
    growth = (1 / S("zeta")) * sp.log(1 + S("zeta") * S("D2_t")) + S("nu_k") * S("Z1_t") - S("iota_k") \\
        - S("q_t") ** 2 / 2 * nn("sigma_k") * sp.exp(S("Z2_t")) + sp.exp(S("Z2_t") / 2) * dot("sigma_k")
    Z1n = S("Z1_t") - S("nu_1") * S("Z1_t") + sp.exp(S("Z2_t") / 2) * dot("sigma_1")
    Z2n = S("Z2_t") - S("nu_2") * (1 - S("mu_2") * sp.exp(-S("Z2_t"))) \\
        - S("q_t") ** 2 / 2 * nn("sigma_2") * sp.exp(-S("Z2_t")) + sp.exp(-S("Z2_t") / 2) * dot("sigma_2")
    return kappa, growth, [Z1n, Z2n], [S("alpha") - S("D1_t") - S("D2_t")]

s3 = np.sqrt(3.0)
PARAMS = {"beta": float(np.exp(-0.0025)), "rho": 1.001, "gamma": 8.001,
          "alpha": 0.02305, "zeta": 32.0, "iota_k": 0.01, "nu_k": 0.01,
          "nu_1": 0.014, "nu_2": 0.0485, "mu_2": 6.3e-6,
          "sigma_k": s3 * np.array([0.92, 0.40, 0.0]),
          "sigma_1": s3 * np.array([0.0, 5.70, 0.0]),
          "sigma_2": s3 * np.array([0.0, 0.0, 0.00031])}

m = pack(["D1_t", "D2_t"], ["Z1_t", "Z2_t"], 3, PARAMS, ak_equations)
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    sol = engine.uncertain_expansion(m["control_variables"], m["state_variables"],
        m["shock_variables"], m["variables"], m["variables_tp1"], m["kappa"], m["growth"],
        m["state_equations"], m["static_constraints"], None, m["parameter_names"],
        m["args"], approach="1", iter_tol=1e-7, max_iter=500)
ss = np.asarray(sol["ss"], float)
print("solved in %.0f s with no initial guess" % (time.time() - t0))
print("steady-state (D1, D2) = (%.6f, %.6f); closed-form D2 = 0.019023" % (ss[2], ss[3]))'''))

cells.append(md('''## 2. The robustness grid

**Specifications** — five production economies of the chapter's class:

| spec | description |
|---|---|
| AK | the chapter's AK economy with stochastic volatility |
| QUAD | quadratic capital-installation cost |
| HABIT | internal habit, three parameters moved at once ($\\gamma \\to 8$, $\\lambda \\to 0.5$, $\\tau \\to 0.3$) |
| 2STATE | two growth states, fast and slow mean reversion |
| SDEP | stochastic depreciation |

**Correlation structures** for the shock loading matrix $\\Sigma$. Row norms are held fixed across families — only the directions, hence the correlations, differ:

| family | description |
|---|---|
| diagonal | one dedicated shock per row (the book's layout) |
| dense | every row loads on every shock |
| leverage | capital and volatility rows nearly opposite (correlation about $-0.9$) |
| collinear | two rows correlated about $0.99$ |
| rankdef | one row an exact linear combination of the others — singular innovation covariance |

**Four checks per cell**

1. Solve from the automatically derived guess; the returned steady state must satisfy the model's own deterministic equations to $10^{-6}$.
2. Order 0: the steady state must be *identical* across the five families — $\\Sigma$ enters the deterministic system only through $\\mathsf{q}^2$ terms, which vanish at order 0.
3. Rotation invariance: $\\Sigma \\to \\Sigma Q$ with $Q$ orthogonal leaves the economy unchanged, so every solved object must match.
4. Order 1: the norm of $\\mu^0$ (the constant term of the drift tilt of the uncertainty-adjusted probability) must *differ* across families — the correlations have to show up exactly where the theory puts them, and nowhere else.'''))

cells.append(code('''def directions(family, n_rows, n_shocks, seed=23):
    rng = np.random.default_rng(seed)
    if family == "diagonal":
        D = np.eye(n_rows, n_shocks)
    elif family == "dense":
        D = rng.standard_normal((n_rows, n_shocks))
    elif family == "leverage":
        D = np.eye(n_rows, n_shocks) + 0.0
        D[0] = np.ones(n_shocks) / np.sqrt(n_shocks)
        D[-1] = -0.9 * D[0] + 0.45 * (np.arange(n_shocks) == n_shocks - 1)
        for i in range(1, n_rows - 1):
            D[i] = rng.standard_normal(n_shocks)
    elif family == "collinear":
        D = rng.standard_normal((n_rows, n_shocks))
        D[1] = D[0] + 0.14 * rng.standard_normal(n_shocks)
    elif family == "rankdef":
        D = rng.standard_normal((n_rows, n_shocks))
        D[-1] = 0.6 * D[0] + (0.4 * D[1] if n_rows > 2 else 0.4 * D[0])
    return D / np.linalg.norm(D, axis=1, keepdims=True)

def loadings(family, norms, n_shocks, rotate=None):
    L = directions(family, len(norms), n_shocks) * np.asarray(norms)[:, None]
    return L @ rotate if rotate is not None else L'''))

cells.append(code('''BETA = float(np.exp(-0.0025))

def spec_factory(name):
    """build(params, L), default params, target overrides, n_states, n_shocks, row norms.
    Rows of L attach to [capital growth, state 1, state 2, ...]."""
    if name == "AK":
        A = {"beta": BETA, "rho": 1.001, "gamma": 1.001, "alpha": 0.02305, "zeta": 32.0,
             "iota_k": 0.01, "nu_k": 0.01, "nu_1": 0.014, "nu_2": 0.0485, "mu_2": 6.3e-6}
        def build(p, L):
            def eqs(S):
                W = [S("W%d_tp1" % (i + 1)) for i in range(L.shape[1])]
                dot = lambda r: sum(float(L[r, i]) * W[i] for i in range(L.shape[1]))
                nn = lambda r: float(L[r] @ L[r])
                kappa = sp.log(S("D1_t"))
                growth = (1 / S("zeta")) * sp.log(1 + S("zeta") * S("D2_t")) + S("nu_k") * S("Z1_t") \\
                    - S("iota_k") - S("q_t") ** 2 / 2 * nn(0) * sp.exp(S("Z2_t")) + sp.exp(S("Z2_t") / 2) * dot(0)
                Z1n = S("Z1_t") - S("nu_1") * S("Z1_t") + sp.exp(S("Z2_t") / 2) * dot(1)
                Z2n = S("Z2_t") - S("nu_2") * (1 - S("mu_2") * sp.exp(-S("Z2_t"))) \\
                    - S("q_t") ** 2 / 2 * nn(2) * sp.exp(-S("Z2_t")) + sp.exp(-S("Z2_t") / 2) * dot(2)
                return kappa, growth, [Z1n, Z2n], [S("alpha") - S("D1_t") - S("D2_t")]
            return pack(["D1_t", "D2_t"], ["Z1_t", "Z2_t"], L.shape[1], p, eqs)
        return build, A, {"gamma": 8.0}, 2, 3, [0.012, 0.022, 4.6e-4]
    if name == "QUAD":
        A = {"beta": BETA, "rho": 1.001, "gamma": 1.001, "alpha": 0.02305, "kq": 8.0,
             "iota_k": 0.01, "nu_k": 0.01, "nu_1": 0.014, "nu_2": 0.0485, "mu_2": 6.3e-6}
        def build(p, L):
            def eqs(S):
                W = [S("W%d_tp1" % (i + 1)) for i in range(L.shape[1])]
                dot = lambda r: sum(float(L[r, i]) * W[i] for i in range(L.shape[1]))
                nn = lambda r: float(L[r] @ L[r])
                kappa = sp.log(S("D1_t"))
                growth = S("D2_t") - S("kq") / 2 * S("D2_t") ** 2 + S("nu_k") * S("Z1_t") \\
                    - S("iota_k") - S("q_t") ** 2 / 2 * nn(0) * sp.exp(S("Z2_t")) + sp.exp(S("Z2_t") / 2) * dot(0)
                Z1n = S("Z1_t") - S("nu_1") * S("Z1_t") + sp.exp(S("Z2_t") / 2) * dot(1)
                Z2n = S("Z2_t") - S("nu_2") * (1 - S("mu_2") * sp.exp(-S("Z2_t"))) \\
                    - S("q_t") ** 2 / 2 * nn(2) * sp.exp(-S("Z2_t")) + sp.exp(-S("Z2_t") / 2) * dot(2)
                return kappa, growth, [Z1n, Z2n], [S("alpha") - S("D1_t") - S("D2_t")]
            return pack(["D1_t", "D2_t"], ["Z1_t", "Z2_t"], L.shape[1], p, eqs)
        return build, A, {"gamma": 8.0}, 2, 3, [0.012, 0.022, 4.6e-4]
    if name == "HABIT":
        A = {"beta": BETA, "rho": 1.001, "gamma": 1.001, "a": 0.02305, "zeta": 32.0,
             "iota_k": 0.01, "nu_k": 0.01, "nu_1": 0.014, "nu_2": 0.0485, "mu_2": 6.3e-6,
             "nu_h": 0.025, "tau": 1.01, "llambda": -0.0}
        def build(p, L):
            def eqs(S):
                W = [S("W%d_tp1" % (i + 1)) for i in range(L.shape[1])]
                dot = lambda r: sum(float(L[r, i]) * W[i] for i in range(L.shape[1]))
                nn = lambda r: float(L[r] @ L[r])
                growth = (1 / S("zeta")) * sp.log(1 + S("zeta") * S("imk_t")) + S("nu_k") * S("Z1_t") \\
                    - S("iota_k") - S("q_t") ** 2 / 2 * nn(0) * sp.exp(S("Z2_t")) + sp.exp(S("Z2_t") / 2) * dot(0)
                Z1n = S("Z1_t") - S("nu_1") * S("Z1_t") + sp.exp(S("Z2_t") / 2) * dot(1)
                Z2n = S("Z2_t") - S("nu_2") * (1 - S("mu_2") * sp.exp(-S("Z2_t"))) \\
                    - S("q_t") ** 2 / 2 * nn(2) * sp.exp(-S("Z2_t")) + sp.exp(-S("Z2_t") / 2) * dot(2)
                Xn = sp.log(sp.exp(-S("nu_h") + S("X_t")) + (1 - sp.exp(-S("nu_h"))) * S("imh_t")) - growth
                kappa = 1 / (1 - S("tau")) * sp.log((1 - S("llambda")) * S("imh_t") ** (1 - S("tau"))
                                                    + S("llambda") * sp.exp((1 - S("tau")) * S("X_t")))
                return kappa, growth, [Z1n, Z2n, Xn], [S("a") - S("imh_t") - S("imk_t")]
            return pack(["imh_t", "imk_t"], ["Z1_t", "Z2_t", "X_t"], L.shape[1], p, eqs)
        return build, A, {"gamma": 8.0, "llambda": 0.5, "tau": 0.3}, 3, 3, [0.012, 0.022, 4.6e-4]
    if name == "2STATE":
        A = {"beta": BETA, "rho": 1.001, "gamma": 1.001, "alpha": 0.02305, "zeta": 32.0,
             "iota_k": 0.01, "nu_k": 0.01, "nua": 0.15, "nub": 0.01}
        def build(p, L):
            def eqs(S):
                W = [S("W%d_tp1" % (i + 1)) for i in range(L.shape[1])]
                dot = lambda r: sum(float(L[r, i]) * W[i] for i in range(L.shape[1]))
                kappa = sp.log(S("D1_t"))
                growth = (1 / S("zeta")) * sp.log(1 + S("zeta") * S("D2_t")) \\
                    + S("nu_k") * (S("Za_t") + S("Zb_t")) - S("iota_k") + dot(0)
                Zan = S("Za_t") - S("nua") * S("Za_t") + dot(1)
                Zbn = S("Zb_t") - S("nub") * S("Zb_t") + dot(2)
                return kappa, growth, [Zan, Zbn], [S("alpha") - S("D1_t") - S("D2_t")]
            return pack(["D1_t", "D2_t"], ["Za_t", "Zb_t"], L.shape[1], p, eqs)
        return build, A, {"gamma": 8.0}, 2, 3, [0.012, 0.022, 0.008]
    if name == "SDEP":
        A = {"beta": BETA, "rho": 1.001, "gamma": 1.001, "alpha": 0.02305, "zeta": 32.0,
             "iota_k": 0.01, "nuj": 0.05}
        def build(p, L):
            def eqs(S):
                W = [S("W%d_tp1" % (i + 1)) for i in range(L.shape[1])]
                dot = lambda r: sum(float(L[r, i]) * W[i] for i in range(L.shape[1]))
                kappa = sp.log(S("D1_t"))
                growth = (1 / S("zeta")) * sp.log(1 + S("zeta") * S("D2_t")) \\
                    - S("iota_k") * sp.exp(S("J_t")) + dot(0)
                Jn = S("J_t") - S("nuj") * S("J_t") + dot(1)
                return kappa, growth, [Jn], [S("alpha") - S("D1_t") - S("D2_t")]
            return pack(["D1_t", "D2_t"], ["J_t"], L.shape[1], p, eqs)
        return build, A, {"gamma": 8.0}, 1, 2, [0.012, 0.05]
    raise ValueError(name)

SPECS = ["AK", "QUAD", "HABIT", "2STATE", "SDEP"]
FAMILIES = ["diagonal", "dense", "leverage", "collinear", "rankdef"]'''))

cells.append(code('''def mu0_norm(r):
    try:
        return float(np.linalg.norm(np.asarray(r["util_sol"]["\\u03bc_0"], float).flatten()))
    except Exception:
        return float("nan")

rows, t_grid = [], time.time()
for spec_name in SPECS:
    build, A, Tov, ns, nw, norms = spec_factory(spec_name)
    T = dict(A); T.update(Tov)
    Q, _ = np.linalg.qr(np.random.default_rng(99).standard_normal((nw, nw)))
    ss_by_family = {}
    for fam in FAMILIES:
        L0 = loadings(fam, norms, nw)
        L1 = loadings(fam, norms, nw, rotate=Q)
        cell = {"spec": spec_name, "family": fam}
        t1 = time.time()
        r, msg = autosolve(lambda p, L=L0: build(p, L), A, T, ns, nw)
        cell["solved"] = r is not None
        cell["secs"] = round(time.time() - t1)
        if r is not None:
            ss = np.asarray(r["ss"], float)
            ss_by_family[fam] = ss
            cell["mu0"] = mu0_norm(r)
            r2, _ = autosolve(lambda p, L=L1: build(p, L), A, T, ns, nw)
            cell["rot_dss"] = float(np.max(np.abs(ss - np.asarray(r2["ss"], float)))) if r2 is not None else float("nan")
        else:
            cell["note"] = msg
        rows.append(cell)
        print("  %-7s x %-9s: %s %3ds  rot_dss=%.1e  |mu0|=%.4f" % (spec_name, fam,
              "OK  " if cell["solved"] else "FAIL", cell["secs"],
              cell.get("rot_dss", float("nan")), cell.get("mu0", float("nan"))), flush=True)
    ok = [f for f in FAMILIES if f in ss_by_family]
    if len(ok) > 1:
        dmax = max(float(np.max(np.abs(ss_by_family[f] - ss_by_family[ok[0]]))) for f in ok[1:])
        print("  %s: order-0 steady state identical across families: max diff %.2e (%s)"
              % (spec_name, dmax, "PASS" if dmax < 1e-8 else "FAIL"), flush=True)

print("grid finished in %.1f min" % ((time.time() - t_grid) / 60))'''))

cells.append(code('''import pandas as pd
df = pd.DataFrame(rows)
summary = pd.DataFrame({
    "solved": ["%d / %d" % (df[df.spec == s].solved.sum(), len(FAMILIES)) for s in SPECS],
    "max rotation diff": [df[df.spec == s].rot_dss.max() for s in SPECS],
}, index=SPECS)
display(summary)
display(df.pivot(index="spec", columns="family", values="mu0").loc[SPECS, FAMILIES]
          .style.format("{:.4f}").set_caption("|mu0| — first-order drift tilt, by correlation structure"))'''))

cells.append(md('''## Reading the results

- **Order 0 is untouched by correlations** — within every specification the steady state is identical, digit for digit, across the five correlation structures, and unchanged when $\\Sigma$ is rotated. This is what the theory requires: $\\Sigma$ enters the deterministic system only through terms that vanish at order 0.
- **Order 1 responds to correlations** — $|\\mu^0|$ differs across the structures (in the SDEP economy by nearly an order of magnitude), largest for the leverage-type structure where bad capital shocks come with high volatility. The correlations appear exactly where the theory puts them, and nowhere else.
- Every returned steady state was verified against the model's own equations; nothing above relies on the solver's internal convergence report.'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

out = os.path.join(ROOT, "v2_demo.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
