"""The six-model lineup, importable with NO top-level execution.
Every builder is written from the original paper (see per-model notes);
every calibration number traces to a published table; SEEDS are each
paper's own closed-form steady-state restrictions. Quarterly throughout;
conversion rules: beta_q=beta_a^(1/4) (or ^(1/n) from monthly),
(1-delta)_q=(1-delta)^(1/n), mu_q=mu/n, iid sigma_q=sigma*sqrt(3) from
monthly or /2 from annual, rho_q=rho^(1/n) (or ^3 from monthly), AR
innovation matched to the same unconditional variance."""
import numpy as np
import sympy as sp


def spec_pack(cv, sv, n_shocks, builder, external=False):
    """Assemble a builder into the engine argument dict (params -> spec)."""
    shocks = [f"W{i+1}_t" for i in range(n_shocks)]
    variables = cv + sv + ["log_gk_t", "q_t"] + shocks
    vt = [v + "p1" for v in variables]

    def build(params):
        flat = {}
        for k, v in params.items():
            a = np.asarray(v, float)
            if a.ndim == 0:
                flat[k] = float(a)
            else:
                for i, x in enumerate(a):
                    flat[f"{k}{i+1}"] = float(x)
        pn = {k: sp.Symbol(k) for k in flat}
        g = {n: sp.Symbol(n) for n in list(pn) + variables + vt}
        S = lambda n: g[n]
        kappa, growth, states, constraints = builder(S)
        return {"control_variables": cv, "state_variables": sv,
                "shock_variables": shocks, "variables": variables,
                "variables_tp1": vt, "kappa": kappa, "growth": growth,
                "state_equations": states, "static_constraints": constraints,
                "parameter_names": pn, "args": list(flat.values()),
                "n_controls": len(cv), "control_names": [c[:-2] for c in cv],
                "state_names": [s[:-2] for s in sv], "external_habit": external}
    return build


def with_loadings(build, L):
    """Wrap a spec builder so the shock vector W is replaced by L @ W:
    every W_i in the equations becomes sum_j L[i,j] W_j. Row norms of L
    equal 1 to preserve each equation's own published volatility; L = Q
    orthogonal is a pure rotation (economy unchanged)."""
    L = np.asarray(L, float)

    def build2(params):
        spec = build(params)
        n = L.shape[0]
        W = [sp.Symbol(f"W{i+1}_tp1") for i in range(n)]
        tmp = {W[i]: sp.Symbol(f"WTMP{i+1}") for i in range(n)}
        mix = {sp.Symbol(f"WTMP{i+1}"):
               sum(float(L[i, j]) * W[j] for j in range(n)) for i in range(n)}
        out = dict(spec)
        out["kappa"] = spec["kappa"].subs(tmp).subs(mix)
        out["growth"] = spec["growth"].subs(tmp).subs(mix)
        out["state_equations"] = [e.subs(tmp).subs(mix) for e in spec["state_equations"]]
        out["static_constraints"] = [e.subs(tmp).subs(mix) for e in spec["static_constraints"]]
        return out
    return build2


MODELS = {}

# --- 1. AK stochastic volatility (book, Section 11.7; appendix table) ------
def _ak(S):
    dot = lambda b: sum(S(b + str(i + 1)) * S(f"W{i+1}_tp1") for i in range(3))
    nn = lambda b: sum(S(b + str(i + 1)) ** 2 for i in range(3))
    kappa = sp.log(S("D1_t"))
    growth = (1 / S("zeta")) * sp.log(1 + S("zeta") * S("D2_t")) + S("nu_k") * S("Z1_t") \
        - S("iota_k") - S("q_t") ** 2 / 2 * nn("sigma_k") * sp.exp(S("Z2_t")) \
        + sp.exp(S("Z2_t") / 2) * dot("sigma_k")
    Z1n = S("Z1_t") - S("nu_1") * S("Z1_t") + sp.exp(S("Z2_t") / 2) * dot("sigma_1")
    Z2n = S("Z2_t") - S("nu_2") * (1 - S("mu_2") * sp.exp(-S("Z2_t"))) \
        - S("q_t") ** 2 / 2 * nn("sigma_2") * sp.exp(-S("Z2_t")) + sp.exp(-S("Z2_t") / 2) * dot("sigma_2")
    return kappa, growth, [Z1n, Z2n], [S("alpha") - S("D1_t") - S("D2_t")]

_s3 = np.sqrt(3.0)
MODELS["AK"] = dict(
    build=spec_pack(["D1_t", "D2_t"], ["Z1_t", "Z2_t"], 3, _ak),
    defaults={"beta": float(np.exp(-0.0025)), "rho": 1.001, "gamma": 1.001,
              "alpha": 0.02305, "zeta": 32.0, "iota_k": 0.01, "nu_k": 0.01,
              "nu_1": 0.014, "nu_2": 0.0485, "mu_2": 6.3e-6,
              "sigma_k": _s3 * np.array([0.92, 0.40, 0.0]),
              "sigma_1": _s3 * np.array([0.0, 5.70, 0.0]),
              "sigma_2": _s3 * np.array([0.0, 0.0, 0.00031])},
    target={"gamma": 8.001}, n_states=2, n_shocks=3, seeds=None,
    source="book Section 11.7 / computation appendix",
    anchor_note="closed-form D2*=0.019023 at rho=1 (book); engine 0.019016 at rho=1.001")

# --- 2. Internal habit (book computation appendix) -------------------------
def _habit(S):
    dot = lambda b: sum(S(b + str(i + 1)) * S(f"W{i+1}_tp1") for i in range(3))
    nn = lambda b: sum(S(b + str(i + 1)) ** 2 for i in range(3))
    cap = S("epsilon") * (S("phi_1") * sp.log(1. + S("phi_2") * S("imk_t")) - S("alpha_k")
                          + S("beta_k") * S("Z_t")
                          - S("q_t") ** 2 * 0.5 * nn("sigma_k") * sp.exp(S("Y_t"))) \
        + sp.sqrt(S("epsilon")) * sp.exp(0.5 * S("Y_t")) * dot("sigma_k")
    Zn = S("Z_t") - S("epsilon") * S("beta_z") * S("Z_t") \
        + sp.sqrt(S("epsilon")) * sp.exp(0.5 * S("Y_t")) * dot("sigma_z")
    Yn = S("Y_t") - S("epsilon") * S("beta_2") * (1 - S("mu_2") * sp.exp(-S("Y_t"))) \
        - S("q_t") ** 2 * 0.5 * nn("sigma_y") * sp.exp(-S("Y_t")) * S("epsilon") \
        + sp.exp(-0.5 * S("Y_t")) * dot("sigma_y") * sp.sqrt(S("epsilon"))
    Xn = sp.log(sp.exp(-S("nu_h") + S("X_t")) + (1 - sp.exp(-S("nu_h"))) * S("imh_t")) - cap
    kappa = 1 / (1 - S("tau")) * sp.log((1 - S("llambda")) * S("imh_t") ** (1 - S("tau"))
                                        + S("llambda") * sp.exp((1 - S("tau")) * S("X_t")))
    return kappa, cap, [Zn, Yn, Xn], [S("a") - S("imh_t") - S("imk_t")]

_s12 = np.sqrt(12.0)
MODELS["HABIT"] = dict(
    build=spec_pack(["imh_t", "imk_t"], ["Z_t", "Y_t", "X_t"], 3, _habit, external=True),
    defaults={"sigma_k": _s12 * np.array([0.92, 0.40, 0.0]),
              "sigma_z": _s12 * np.array([0.0, 5.7, 0.0]),
              "sigma_y": _s12 * np.array([0.0, 0.0, 0.00031]),
              "epsilon": 0.25, "beta": float(np.exp(-0.0025)), "gamma": 1.001,
              "rho": 1.001, "a": 0.0922, "phi_1": 0.125, "phi_2": 8.0,
              "alpha_k": 0.04, "beta_k": 0.04, "beta_z": 0.056, "beta_2": 0.194,
              "mu_2": 6.3e-6, "nu_h": 0.025, "tau": 1.01, "llambda": -0.0},
    target={"gamma": 8.0, "llambda": 0.67, "tau": 0.01}, n_states=3, n_shocks=3,
    seeds=None, source="book computation appendix (external habit)",
    anchor_note="validated against the appendix notebook's stored solution")

# --- 3. Kaltenbrunner-Lochstoer (RFS 2010), LRR II -------------------------
def _kl(S):
    ik = S("is_t") * sp.exp((1 - S("alpha")) * S("w_t"))
    phi_i = S("a1") + S("a2") / (1 - 1 / S("xi")) * ik ** (1 - 1 / S("xi"))
    growth = sp.log(1 - S("delta") + phi_i)
    kappa = sp.log(S("cs_t")) + (1 - S("alpha")) * S("w_t")
    wn = S("w_t") + S("mu") - growth + S("sigma") * S("W1_tp1")
    return kappa, growth, [wn], [1 - S("cs_t") - S("is_t")]

_KL_MU, _KL_D, _KL_XI = 0.004, 0.021, 18.0
_KL_B = 0.998
MODELS["KL"] = dict(
    build=spec_pack(["cs_t", "is_t"], ["w_t"], 1, _kl),
    defaults={"beta": _KL_B, "rho": 1.001, "gamma": 1.001, "alpha": 0.36,
              "delta": _KL_D, "mu": _KL_MU, "xi": _KL_XI, "sigma": 0.0411,
              "a1": float((1 - _KL_D - np.exp(_KL_MU)) / (_KL_XI - 1)),
              "a2": float((np.exp(_KL_MU) - 1 + _KL_D) ** (1 / _KL_XI))},
    target={"gamma": 5.0, "rho": 1.0 / 1.5}, n_states=1, n_shocks=1,
    seeds={"w": float(np.log((np.exp(2 / 3 * _KL_MU) / _KL_B - (1 - _KL_D)) / 0.36) / (1 - 0.36))},
    source="Kaltenbrunner & Lochstoer (RFS 2010), Table 1 p.3197 + Table 4 p.3210 (LRR II)",
    anchor_note="I/K=e^mu-1+delta (fn.4); Euler: alpha*Y/K = e^(rho*mu)/beta-(1-delta)")

# --- 4. Ai-Croce-Li (RFS 2013), Extension 1 via BH-Dynare form -------------
def _acl(S):
    yk = sp.exp((1 - S("alpha")) * S("wa_t"))
    ik = S("is_t") * yk
    jk = S("js_t") * yk
    sk = sp.exp(S("s_t"))
    Gk = (S("nu") * ik ** (1 - 1 / S("eta"))
          + (1 - S("nu")) * sk ** (1 - 1 / S("eta"))) ** (1 / (1 - 1 / S("eta")))
    Hk = S("a1") / (1 - 1 / S("xi")) * jk ** (1 - 1 / S("xi")) + S("a2")
    varpi = sp.exp(-((1 - S("alpha")) / S("alpha")) * (S("x_t") + S("sig_a") * S("W1_tp1")))
    growth = sp.log(1 - S("dK") + varpi * Gk)
    kappa = sp.log(S("cs_t")) + (1 - S("alpha")) * S("wa_t")
    wan = S("wa_t") + S("mu") + S("x_t") + S("sig_a") * S("W1_tp1") - growth
    sn = sp.log((1 - S("dS")) * (sk - Gk) + Hk) - growth
    xn = S("rho_x") * S("x_t") + S("sig_x") * S("W2_tp1")
    return kappa, growth, [wan, sn, xn], [1 - S("cs_t") - S("is_t") - S("js_t")]

def _acl_chain():
    """BH-Dynare closed-form ss chain (frequency-generic), quarterly."""
    a, b_a, mu_a, lam_a = 0.3, 0.971, 0.02, 0.11
    eta, phw, xi = 2.5, 0.88, 5.0
    bq, muq = b_a ** 0.25, mu_a / 4
    lamq = 1 - (1 - lam_a) ** 0.25
    qb = np.log(bq) - 0.5 * muq
    ims = eta * np.log((np.exp(-qb) - 1 + lamq) * phw / (1 - phw))
    mms = 1 / (1 - 1 / eta) * np.log(phw * np.exp((1 - 1 / eta) * ims) + (1 - phw))
    jms = np.log(np.exp(muq) - (1 - np.exp(mms)) * (1 - lamq))
    mmi = 1 / (1 - 1 / eta) * np.log(phw + (1 - phw) * np.exp((1 - 1 / eta) * (-ims)))
    qk = np.log(1 - lamq + 1 / phw * np.exp(-(1 / eta) * mmi))
    kb = 1 / (a - 1) * np.log((np.exp(qk) * (1 / np.exp(qb) - 1 + lamq)) / a)
    mb = kb + np.log(np.exp(muq) - 1 + lamq)
    sb = mb - mms
    jb = jms + sb
    jkb = jb - kb
    return dict(bq=bq, muq=muq, lamq=lamq, kb=kb, sb=sb,
                a1=float(np.exp(jkb) ** (1 / xi)), a2=float(-np.exp(jkb) / (xi - 1)))

_A = _acl_chain()
_RXA = 0.925
MODELS["ACL"] = dict(
    build=spec_pack(["cs_t", "is_t", "js_t"], ["wa_t", "s_t", "x_t"], 2, _acl),
    defaults={"beta": _A["bq"], "rho": 0.5, "gamma": 1.001, "alpha": 0.3,
              "dK": _A["lamq"], "dS": _A["lamq"], "nu": 0.88, "eta": 2.5, "xi": 5.0,
              "mu": _A["muq"], "sig_a": 0.0508 / 2, "rho_x": _RXA ** 0.25,
              "sig_x": 0.008636 * float(np.sqrt((1 - _RXA ** 0.5) / (1 - _RXA ** 2))),
              "a1": _A["a1"], "a2": _A["a2"]},
    target={"gamma": 10.0}, n_states=3, n_shocks=2,
    seeds={"wa": float(-_A["kb"]), "s": float(_A["sb"] - _A["kb"])},
    source="Ai, Croce & Li (RFS 2013) Table C.2 Ext.1; BH-Dynare aicroceli_final.mod",
    anchor_note="all six ss quantities match the BH closed-form chain to ~1e-10")

# --- 5. Croce (2008 IGIER WP), fixed-labor benchmark -----------------------
def _croce(S):
    yk = sp.exp((1 - S("alpha")) * S("wa_t")) * S("nbar") ** (1 - S("alpha"))
    ik = S("is_t") * yk
    G = S("a1") / (1 - 1 / S("tau")) * ik ** (1 - 1 / S("tau")) + S("a2")
    growth = sp.log(1 - S("dk") + G)
    kappa = sp.log(S("cs_t")) + (1 - S("alpha")) * (S("wa_t") + sp.log(S("nbar")))
    wan = S("wa_t") + S("mu") + S("x_t") + S("sig_a") * S("W1_tp1") - growth
    xn = S("rho_x") * S("x_t") + S("sig_x") * S("W2_tp1")
    return kappa, growth, [wan, xn], [1 - S("cs_t") - S("is_t")]

_C_MU, _C_SIG, _C_RHO, _C_DK, _C_TAU = 3 * 0.00165, 0.006 * np.sqrt(3.0), 0.98 ** 3, 1 - 0.995 ** 3, 0.98
_C_B = 0.98 ** 0.25
_C_XB = float(np.exp(_C_MU) - 1 + _C_DK)
_C_YK = (np.exp(_C_MU / 2) / _C_B - 1 + _C_DK) / 0.33
MODELS["CROCE"] = dict(
    build=spec_pack(["cs_t", "is_t"], ["wa_t", "x_t"], 2, _croce),
    defaults={"beta": _C_B, "rho": 0.5, "gamma": 1.001, "alpha": 0.33,
              "dk": _C_DK, "tau": _C_TAU, "mu": _C_MU, "sig_a": _C_SIG,
              "rho_x": _C_RHO,
              "sig_x": 0.055 * 0.006 * float(np.sqrt((1 - 0.98 ** 6) / (1 - 0.98 ** 2))),
              "nbar": 0.18, "a1": float(_C_XB ** (1 / _C_TAU)),
              "a2": float(-_C_XB / (_C_TAU - 1))},
    target={"gamma": 30.0}, n_states=2, n_shocks=2,
    seeds={"wa": float(np.log((_C_YK / 0.18 ** 0.67) ** (1 / 0.67)))},
    source="Croce (2008 IGIER WP 260508) Table 3A p.38; fixed labor per fn.14 (n=0.18)",
    anchor_note="growth=mu; I/K=e^mu-1+dk; Euler alpha*Y/K = e^(mu/Psi)/beta-(1-dk)")

# --- 6. Tallarini (JME 2000), production economy, labor fixed at Nbar ------
def _tall(S):
    yk = sp.exp(S("a") * S("wx_t")) * S("nbar") ** S("a")
    growth = sp.log(1 - S("delta") + S("is_t") * yk)
    kappa = sp.log(S("cs_t")) + S("a") * S("wx_t") + S("a") * sp.log(S("nbar"))
    wxn = S("wx_t") + S("g") + S("sig") * S("W1_tp1") - growth
    return kappa, growth, [wxn], [1 - S("cs_t") - S("is_t")]

_T_G, _T_D, _T_B, _T_A = 0.004, 0.021, 0.9926, 0.661
_T_YK = (np.exp(_T_G) / _T_B - 1 + _T_D) / (1 - _T_A)
MODELS["TALLARINI"] = dict(
    build=spec_pack(["cs_t", "is_t"], ["wx_t"], 1, _tall),
    defaults={"beta": _T_B, "rho": 1.001, "gamma": 1.001, "a": _T_A,
              "delta": _T_D, "g": _T_G, "sig": 0.0115, "nbar": 0.2305},
    target={"gamma": 100.0}, n_states=1, n_shocks=1,
    seeds={"wx": float(np.log((_T_YK / 0.2305 ** _T_A) ** (1 / _T_A)))},
    source="Tallarini (JME 2000) Table 4 p.524; labor fixed at his Nbar=0.2305 "
           "(our restriction; preferences then = his consumption-only Eq.(1))",
    anchor_note="growth=g; I/K=e^g-1+delta; Euler (1-a)*Y/K = e^g/beta-(1-delta)")
