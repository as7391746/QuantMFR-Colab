"""All computation for the Figures 11.1-11.3 replication: the second-order
small-noise expansion of the Chapter-11 AK economy (Appendix "approach one",
order by order) and Borovicka-Hansen (2014) shock elasticities. Each step
cites the chapter/appendix equation it implements. This file is the
notebook's SOLVE cell (spliced verbatim by make_notebook.py).

Expansion conventions (chapter, section "Small-noise expansion"):
    X_t(q) ~ X^0 + q X^1_t + (q^2/2) X^2_t,   q = 1 is the model;
    gamma - 1 = (gamma_o - 1)/q  (risk aversion scales with 1/q), so the
    change of measure N^0 survives at first order with mean shift mu0.

Unknown jump objects, solved as functions of the state:
    v_t = Vhat_t - Ghat_t   (scaled log value)
    d_t = D2_t              (investment-capital ratio; D1 = alpha - D2)
Order-k representations:
    v^1_t = ups0 + ups1 . X1_t                     d^1_t = d0 + d1 . X1_t
    v^2_t = quad(X1_t) + phix2 . X2_t              d^2_t analogous
where X1 = (Z1^1, Z2^1) is the first-order state (AR(1): X1' = A X1 + S W')
and X2 = (Z1^2, Z2^2) is the second-order state.

Under the N^0 change of measure (Appendix, eq V-R2 discussion) the shock is
W' ~ N(mu0, I) with mu0 = (1 - gamma_o) * [shock loading of (v^1' + ghat^1')].

Pricing: every priced log increment (consumption growth, SDF, log(I/K)) is
exponential-linear-quadratic in (X1, X2, W'); Gaussian integrals keep the
class closed, so the elasticities

    exposure elasticity  eps(x0, t) = nu . E[M_t W_1 | x0] / E[M_t | x0]
    price elasticity     = exposure(G) - exposure(G S)

follow from one backward recursion per horizon, no simulation; eps is LINEAR
in x0, and the reported quantile-q curve is the q-quantile of eps under the
stationary distribution of X1 (the convention of the published runs).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from model import PARAMS, SCENARIOS, model


# =============================================================================
# Quadratic-form container:  f(X1, X2) = c + l.X1 + X1'Q X1 + lx2.X2
# (all conditional expectations in the expansion stay in this class)
# =============================================================================

class QF:
    def __init__(self, c=0.0, l=None, Q=None, lx2=None):
        self.c = float(c)
        self.l = np.zeros(2) if l is None else np.asarray(l, float)
        self.Q = np.zeros((2, 2)) if Q is None else np.asarray(Q, float)
        self.lx2 = np.zeros(2) if lx2 is None else np.asarray(lx2, float)

    def __add__(self, o):
        if isinstance(o, QF):
            return QF(self.c + o.c, self.l + o.l, self.Q + o.Q,
                      self.lx2 + o.lx2)
        return QF(self.c + o, self.l, self.Q, self.lx2)

    __radd__ = __add__

    def __mul__(self, s):
        return QF(self.c * s, self.l * s, self.Q * s, self.lx2 * s)

    __rmul__ = __mul__

    def coeffs(self):
        return np.concatenate([[self.c], self.l,
                               [self.Q[0, 0], self.Q[0, 1] + self.Q[1, 0],
                                self.Q[1, 1]], self.lx2])

    @staticmethod
    def from_coeffs(u):
        Q = np.array([[u[3], u[4] / 2], [u[4] / 2, u[5]]])
        return QF(u[0], u[1:3], Q, u[6:8])

    def __call__(self, x1, x2=(0.0, 0.0)):
        x1 = np.asarray(x1, float)
        return (self.c + self.l @ x1 + x1 @ self.Q @ x1
                + np.asarray(x2, float) @ self.lx2)


NC = 8  # number of QF coefficients


# =============================================================================
# Order 0 — deterministic steady state (chapter {eq}`equation4` at q = 0)
# =============================================================================

def order0(p):
    """Solve (D2^0, v^0); check the rho=1 closed form D2* from the chapter."""
    beta, rho, alpha, zeta, iota_k = (p["beta"], p["rho"], p["alpha"],
                                      p["zeta"], p["iota_k"])

    def value(d2):
        g0 = np.log1p(zeta * d2) / zeta - iota_k          # capital growth
        c0 = np.log(alpha - d2)                           # Chat - Ghat
        if abs(rho - 1.0) < 1e-12:
            # rho = 1 limit of the CES aggregator
            v0 = c0 + beta * g0 / (1 - beta)
        else:
            # solve v = (1/(1-rho)) log[(1-b)e^{(1-r)c} + b e^{(1-r)(v+g)}]
            # => e^{(1-r)v} (1 - b e^{(1-r)g}) = (1-b) e^{(1-r)c}
            lam_g = beta * np.exp((1 - rho) * g0)
            v0 = (np.log((1 - beta) * np.exp((1 - rho) * c0) / (1 - lam_g))
                  / (1 - rho))
        return v0, g0, c0

    def foc(d2):
        v0, g0, c0 = value(d2)
        r0 = v0 + g0                                       # Rhat^0 - Ghat^0
        return (np.log(1 - beta) + (1 - rho) * c0 - c0
                - np.log(beta) - (1 - rho) * r0 + np.log1p(zeta * d2))

    # feasible domain: transversality beta e^{(1-rho) g0(d2)} < 1 bounds d2
    # from below (rho > 1) or above (rho < 1); g0 is increasing in d2.
    lo, hi = 1e-8, alpha - 1e-8
    if abs(rho - 1.0) > 1e-12:
        g_crit = -np.log(beta) / (1 - rho)   # g0 at which lam_g = 1
        d_crit = (np.exp(zeta * (g_crit + iota_k)) - 1.0) / zeta
        if rho > 1:
            lo = max(lo, d_crit * (1 + 1e-10) + 1e-12)
        else:
            hi = min(hi, d_crit * (1 - 1e-10) - 1e-12)
    d2 = brentq(foc, lo, hi, xtol=1e-15)
    v0, g0, c0 = value(d2)
    lam = beta * np.exp((1 - rho) * g0)   # chapter's lambda = beta e^{(1-rho)eta_c^0}
    return {"d2": d2, "v0": v0, "g0": g0, "c0": c0, "lam": lam,
            "kC": 1.0 / (alpha - d2),       # d(-log D1)/dD2
            "kD": 1.0 / (1.0 + zeta * d2)}  # d[(1/z)log(1+z D2)]/dD2


# =============================================================================
# Order 1 — linear system (Appendix eqs around `first_recursive_update`)
# =============================================================================

def order1(p, o0):
    rho, gam = p["rho"], p["gamma"]
    lam, kC, kD = o0["lam"], o0["kC"], o0["kD"]
    A = np.diag([p["a1"], p["a2"]])
    S = np.vstack([p["s1"], p["s2"]])      # (2,3): X1' = A X1 + S W'
    nu_k, sk = p["nu_k"], p["sk"]
    e1 = np.array([1.0, 0.0])

    # theta: FOC slope  (rho kC + zeta kD) d^1 = (1-rho)(Rhat^1 - Ghat^1)
    theta = (1 - rho) / (rho * kC + p["zeta"] * kD)

    # coefficient equations (first q-derivative of the FOC and of the value
    # recursion, matching X1 coefficients):
    #   d1 = theta (A' ups1 + nu_k e1 + kD d1)
    #   ups1 = -(1-lam) kC d1 + lam (A' ups1 + nu_k e1 + kD d1)
    # Solve the joint 4x4 linear system in (d1, ups1).
    M = np.zeros((4, 4))
    b = np.zeros(4)
    # rows 0-1: d1 - theta kD d1 - theta A' ups1 = theta nu_k e1
    M[0:2, 0:2] = np.eye(2) * (1 - theta * kD)
    M[0:2, 2:4] = -theta * A.T
    b[0:2] = theta * nu_k * e1
    # rows 2-3: ups1 + (1-lam) kC d1 - lam kD d1 - lam A' ups1 = lam nu_k e1
    M[2:4, 0:2] = ((1 - lam) * kC - lam * kD) * np.eye(2)
    M[2:4, 2:4] = np.eye(2) - lam * A.T
    b[2:4] = lam * nu_k * e1
    sol = np.linalg.solve(M, b)
    d1, ups1 = sol[0:2], sol[2:4]

    # shock loading of (v^1' + ghat^1') and the N^0 mean shift (chapter mu^0)
    Sig_w = S.T @ ups1 + sk                # (3,)
    mu0 = (1 - gam) * Sig_w

    # constants: r1 = E_t[v1' + g1'] + ((1-gamma_o)/2)|Sig_w|^2 with
    # E_t under the PHYSICAL measure (first-order objects, W mean 0):
    #   r1_t = ups0 + kD d0 + vol_adj + (A'ups1 + kD d1 + nu_k e1).X1_t
    #   d0   = theta (ups0 + kD d0 + vol_adj)
    #   ups0 = -(1-lam) kC d0 + lam (ups0 + kD d0 + vol_adj)
    vol_adj = 0.5 * (1 - gam) * float(Sig_w @ Sig_w)
    M2 = np.array([[1 - theta * kD, -theta],
                   [(1 - lam) * kC - lam * kD, 1 - lam]])
    b2 = np.array([theta * vol_adj, lam * vol_adj])
    d0, ups0 = np.linalg.solve(M2, b2)

    r1_l = A.T @ ups1 + kD * d1 + nu_k * e1      # X1 loading of Rhat^1-Ghat^1
    r1_c = ups0 + kD * d0 + vol_adj
    return {"d0": d0, "d1": d1, "ups0": ups0, "ups1": ups1,
            "Sig_w": Sig_w, "mu0": mu0, "A": A, "S": S,
            "r1_c": r1_c, "r1_l": r1_l}


# =============================================================================
# Order 2 — linear coefficient system solved by probing (Appendix B order two)
# =============================================================================

def _E_next(qf, A, S, mu0, x2_law):
    """E^{N0}_t of a QF evaluated at (X1_{t+1}, X2_{t+1}) plus optional
    'bilinear' terms handled by the caller. X1' = A X1 + S W', W' ~ N(mu0, I);
    X2' = A X2 + x2_law(X1, W') with x2_law supplied as its own QF pieces."""
    # E[c + l.X1' + X1''Q X1' + lx2.X2']
    out = QF()
    Sm = S @ mu0
    # linear part
    out.c += qf.c + qf.l @ Sm
    out.l += A.T @ qf.l
    # quadratic part: E[(AX1+SW)'Q(AX1+SW)]
    out.Q += A.T @ qf.Q @ A
    out.l += A.T @ (qf.Q + qf.Q.T) @ Sm
    out.c += Sm @ qf.Q @ Sm + np.trace(S.T @ qf.Q @ S)
    # X2 part: E[lx2 . X2'] = lx2 . (A X2 + E x2_law)
    out.lx2 += A.T @ qf.lx2
    ex2 = x2_law(qf.lx2)          # returns QF in (X1_t, const)
    return out + ex2


def make_x2_law(p, mu0):
    """E^{N0}_t[ lx2 . (X2' - A X2) ] as a QF in X1_t.

    Second-order state dynamics (from chapter {eq}`equation2`, exact Taylor
    coefficients around Z1=0, exp(Z2)=mu_2; NOTE the chapter's combined
    second-order display adds a -mu_2|sigma_1|^2/2 constant to Z1 that is NOT
    in {eq}`equation2` nor in HKT eq. (15) -- we follow the primitive
    equations and flag the display in the README):
      Z1^2' = a1 Z1^2 + sqrt(mu_2) Z2^1 sigma_1 . W'
      Z2^2' = a2 Z2^2 + nu_2 (Z2^1)^2 - |sigma_2|^2/mu_2
              - (1/sqrt(mu_2)) Z2^1 sigma_2 . W'
    """
    s1m = np.sqrt(p["mu_2"]) * p["sigma_1"] @ mu0
    s2m = (p["sigma_2"] / np.sqrt(p["mu_2"])) @ mu0
    sig2_sq = float(p["sigma_2"] @ p["sigma_2"])

    def law(lx2):
        out = QF()
        # E[Z1^2' - a1 Z1^2] = sqrt(mu2) Z2^1 (sigma_1 . mu0)  -> linear in X1
        out.l += lx2[0] * s1m * np.array([0.0, 1.0])
        # E[Z2^2' - a2 Z2^2] = nu2 (Z2^1)^2 - |s2|^2/mu2 - Z2^1 s2m
        out.Q += lx2[1] * p["nu_2"] * np.diag([0.0, 1.0])
        out.c += lx2[1] * (-sig2_sq / p["mu_2"])
        out.l += lx2[1] * (-s2m) * np.array([0.0, 1.0])
        return out

    return law


def order2(p, o0, o1):
    rho = p["rho"]
    lam, kC, kD, zeta = o0["lam"], o0["kC"], o0["kD"], p["zeta"]
    A, S, mu0 = o1["A"], o1["S"], o1["mu0"]
    d1, ups1 = o1["d1"], o1["ups1"]
    nu_k, sk, e1 = p["nu_k"], p["sk"], np.array([1.0, 0.0])
    theta = (1 - rho) / (rho * kC + zeta * kD)
    x2_law = make_x2_law(p, mu0)
    sk_z = np.sqrt(p["mu_2"]) * p["sigma_k"] @ mu0  # E[Z2^1 sigma_k.W'] factor

    # (r1 - c1)^2 as a QF (needed by the value equation's quadratic bonus):
    # r1_t - c1_t where c1 = -kC d^1
    rc_l = o1["r1_l"] + kC * d1
    rc_c = o1["r1_c"] + kC * o1["d0"]

    def qf_square(c, l):
        return QF(c * c, 2 * c * l, np.outer(l, l))

    rc2 = qf_square(rc_c, rc_l)

    def d1sq_qf():
        return qf_square(o1["d0"], d1)

    def solve_given(vguess):
        """One evaluation of the order-2 system: given v2 (QF), return the
        implied v2 from the appendix recursion; also return d2 (QF)."""
        # ghat^2_{t+1} pieces evaluated at (X1_t, X2_t, W'):
        #   kD d^2_t - zeta kD^2 (d^1_t)^2 + nu_k Z1^2_t - |sigma_k|^2 mu_2
        #   + sqrt(mu_2) Z2^1_t sigma_k . W'
        # R^2_t - G^2_t = E^{N0}_t[v2(X1',X2') + ghat^2'].
        Ev_next = _E_next(vguess, A, S, mu0, x2_law)
        r2 = Ev_next + QF(
            c=-float(p["sigma_k"] @ p["sigma_k"]) * p["mu_2"],
            l=np.array([0.0, sk_z]),           # E[sqrt(mu2) Z2^1 sk.W']
            lx2=nu_k * e1)                     # nu_k Z1^2_t
        # ghat^2 also contains kD d^2_t and the -(zeta kD^2)(d^1)^2 term,
        # both dated t -> enter r2 directly:
        # r2 += kD d2 - zeta kD^2 (d1)^2 ; d2 itself depends on r2 (FOC).
        # FOC order 2:
        #   (rho kC + zeta kD) d2 = (1-rho) r2_full
        #        - rho kC^2 (d1)^2 + zeta^2 kD^2 (d1)^2
        # where r2_full = r2 + kD d2 - zeta kD^2 (d1)^2. Substitute:
        #   d2 [(rho kC + zeta kD) - (1-rho) kD] =
        #      (1-rho)(r2 - zeta kD^2 (d1)^2) + (zeta^2 kD^2 - rho kC^2)(d1)^2
        denom = (rho * kC + zeta * kD) - (1 - rho) * kD
        d1sq = d1sq_qf()
        d2 = ((1 - rho) * (r2 + (-zeta * kD ** 2) * d1sq)
              + (zeta ** 2 * kD ** 2 - rho * kC ** 2) * d1sq) * (1.0 / denom)
        r2_full = r2 + kD * d2 + (-zeta * kD ** 2) * d1sq
        # c^2_t = -kC d^2 - kC^2 (d^1)^2
        c2 = (-kC) * d2 + (-kC ** 2) * d1sq
        # value second order (exact 2nd derivative of the CES log-sum-exp):
        #   v2 = (1-lam) c2 + lam r2_full + (1-rho) lam (1-lam) (r1 - c1)^2
        v2_new = ((1 - lam) * c2 + lam * r2_full
                  + (1 - rho) * lam * (1 - lam) * rc2)
        return v2_new, d2, r2_full

    # v2 -> v2_new is affine in v2's coefficients: solve (I - M)u = b
    zero = QF()
    b_qf, _, _ = solve_given(zero)
    b = b_qf.coeffs()
    Mmat = np.zeros((NC, NC))
    for j in range(NC):
        u = np.zeros(NC)
        u[j] = 1.0
        out, _, _ = solve_given(QF.from_coeffs(u))
        Mmat[:, j] = out.coeffs() - b
    u = np.linalg.solve(np.eye(NC) - Mmat, b)
    v2 = QF.from_coeffs(u)
    _, d2, r2_full = solve_given(v2)
    return {"v2": v2, "d2": d2, "r2": r2_full}


# =============================================================================
# LQ exponent class:  Y' = c + x.X1 + X1'xx X1 + x2.X2 + w.W' + W''ww W' + X1'xw W'
# (every priced log increment lives here; Gaussian integrals keep it closed)
# =============================================================================

class LQ:
    __slots__ = ("c", "x", "xx", "x2", "w", "ww", "xw")

    def __init__(self, c=0.0, x=None, xx=None, x2=None, w=None, ww=None,
                 xw=None, nx=2, nw=3):
        self.c = float(c)
        self.x = np.zeros(nx) if x is None else np.asarray(x, float)
        self.xx = np.zeros((nx, nx)) if xx is None else np.asarray(xx, float)
        self.x2 = np.zeros(nx) if x2 is None else np.asarray(x2, float)
        self.w = np.zeros(nw) if w is None else np.asarray(w, float)
        self.ww = np.zeros((nw, nw)) if ww is None else np.asarray(ww, float)
        self.xw = np.zeros((nx, nw)) if xw is None else np.asarray(xw, float)

    def __add__(self, o):
        if isinstance(o, LQ):
            return LQ(self.c + o.c, self.x + o.x, self.xx + o.xx,
                      self.x2 + o.x2, self.w + o.w, self.ww + o.ww,
                      self.xw + o.xw)
        return LQ(self.c + o, self.x, self.xx, self.x2, self.w, self.ww,
                  self.xw)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-1.0) * o

    def __mul__(self, s):
        return LQ(self.c * s, self.x * s, self.xx * s, self.x2 * s,
                  self.w * s, self.ww * s, self.xw * s)

    __rmul__ = __mul__


class Dynamics:
    """X1' = A X1 + S W';   X2' = A X2 + g_c + X1'g_xx X1 + X1'g_xw[i] W'."""

    def __init__(self, A, S, g_c, g_xx, g_xw):
        self.A, self.S = A, S
        self.g_c = g_c        # (2,)
        self.g_xx = g_xx      # (2,2,2): g_xx[i] quadratic form for X2'_i
        self.g_xw = g_xw      # (2,2,3): g_xw[i] bilinear X1 (x) W for X2'_i


def compose(F, dyn):
    """F(X1_{t+1}, X2_{t+1}) -> LQ in (X1_t, X2_t, W_{t+1}).

    F may itself carry (w, ww, xw) = 0 pieces only (it is a time-t+1 state
    function); X2-quadratic terms never arise (increments are linear in X2).
    """
    A, S = dyn.A, dyn.S
    out = LQ(F.c)
    # linear state part: F.x . (A X1 + S W)
    out.x = A.T @ F.x
    out.w = S.T @ F.x
    # quadratic: (A X1 + S W)' F.xx (A X1 + S W)
    out.xx = A.T @ F.xx @ A
    out.ww = S.T @ F.xx @ S
    out.xw = A.T @ (F.xx + F.xx.T) @ S
    # X2 part: F.x2 . X2' = F.x2 . (A X2 + g_c + quad + bilinear)
    out.x2 = A.T @ F.x2
    out.c += F.x2 @ dyn.g_c
    for i in range(2):
        out.xx += F.x2[i] * dyn.g_xx[i]
        out.xw += F.x2[i] * dyn.g_xw[i]
    return out


def integrate_W(F):
    """log E[exp(F)|X1, X2] over W ~ N(0, I): stays LQ (no W pieces).

    E[exp(b.W + W'C W)] = det(I-2C)^{-1/2} exp( b'(I-2C)^{-1} b / 2 ),
    here b = F.w + F.xw'X1  (state-dependent).
    """
    nw = F.w.shape[0]
    Iw = np.eye(nw)
    Minv = np.linalg.inv(Iw - 2.0 * F.ww)
    logdet = -0.5 * np.linalg.slogdet(Iw - 2.0 * F.ww)[1]
    out = LQ(F.c + logdet + 0.5 * F.w @ Minv @ F.w,
             F.x + F.xw @ Minv @ F.w,
             F.xx + 0.5 * F.xw @ Minv @ F.xw.T,
             F.x2)
    return out


# =============================================================================
# Assemble the priced log increments from the expansion solution
# =============================================================================

def _qf_to_lq(qf):
    return LQ(qf.c, qf.l, qf.Q, qf.lx2)


def _level_square(c, l):
    """(c + l.X1)^2 as an LQ."""
    return LQ(c * c, 2 * c * l, np.outer(l, l))


def elastic_build(p, o0, o1, o2):
    """Return dict with dynamics and the LQ increments {gc, sdf, invk}."""
    kC, kD, zeta = o0["kC"], o0["kD"], p["zeta"]
    rho, beta, gam = p["rho"], p["beta"], p["gamma"]
    A, S = o1["A"], o1["S"]
    d0, d1 = o1["d0"], o1["d1"]
    D20 = o0["d2"]
    nu_k, sk = p["nu_k"], p["sk"]
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    mu2 = p["mu_2"]
    sig_k, sig_1, sig_2 = p["sigma_k"], p["sigma_1"], p["sigma_2"]

    # ---- second-order state dynamics (exact Taylor of {eq}`equation2`) ----
    g_c = np.array([0.0, -float(sig_2 @ sig_2) / mu2])
    g_xx = np.zeros((2, 2, 2))
    g_xx[1][1, 1] = p["nu_2"]                        # nu_2 (Z2^1)^2
    g_xw = np.zeros((2, 2, 3))
    g_xw[0][1] = np.sqrt(mu2) * sig_1                #  sqrt(mu2) Z2^1 s1.W
    g_xw[1][1] = -sig_2 / np.sqrt(mu2)               # -(1/sqrt(mu2)) Z2^1 s2.W
    dyn = Dynamics(A, S, g_c, g_xx, g_xw)

    d2lq = _qf_to_lq(o2["d2"])
    d1sq = _level_square(d0, d1)

    # ---- capital log growth at q = 1 --------------------------------------
    ghat = LQ(o0["g0"] + kD * d0, kD * d1 + nu_k * e1, w=sk)
    ghat = ghat + 0.5 * (kD * d2lq + (-zeta * kD ** 2) * d1sq
                         + LQ(-float(sig_k @ sig_k) * mu2, x2=nu_k * e1))
    ghat.xw += 0.5 * np.outer(e2, np.sqrt(mu2) * sig_k)   # (Z2^1/2) sk.W

    # ---- consumption-capital ratio level and its increment -----------------
    clevel = (LQ(o0["c0"] - kC * d0, -kC * d1)
              + 0.5 * ((-kC) * d2lq + (-kC ** 2) * d1sq))
    gc = compose(clevel, dyn) - clevel + ghat            # log C' - log C

    # ---- log(I/K) level and increment (second-order Taylor of log D2) -----
    ilevel = (LQ(np.log(D20) + d0 / D20, d1 / D20)
              + 0.5 * ((1.0 / D20) * d2lq + (-1.0 / D20 ** 2) * d1sq))
    invk = compose(ilevel, dyn) - ilevel                 # log(I/K)' - log(I/K)

    # ---- vmr1, vmr2 (V-hat minus R-hat pieces) ------------------------------
    v1_next = compose(LQ(o1["ups0"], o1["ups1"]), dyn)   # v^1_{t+1}
    g1_next = LQ(kD * d0, kD * d1 + nu_k * e1, w=sk)     # ghat^1_{t+1}
    r1 = LQ(o1["r1_c"], o1["r1_l"])
    vmr1 = v1_next + g1_next - r1

    v2_next = compose(_qf_to_lq(o2["v2"]), dyn)
    g2_next = (kD * d2lq + (-zeta * kD ** 2) * d1sq
               + LQ(-float(sig_k @ sig_k) * mu2, x2=nu_k * e1))
    g2_next.xw += np.outer(e2, np.sqrt(mu2) * sig_k)     # sqrt(mu2) Z2^1 sk.W
    vmr2 = v2_next + g2_next - _qf_to_lq(o2["r2"])

    # ---- SDF increment (matches the published runs' construction:
    #      full quadratic N-tilde, normalized to a martingale increment) ----
    vmr = vmr1 + 0.5 * vmr2
    n_tilde = (1 - gam) * vmr - integrate_W((1 - gam) * vmr)
    sdf = (LQ(np.log(beta)) + (-rho) * gc + (rho - 1) * vmr + n_tilde)

    return {"dyn": dyn, "gc": gc, "sdf": sdf, "invk": invk}


# =============================================================================
# Elasticities — Borovicka-Hansen (2014), quantile convention of the runs
# =============================================================================

def _stationary_cov(dyn):
    A, S = dyn.A, dyn.S
    n = A.shape[0]
    vec = np.linalg.solve(np.eye(n * n) - np.kron(A, A),
                          (S @ S.T).reshape(-1))
    return vec.reshape(n, n)


def _elasticity_curve(Y, dyn, T, shock, percentile):
    """One backward pass; returns the T-vector of quantile-q elasticities."""
    nu = np.zeros(3)
    nu[shock] = 1.0
    p = norm.ppf(percentile)
    cov = _stationary_cov(dyn)
    out = np.empty(T)
    Phi = LQ()
    for t in range(T):
        pre = Y + (compose(Phi, dyn) if t > 0 else LQ())
        Minv = np.linalg.inv(np.eye(3) - 2.0 * pre.ww)
        const = nu @ Minv @ pre.w                       # eps at X1_0 = 0
        Arow = nu @ Minv @ pre.xw.T                     # d eps / d X1_0
        out[t] = const + p * np.sqrt(Arow @ cov @ Arow) if percentile != 0.5 \
            else const
        Phi = integrate_W(pre)
    return out


def exposure_elasticity(inc, T, shock, percentile=0.5, process="gc"):
    return _elasticity_curve(inc[process], inc["dyn"], T, shock, percentile)


def price_elasticity(inc, T, shock, percentile=0.5, process="gc"):
    """exposure(G) - exposure(G S), with the quantile applied to the joint
    linear-in-state coefficient (the published runs' convention)."""
    nu = np.zeros(3)
    nu[shock] = 1.0
    p = norm.ppf(percentile)
    dyn = inc["dyn"]
    cov = _stationary_cov(dyn)
    G, GS = inc[process], inc[process] + inc["sdf"]
    out = np.empty(T)
    PhiG, PhiGS = LQ(), LQ()
    for t in range(T):
        preG = G + (compose(PhiG, dyn) if t > 0 else LQ())
        preGS = GS + (compose(PhiGS, dyn) if t > 0 else LQ())
        MG = np.linalg.inv(np.eye(3) - 2.0 * preG.ww)
        MGS = np.linalg.inv(np.eye(3) - 2.0 * preGS.ww)
        const = nu @ (MG @ preG.w - MGS @ preGS.w)
        Arow = nu @ (MG @ preG.xw.T - MGS @ preGS.xw.T)
        out[t] = const + p * np.sqrt(Arow @ cov @ Arow) if percentile != 0.5 \
            else const
        PhiG, PhiGS = integrate_W(preG), integrate_W(preGS)
    return out
