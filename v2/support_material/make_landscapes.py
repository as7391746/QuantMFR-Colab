"""Numerical geometry of all six economies — one uniform construction.

For each model M a RECONSTRUCTION MAP R_M takes two axis coordinates
(z1, z2) — the model's fragile directions — and rebuilds every other
coordinate from the model's own equations:

  - the remaining controls from the static constraint,
  - self-pinning states at their own deterministic fixed points,
  - the growth rate and kappa from the model's closed forms,
  - the value ratio vmk from the deterministic value recursion
    (closed form; refined by a scalar bracket solve where the
    recursion is not the textbook one, e.g. the habit model),
  - the multipliers by ONE LINEAR least-squares step (the first-order
    -condition block is linear in them).

No joint nonlinear root solve is performed at any grid point.  The plotted
surface is

    L_M(z1, z2) = log10( ||F_M(R_M(z1, z2))||_inf )

with F_M the engine's complete deterministic steady-state system.  Blank
cells are OUTSIDE THE ADMISSIBLE RECONSTRUCTION DOMAIN (negative
consumption, or the deterministic value recursion undefined because the
transversality margin is negative) — a property of the candidate slice,
not a statement that the model has no solution.

The floor contour carries a sparse INITIALIZATION GRID: at every k-th
admissible cell the actual engine solve is started from R_M(z1, z2) and
classified — converged to the verified steady state / converged to a
different root / no convergence within the probe budget.  That overlay,
not the surface, is what justifies the word "basin".

Usage:  python make_landscapes.py KL          (one model)
        python make_landscapes.py overview    (2x3 panel from saved npz)
        python make_landscapes.py all         (everything, sequentially)
"""
import sys, os, re, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, V2)
import numpy as np, sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.optimize import brentq
import uncertain_expansion_faisal_feb26 as robust
from uncertain_expansion_faisal_feb26 import compile_equations
from autosolve import derive_guess, autosolve, _solve_checked, _ss_names
from models_sourced import MODELS

ZMIN, ZMAX = -13.0, 2.0          # one residual color scale for all six
NORM = mcolors.Normalize(vmin=ZMIN, vmax=ZMAX)
CMAP = "viridis_r"


# ---------------------------------------------------------------- engine F
def compile_F(name):
    """Lambdified complete deterministic ss system, preprocessing identical
    to the engine's generate_ss_function (tp1 collapsed, growth variable
    substituted, q and shocks zeroed)."""
    M = MODELS[name]
    T = dict(M["defaults"]); T.update(M["target"])
    spec = M["build"](T)
    nS, nW = M["n_states"], M["n_shocks"]
    n_J = spec["n_controls"] + (nS + 1) + 2
    eqs, variables, vt, H, L = compile_equations(
        parameter_names=spec["parameter_names"], variables=spec["variables"],
        variables_tp1=spec["variables_tp1"],
        control_variables=spec["control_variables"],
        state_variables=list(spec["state_variables"]),
        output_constraint=spec["kappa"], capital_growth=spec["growth"],
        state_equations=spec["state_equations"],
        static_constraints=spec["static_constraints"],
        var_shape=[n_J, nS + 1, nW],
        ExternalHabit=spec.get("external_habit", False))
    subs = robust.automate_step_1(variables)
    nG = variables.index(sp.Symbol("log_gk_t"))
    nQ = variables.index(sp.Symbol("q_t"))
    subs[variables[0]] = variables[nG]; subs[vt[0]] = variables[nG]
    subs[variables[nQ]] = 0.; subs[vt[nQ]] = 0.
    for w in variables[nQ + 1:nQ + nW + 1]: subs[w] = 0.
    for w in vt[nQ + 1:nQ + nW + 1]: subs[w] = 0.
    eqs = [e.subs(subs) for e in eqs]
    var_solve = variables[1:nQ]
    pd = {v: val for v, val in zip(spec["parameter_names"], spec["args"])}
    fl = sp.lambdify(var_solve, [sp.sympify(e).subs(pd) for e in eqs], "numpy")
    names = [str(v) for v in var_solve]

    def Fraw(x):
        try:
            v = np.array(fl(*x), dtype=complex).flatten()
            return np.where(np.abs(v.imag) > 1e-10, np.nan, v.real)
        except Exception:
            return np.full(len(eqs), np.nan)

    iv = {n: i for i, n in enumerate(names)}
    IM = [i for i, n in enumerate(names) if re.fullmatch(r"m(s|g|\d+)_t", n)]
    return dict(M=M, T=T, spec=spec, nS=nS, nW=nW, Fraw=Fraw,
                names=names, iv=iv, IM=IM, n=len(names))


def _finish(ctx, x):
    """vmk refinement (only if the closed form leaves the recursion row
    nonzero, e.g. external habit) + linear multiplier least squares."""
    Fr, iv, IM = ctx["Fraw"], ctx["iv"], ctx["IM"]
    jv = iv["vmk_t"]
    F0 = Fr(x)
    if not np.isfinite(F0[0]):
        return None, np.nan
    if abs(F0[0]) > 1e-9:                      # recursion row not closed-form
        def f0(v):
            xx = x.copy(); xx[jv] = v
            r = Fr(xx)[0]
            return r if np.isfinite(r) else np.nan
        done = False
        for span in (0.5, 1.0, 2.0, 4.0, 8.0):
            a, b = x[jv] - span, x[jv] + span
            fa, fb = f0(a), f0(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                x[jv] = brentq(f0, a, b, xtol=1e-13); done = True; break
        if not done and abs(F0[0]) > 1e-6:
            return None, np.nan
        F0 = Fr(x)
    if np.any(~np.isfinite(F0)):
        return None, np.nan
    cols = []
    for j in IM:                               # FOC block linear in multipliers
        xj = x.copy(); xj[j] = 1.0
        cols.append(Fr(xj) - F0)
    A = np.column_stack(cols)
    mstar, *_ = np.linalg.lstsq(A, -F0, rcond=None)
    x[IM] = mstar
    F = Fr(x)
    if np.any(~np.isfinite(F)):
        return None, np.nan
    return x, float(np.max(np.abs(F)))


def _vmk_closed(T, kap, g):
    be, rho = T["beta"], T["rho"]
    lam = be * np.exp((1 - rho) * g)
    if not np.isfinite(lam) or lam >= 1:
        return None                            # transversality violated
    return np.log((1 - be) * np.exp((1 - rho) * kap) / (1 - lam)) / (1 - rho)


# ------------------------------------------------- per-model reconstructions
def recon_KL(ctx, w, is_):
    T, iv = ctx["T"], ctx["iv"]
    cs = 1.0 - is_
    if cs <= 0 or is_ <= 0: return None, np.nan
    ik = is_ * np.exp((1 - T["alpha"]) * w)
    phi = T["a1"] + T["a2"] / (1 - 1 / T["xi"]) * ik ** (1 - 1 / T["xi"])
    g = np.log(1 - T["delta"] + phi)
    if not np.isfinite(g): return None, np.nan
    kap = np.log(cs) + (1 - T["alpha"]) * w
    vmk = _vmk_closed(T, kap, g)
    if vmk is None: return None, np.nan
    x = np.zeros(ctx["n"])
    x[iv["vmk_t"]], x[iv["log_cmk_t"]] = vmk, kap
    x[iv["cs_t"]], x[iv["is_t"]] = cs, is_
    x[iv["log_gk_t"]], x[iv["w_t"]] = g, w
    return _finish(ctx, x)


def recon_CROCE(ctx, wa, is_):
    T, iv = ctx["T"], ctx["iv"]
    cs = 1.0 - is_
    if cs <= 0 or is_ <= 0: return None, np.nan
    yk = np.exp((1 - T["alpha"]) * wa) * T["nbar"] ** (1 - T["alpha"])
    ik = is_ * yk
    G = T["a1"] / (1 - 1 / T["tau"]) * ik ** (1 - 1 / T["tau"]) + T["a2"]
    g = np.log(1 - T["dk"] + G)
    if not np.isfinite(g): return None, np.nan
    kap = np.log(cs) + (1 - T["alpha"]) * (wa + np.log(T["nbar"]))
    vmk = _vmk_closed(T, kap, g)
    if vmk is None: return None, np.nan
    x = np.zeros(ctx["n"])
    x[iv["vmk_t"]], x[iv["log_cmk_t"]] = vmk, kap
    x[iv["cs_t"]], x[iv["is_t"]] = cs, is_
    x[iv["log_gk_t"]], x[iv["wa_t"]] = g, wa
    x[iv["x_t"]] = 0.0                         # AR(1) state at its own fixed point
    return _finish(ctx, x)


def recon_TALL(ctx, wx, is_):
    T, iv = ctx["T"], ctx["iv"]
    cs = 1.0 - is_
    if cs <= 0 or is_ <= 0: return None, np.nan
    yk = np.exp(T["a"] * wx) * T["nbar"] ** T["a"]
    g = np.log(1 - T["delta"] + is_ * yk)
    if not np.isfinite(g): return None, np.nan
    kap = np.log(cs) + T["a"] * (wx + np.log(T["nbar"]))
    vmk = _vmk_closed(T, kap, g)
    if vmk is None: return None, np.nan
    x = np.zeros(ctx["n"])
    x[iv["vmk_t"]], x[iv["log_cmk_t"]] = vmk, kap
    x[iv["cs_t"]], x[iv["is_t"]] = cs, is_
    x[iv["log_gk_t"]], x[iv["wx_t"]] = g, wx
    return _finish(ctx, x)


def recon_AK(ctx, z2, d2):
    T, iv = ctx["T"], ctx["iv"]
    d1 = T["alpha"] - d2                       # static constraint
    if d1 <= 0 or d2 <= 0: return None, np.nan
    g = (1 / T["zeta"]) * np.log(1 + T["zeta"] * d2) - T["iota_k"]  # Z1 = 0
    kap = np.log(d1)
    vmk = _vmk_closed(T, kap, g)
    if vmk is None: return None, np.nan
    x = np.zeros(ctx["n"])
    x[iv["vmk_t"]], x[iv["log_cmk_t"]] = vmk, kap
    x[iv["D1_t"]], x[iv["D2_t"]] = d1, d2
    x[iv["log_gk_t"]] = g
    x[iv["Z1_t"]], x[iv["Z2_t"]] = 0.0, z2     # Z1 at its own fixed point
    return _finish(ctx, x)


def recon_HABIT(ctx, X, imk):
    T, iv = ctx["T"], ctx["iv"]
    imh = T["a"] - imk                         # static constraint
    if imh <= 0 or imk <= 0: return None, np.nan
    eps = T["epsilon"]
    g = eps * (T["phi_1"] * np.log(1 + T["phi_2"] * imk) - T["alpha_k"])  # Z = 0
    bundle = (1 - T["llambda"]) * imh ** (1 - T["tau"]) \
        + T["llambda"] * np.exp((1 - T["tau"]) * X)
    if bundle <= 0: return None, np.nan
    kap = np.log(bundle) / (1 - T["tau"])
    vmk = _vmk_closed(T, kap, g)
    if vmk is None: return None, np.nan
    x = np.zeros(ctx["n"])
    x[iv["vmk_t"]], x[iv["log_cmk_t"]] = vmk, kap
    x[iv["imh_t"]], x[iv["imk_t"]] = imh, imk
    x[iv["log_gk_t"]], x[iv["X_t"]] = g, X
    x[iv["Z_t"]] = 0.0                         # own fixed point
    x[iv["Y_t"]] = np.log(T["mu_2"])           # own fixed point
    return _finish(ctx, x)


def recon_ACL(ctx, wa, is_):
    """x at its own fixed point (0); intangible investment from the model's
    OWN normalization H(J) = J with H_J = 1 (their Section V.A), which
    gives J/K = a1^xi in closed form; c^s from the constraint; the ratio
    state s from ITS OWN deterministic fixed-point equation by a scalar
    bracket solve.  Nothing is copied from a solved solution."""
    T, iv = ctx["T"], ctx["iv"]
    al, dK, dS = T["alpha"], T["dK"], T["dS"]
    nu, eta, xi = T["nu"], T["eta"], T["xi"]
    yk = np.exp((1 - al) * wa)
    jk = T["a1"] ** xi                         # H(J)=J tangency: H_J(jk)=1
    js = jk / yk
    cs = 1.0 - is_ - js
    if cs <= 0 or is_ <= 0 or js <= 0: return None, np.nan
    ik = is_ * yk
    Hk = T["a1"] / (1 - 1 / xi) * jk ** (1 - 1 / xi) + T["a2"]

    def srow(s):
        sk = np.exp(s)
        Gk = (nu * ik ** (1 - 1 / eta)
              + (1 - nu) * sk ** (1 - 1 / eta)) ** (1 / (1 - 1 / eta))
        g = np.log(1 - dK + Gk)                # varpi = 1 at x = 0, W = 0
        inner = (1 - dS) * (sk - Gk) + Hk
        if inner <= 0 or not np.isfinite(g): return np.nan, np.nan
        return np.log(inner) - g - s, g

    grid_s = np.linspace(-6.0, 4.0, 121)
    vals = np.array([srow(s)[0] for s in grid_s])
    s_star, g = np.nan, np.nan
    for i in range(len(grid_s) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            s_star = brentq(lambda s: srow(s)[0], grid_s[i], grid_s[i + 1],
                            xtol=1e-12)
            g = srow(s_star)[1]
            break
    if not np.isfinite(s_star): return None, np.nan
    kap = np.log(cs) + (1 - al) * wa
    vmk = _vmk_closed(T, kap, g)
    if vmk is None: return None, np.nan
    x = np.zeros(ctx["n"])
    x[iv["vmk_t"]], x[iv["log_cmk_t"]] = vmk, kap
    x[iv["cs_t"]], x[iv["is_t"]], x[iv["js_t"]] = cs, is_, js
    x[iv["log_gk_t"]] = g
    x[iv["wa_t"]], x[iv["s_t"]], x[iv["x_t"]] = wa, s_star, 0.0
    return _finish(ctx, x)


# ---------------------------------------------------------------- registry
CFG = {
    "KL": dict(recon=recon_KL, ax=("w_t", "is_t"),
               lab=(r"$\omega=\log(Z/K)$", r"$i^s$ investment share"),
               g1=np.linspace(-9.0, 2.0, 200), g2=np.linspace(0.03, 1.15, 200),
               np1=8, np2=8, budget=45,
               rule="closed forms; multipliers by linear least squares"),
    "CROCE": dict(recon=recon_CROCE, ax=("wa_t", "is_t"),
               lab=(r"$\omega^a=\log(A/K)$", r"$i^s$ investment share"),
               g1=np.linspace(-7.0, 3.0, 200), g2=np.linspace(0.02, 1.10, 200),
               np1=8, np2=8, budget=45,
               rule="closed forms, $x=0$; multipliers by linear least squares"),
    "TALLARINI": dict(recon=recon_TALL, ax=("wx_t", "is_t"),
               lab=(r"$\omega^x=\log(X/K)$", r"$i^s$ investment share"),
               g1=np.linspace(-6.5, 3.0, 200), g2=np.linspace(0.02, 1.10, 200),
               np1=8, np2=8, budget=30,
               rule="closed forms; multipliers by linear least squares"),
    "AK": dict(recon=recon_AK, ax=("Z2_t", "D2_t"),
               lab=(r"$Z^2$ log stochastic-volatility state", r"$D^2$ investment/capital"),
               g1=np.linspace(-17.0, -6.0, 200), g2=np.linspace(0.0005, 0.0228, 200),
               np1=8, np2=8, budget=30,
               rule="closed forms, $Z^1=0$; multipliers by linear least squares"),
    "HABIT": dict(recon=recon_HABIT, ax=("X_t", "imk_t"),
               lab=(r"$X$ log habit stock", r"$i^{mk}$ capital investment / capital"),
               g1=np.linspace(-8.0, 1.0, 200), g2=np.linspace(0.002, 0.0918, 200),
               np1=7, np2=7, budget=60,
               rule="closed forms, $Z=0$, $Y=\\log\\mu_2$; value recursion by scalar bracket solve; multipliers by linear least squares"),
    "ACL": dict(recon=recon_ACL, ax=("wa_t", "is_t"),
               lab=(r"$\omega^a=\log(A/K)$", r"$i^s$ tangible-investment share"),
               g1=np.linspace(-6.0, 3.0, 140), g2=np.linspace(0.02, 0.90, 140),
               np1=5, np2=5, budget=100,
               rule="$x=0$; $J/K=a_1^{\\xi}$ from the model's own $H(J)=J$ normalization; $s$ from its own fixed-point equation (scalar bracket solve); multipliers by linear least squares"),
}

TITLE = {"AK": "AK (book §11.7)", "HABIT": "HABIT (book appendix)",
         "KL": "KL (RFS 2010)", "ACL": "ACL (RFS 2013)",
         "CROCE": "CROCE (2008 WP of JME 2014)", "TALLARINI": "TALLARINI (JME 2000)"}


def compute(name):
    cfg = CFG[name]
    ctx = compile_F(name)
    M, T, iv = ctx["M"], ctx["T"], ctx["iv"]
    a1n, a2n = cfg["ax"]
    G1, G2 = cfg["g1"], cfg["g2"]
    t0 = time.time()

    Z = np.full((len(G2), len(G1)), np.nan)
    for i2, v2 in enumerate(G2):
        for i1, v1 in enumerate(G1):
            _, r = cfg["recon"](ctx, v1, v2)
            Z[i2, i1] = np.log10(r + 1e-16) if np.isfinite(r) else np.nan
    print(f"[{name}] grid {Z.shape} in {time.time()-t0:.0f}s "
          f"({100*np.mean(np.isfinite(Z)):.0f}% admissible)", flush=True)

    # markers: unseeded auto guess / + optional paper seed, at their OWN residuals
    def own(g):
        F = ctx["Fraw"](np.asarray(g, float))
        return float(np.log10(np.nanmax(np.abs(F)) + 1e-16))
    g_cold, *_ = derive_guess(ctx["spec"], T, ctx["nS"], ctx["nW"])
    cold = (float(g_cold[iv[a1n]]), float(g_cold[iv[a2n]]), own(g_cold))
    seed = None
    if M["seeds"]:
        g_seed, *_ = derive_guess(ctx["spec"], T, ctx["nS"], ctx["nW"],
                                  state_overrides=M["seeds"])
        seed = (float(g_seed[iv[a1n]]), float(g_seed[iv[a2n]]), own(g_seed))

    # verified steady state (reference for the probe classification)
    r_ref, err = autosolve(M["build"], M["defaults"], T, ctx["nS"], ctx["nW"],
                           timeout=600, state_seeds=M["seeds"])
    assert r_ref is not None, f"reference solve failed: {err}"
    ss_ref = np.asarray(r_ref["ss"], float)
    full_ref = np.concatenate([[float(np.asarray(r_ref["recursive_ss"], float)[1])], ss_ref])
    xstar = (float(full_ref[iv[a1n]]), float(full_ref[iv[a2n]]))
    am = np.unravel_index(np.nanargmin(Z), Z.shape)
    star = (float(G1[am[1]]), float(G2[am[0]]), float(np.nanmin(Z)))
    print(f"[{name}] argmin {star[:2]} z={star[2]:.2f}; verified ss at {xstar}", flush=True)

    # sparse initialization grid: solve from R_M(z), classify the outcome.
    # Probe points are an evenly spaced subsample of the ADMISSIBLE cells:
    # the grid is binned np1 x np2, and each bin contributes its admissible
    # cell nearest the bin center (bins with no admissible cell contribute
    # nothing) — so coverage follows the admissible region.
    spec = ctx["spec"]
    sn = _ss_names(spec, ctx["nS"], ctx["nW"])[1:]   # engine's ss drops vmk
    mask = np.isfinite(Z)
    pts = []
    for b2 in np.array_split(np.arange(len(G2)), cfg["np2"]):
        for b1 in np.array_split(np.arange(len(G1)), cfg["np1"]):
            sub = mask[np.ix_(b2, b1)]
            if not sub.any():
                continue
            ii = np.argwhere(sub)
            c = np.array([(len(b2) - 1) / 2.0, (len(b1) - 1) / 2.0])
            k = ii[np.argmin(((ii - c) ** 2).sum(1))]
            pts.append((float(G1[b1[k[1]]]), float(G2[b2[k[0]]])))
    probes = []
    for v1, v2 in pts:
            x0, r0 = cfg["recon"](ctx, v1, v2)
            if x0 is None:
                continue                        # inadmissible: no probe
            t1 = time.time()
            rp, msg = _solve_checked(spec, T, sn, x0, timeout=cfg["budget"])
            if rp is None:
                out = 0                          # no convergence in budget
            else:
                d = float(np.max(np.abs(np.asarray(rp["ss"], float) - ss_ref)))
                out = 2 if d < 1e-6 else 1       # 2 = verified ss, 1 = other root
            probes.append((v1, v2, out, round(time.time() - t1, 1)))
            print(f"[{name}] probe ({v1:+.2f},{v2:.3f}) -> {['fail','other','ss*'][out]}"
                  f" {time.time()-t1:.0f}s", flush=True)
    probes = np.array(probes, float) if probes else np.zeros((0, 4))

    np.savez(os.path.join(HERE, f"landscape_{name.lower()}.npz"),
             Z=Z, G1=G1, G2=G2, cold=np.array(cold),
             seed=np.array(seed if seed else [np.nan] * 3),
             star=np.array(star), xstar=np.array(xstar), probes=probes,
             labels=np.array(cfg["lab"], dtype=object),
             rule=np.array(cfg["rule"], dtype=object))
    print(f"[{name}] total {time.time()-t0:.0f}s", flush=True)
    return name


def plot(name):
    d = np.load(os.path.join(HERE, f"landscape_{name.lower()}.npz"),
                allow_pickle=True)
    Z, G1, G2 = d["Z"], d["G1"], d["G2"]
    cold, seed, star, xstar, probes = d["cold"], d["seed"], d["star"], d["xstar"], d["probes"]
    lab = list(d["labels"]); rule = str(d["rule"])
    WW, II = np.meshgrid(G1, G2)
    Zc = np.ma.masked_invalid(np.clip(Z, ZMIN, ZMAX))

    fig = plt.figure(figsize=(12.5, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(WW, II, Zc, cmap=CMAP, norm=NORM, alpha=0.9,
                    linewidth=0, antialiased=True, rstride=2, cstride=2)
    zf = ZMIN - 2.0
    ax.contourf(WW, II, Zc, zdir="z", offset=zf, levels=24, cmap=CMAP,
                norm=NORM, alpha=0.65)
    # convergence overlay on the floor
    for out, c, m, lbl in ((2, "#2ca02c", "o", "solve from this cell reaches the verified steady state"),
                           (1, "#ff7f0e", "s", "converges to a different root"),
                           (0, "#d62728", "x", "no convergence within the probe budget")):
        P = probes[probes[:, 2] == out] if len(probes) else np.zeros((0, 4))
        if len(P):
            ax.scatter(P[:, 0], P[:, 1], np.full(len(P), zf), marker=m, s=26,
                       color=c, depthshade=False, label=lbl)
    ax.plot([star[0]] * 2, [star[1]] * 2, [zf, star[2]], color="gold", lw=1.5)
    ax.scatter([star[0]], [star[1]], [zf], marker="*", s=260, color="gold",
               edgecolor="k", label="lowest sampled residual on the reconstruction manifold", zorder=9)
    ax.scatter([xstar[0]], [xstar[1]], [zf], marker="P", s=90, color="#2ca02c",
               edgecolor="k", label="verified reference steady state", zorder=9)
    ax.plot([cold[0]] * 2, [cold[1]] * 2, [zf, cold[2]], color="red", lw=1.0, ls=":")
    ax.scatter([cold[0]], [cold[1]], [cold[2]], marker="o", s=110, color="#c44e52",
               edgecolor="k", label="unseeded auto guess (at its own residual)", zorder=9)
    if np.isfinite(seed[0]):
        ax.plot([seed[0]] * 2, [seed[1]] * 2, [zf, seed[2]], color="k", lw=1.0, ls=":")
        ax.scatter([seed[0]], [seed[1]], [seed[2]], marker="D", s=100, color="white",
                   edgecolor="k", label="auto guess + optional paper seed", zorder=9)
    ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
    ax.set_zlabel(r"$\log_{10}\|F\|_\infty$ (full system, reconstructed slice)")
    ax.set_zlim(zf, ZMAX)
    ax.set_title(f"{TITLE[name]} — loss landscape on the reconstruction manifold\n"
                 f"reconstruction: {rule}; no joint nonlinear solve at grid points.\n"
                 "Blank: outside the admissible reconstruction domain. "
                 "Floor markers: engine solves started from the reconstructed cell.",
                 fontsize=10)
    ax.view_init(elev=28, azim=-63)
    ax.legend(loc="upper left", fontsize=7)
    out = os.path.join(HERE, f"landscape_{name.lower()}.png")
    plt.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, flush=True)


def overview():
    order = ["AK", "HABIT", "KL", "ACL", "CROCE", "TALLARINI"]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    for k, name in enumerate(order):
        axp = axes[k // 3, k % 3]
        d = np.load(os.path.join(HERE, f"landscape_{name.lower()}.npz"),
                    allow_pickle=True)
        Z, G1, G2 = d["Z"], d["G1"], d["G2"]
        cold, seed, star, xstar, probes = (d["cold"], d["seed"], d["star"],
                                           d["xstar"], d["probes"])
        lab = list(d["labels"])
        WW, II = np.meshgrid(G1, G2)
        Zc = np.ma.masked_invalid(np.clip(Z, ZMIN, ZMAX))
        cf = axp.contourf(WW, II, Zc, levels=24, cmap=CMAP, norm=NORM)
        axp.set_facecolor("#c8c8c8")
        for out, c, m in ((2, "#2ca02c", "o"), (1, "#ff7f0e", "s"), (0, "#d62728", "x")):
            P = probes[probes[:, 2] == out] if len(probes) else np.zeros((0, 4))
            if len(P):
                axp.scatter(P[:, 0], P[:, 1], marker=m, s=18, color=c, zorder=4)
        axp.scatter([star[0]], [star[1]], marker="*", s=170, color="gold",
                    edgecolor="k", zorder=6)
        axp.scatter([xstar[0]], [xstar[1]], marker="P", s=60, color="#2ca02c",
                    edgecolor="k", zorder=6)
        axp.scatter([cold[0]], [cold[1]], marker="o", s=70, color="#c44e52",
                    edgecolor="k", zorder=6)
        if np.isfinite(seed[0]):
            axp.scatter([seed[0]], [seed[1]], marker="D", s=55, color="white",
                        edgecolor="k", zorder=6)
        axp.set_title(TITLE[name], fontsize=11)
        axp.set_xlabel(lab[0], fontsize=9); axp.set_ylabel(lab[1], fontsize=9)
        axp.tick_params(labelsize=8)
    fig.suptitle("Loss landscapes of the six economies on their reconstruction manifolds — "
                 "one residual color scale.  ● unseeded auto guess   ◇ + optional paper seed   "
                 "★ lowest sampled residual   + verified steady state;\n"
                 "floor dots: engine solves started from the reconstructed cell "
                 "(green: reaches the verified steady state; orange: a different root; red: no convergence). "
                 "Gray: outside the admissible reconstruction domain.", fontsize=10)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
                      ax=axes, shrink=0.8, pad=0.02)
    cb.set_label(r"$\log_{10}\|F\|_\infty$")
    out = os.path.join(HERE, "landscapes_overview.png")
    plt.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "overview":
        overview()
    elif arg == "all":
        for m in ["AK", "HABIT", "KL", "ACL", "CROCE", "TALLARINI"]:
            compute(m); plot(m)
        overview()
    else:
        compute(arg); plot(arg)
