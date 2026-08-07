"""Natural-matching construction of `initial_guess` (v2.1).

Division of labor. Whatever a single equation naturally claims is taken
from that equation: each state from its own evolution equation, one
decision from the resource constraint, the value entries in closed form,
the multipliers by a linear solve. The remaining choice margins — whose
information lives in the first-order conditions — are solved against
their own first-order conditions, as one small joint block where they
couple. The typed steps supply feasibility and structure; the
first-order conditions supply the optimality that was missing. There is
no trial growth rate: the growth coordinate is computed from the growth
equation, which defines it given the decisions and states.

The output is not an approximation: on the test economies the
constructed point satisfies the complete steady-state system to machine
precision, and the root solve of `uncertain_expansion` (untouched)
confirms it.
"""
import numpy as np
import sympy as sp
from scipy.optimize import brentq, least_squares

import uncertain_expansion_faisal_feb26 as robust
from uncertain_expansion_faisal_feb26 import compile_equations


def _scan(f, lo, hi, n=160):
    xs = np.linspace(lo, hi, n)
    px = pv = None
    for x in xs:
        try:
            v = f(x)
        except Exception:
            v = np.nan
        if not np.isfinite(v):
            px = pv = None
            continue
        if pv is not None and np.sign(v) != np.sign(pv) and pv != 0:
            try:
                return brentq(f, px, x, xtol=1e-12)
            except Exception:
                pass
        px, pv = x, v
    return None


def compile_system(spec, n_states, n_shocks):
    """Lambdified complete deterministic steady-state system, with the
    preprocessing of generate_ss_function (tp1 collapsed, growth variable
    substituted, q and the shock vector set to zero)."""
    n_J = spec["n_controls"] + (n_states + 1) + 2
    eqs, variables, vt, H, L = compile_equations(
        parameter_names=spec["parameter_names"], variables=spec["variables"],
        variables_tp1=spec["variables_tp1"],
        control_variables=spec["control_variables"],
        state_variables=list(spec["state_variables"]),
        output_constraint=spec["kappa"], capital_growth=spec["growth"],
        state_equations=spec["state_equations"],
        static_constraints=spec["static_constraints"],
        var_shape=[n_J, n_states + 1, n_shocks],
        ExternalHabit=spec.get("external_habit", False))
    subs = robust.automate_step_1(variables)
    nG = variables.index(sp.Symbol("log_gk_t"))
    nQ = variables.index(sp.Symbol("q_t"))
    subs[variables[0]] = variables[nG]; subs[vt[0]] = variables[nG]
    subs[variables[nQ]] = 0.; subs[vt[nQ]] = 0.
    for w in variables[nQ + 1:nQ + n_shocks + 1]: subs[w] = 0.
    for w in vt[nQ + 1:nQ + n_shocks + 1]: subs[w] = 0.
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

    import re
    iv = {nm: i for i, nm in enumerate(names)}
    IM = [i for i, nm in enumerate(names) if re.fullmatch(r"m(s|g|\d+)_t", nm)]
    return dict(Fraw=Fraw, names=names, iv=iv, IM=IM, n=len(names),
                n_eq=len(eqs))


def construct(spec, params, n_states, n_shocks, n_starts=5):
    """Natural-matching starting vector. Returns (guess, error, info);
    guess is None when the joint block does not solve from any start."""
    from autosolve import _subs_deterministic, _flatten

    ctx = compile_system(spec, n_states, n_shocks)
    names, iv, Fr, IM, n = ctx["names"], ctx["iv"], ctx["Fraw"], ctx["IM"], ctx["n"]
    ctrl = list(spec["control_variables"])
    stat = list(spec["state_variables"])
    beta, rho = float(np.asarray(params["beta"])), float(np.asarray(params["rho"]))

    pvals = _flatten(params)
    states_d = [_subs_deterministic(e, spec, pvals) for e in spec["state_equations"]]
    growth_d = _subs_deterministic(spec["growth"], spec, pvals)
    kappa_d = _subs_deterministic(spec["kappa"], spec, pvals)
    cons_d = [_subs_deterministic(e, spec, pvals) for e in spec["static_constraints"]]
    ssym = {s: sp.Symbol(s) for s in stat}
    csym = {c: sp.Symbol(c) for c in ctrl}

    def mknum(vals):
        def num(expr, sym=None, x=None):
            p = dict(vals)
            if sym is not None:
                p[sym] = x
            v = complex(expr.subs(p).evalf())
            return np.nan if abs(v.imag) > 1e-10 else v.real
        return num

    # classification -------------------------------------------------------
    # residual decision: the first control the (single) constraint claims
    con_free = {c for c in ctrl
                if any(csym[c] in e.free_symbols for e in cons_d)}
    resid_ctrl = next((c for c in ctrl if c in con_free), ctrl[0])
    z_ctrl = [c for c in ctrl if c != resid_ctrl]

    # states: an equation that depends on its own state only through the
    # growth equation is a declared ratio's stationarity condition — it
    # determines no state, and its state joins the joint block
    neutral = {csym[c]: 0.01 for c in ctrl}
    neutral.update({ssym[s]: 0.0 for s in stat})
    num0 = mknum(neutral)
    trend_states, self_states = [], []
    for e, s in zip(states_d, stat):
        q0 = num0(e, ssym[s], 0.0) - 0.0 + num0(growth_d, ssym[s], 0.0)
        q1 = num0(e, ssym[s], 1.0) - 1.0 + num0(growth_d, ssym[s], 1.0)
        if np.isfinite(q0) and np.isfinite(q1) and abs(q1 - q0) < 1e-9:
            trend_states.append(s)
        else:
            self_states.append(s)
    z_names = z_ctrl + trend_states
    info = dict(resid_ctrl=resid_ctrl, z_ctrl=z_ctrl,
                trend_states=trend_states, self_states=self_states,
                z_names=z_names)

    # reconstruction: the typed steps, run inside the joint block ---------
    def reconstruct(z):
        vals = dict(neutral)
        for nm, v in zip(z_ctrl, z[:len(z_ctrl)]):
            vals[csym[nm]] = float(v)
        for nm, v in zip(trend_states, z[len(z_ctrl):]):
            vals[ssym[nm]] = float(v)
        num = mknum(vals)
        r = _scan(lambda x: num(cons_d[0], csym[resid_ctrl], x), 1e-8, 1.0) \
            or _scan(lambda x: num(cons_d[0], csym[resid_ctrl], x), 1e-8, 10.0)
        if r is None:
            return None, np.inf
        vals[csym[resid_ctrl]] = r
        num = mknum(vals)
        for _ in range(2):
            for e, s in zip(states_d, stat):
                if s not in self_states:
                    continue
                rr = _scan(lambda x: num(e, ssym[s], x) - x, -30.0, 30.0)
                if rr is None:
                    return None, np.inf
                vals[ssym[s]] = rr
                num = mknum(vals)
        g = num(growth_d)               # the growth equation defines it
        kap = num(kappa_d)
        if not (np.isfinite(g) and np.isfinite(kap)):
            return None, np.inf
        lam = beta * np.exp((1 - rho) * g)
        if not np.isfinite(lam) or lam >= 1:
            return None, np.inf
        vmk = (kap + np.log((1 - beta) / (1 - lam)) / (1 - rho)
               if abs(rho - 1) > 1e-6 else kap + beta * g / (1 - beta))
        x = np.zeros(n)
        for c in ctrl:
            x[iv[c]] = vals[csym[c]]
        for s in stat:
            x[iv[s]] = vals[ssym[s]]
        x[iv["log_gk_t"]] = g
        x[iv["log_cmk_t"]] = kap
        x[iv["vmk_t"]] = vmk
        F0 = Fr(x)
        if not np.isfinite(F0[0]):
            return None, np.inf
        if abs(F0[0]) > 1e-9:           # recursion row not the textbook one
            def f0(v):
                xx = x.copy(); xx[iv["vmk_t"]] = v
                rv = Fr(xx)[0]
                return rv if np.isfinite(rv) else np.nan
            for span in (0.5, 1.0, 2.0, 4.0):
                a, b = vmk - span, vmk + span
                fa, fb = f0(a), f0(b)
                if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                    x[iv["vmk_t"]] = brentq(f0, a, b, xtol=1e-13)
                    break
            F0 = Fr(x)
        cols = []
        for j in IM:                    # multipliers: linear, exact
            xj = x.copy(); xj[j] = 1.0
            cols.append(Fr(xj) - F0)
        mstar, *_ = np.linalg.lstsq(np.column_stack(cols), -F0, rcond=None)
        x[IM] = mstar
        F = Fr(x)
        if np.any(~np.isfinite(F)):
            return None, np.inf
        return x, float(np.max(np.abs(F)))

    if not z_names:                     # everything typed: nothing to search
        x, e = reconstruct(np.array([]))
        return x, e, info

    def resid(z):
        x, e = reconstruct(z)
        if x is None:
            return np.full(ctx["n_eq"], 10.0)
        F = Fr(x)
        return np.where(np.isfinite(F), F, 10.0)

    starts_c = [0.02, 0.06, 0.10]
    starts_s = [-4.0, -2.0, -1.0, 0.5, -6.0]
    cands = []
    for k in range(n_starts):
        z0 = np.array([starts_c[(k + i) % len(starts_c)] for i in range(len(z_ctrl))]
                      + [starts_s[(k + i) % len(starts_s)] for i in range(len(trend_states))])
        try:
            sol = least_squares(resid, z0, method="lm", xtol=1e-14, max_nfev=3000)
        except Exception:
            continue
        x, e = reconstruct(sol.x)
        if x is not None:
            cands.append((e, x))
    if not cands:
        return None, np.inf, info
    cands.sort(key=lambda c: c[0])
    return cands[0][1], cands[0][0], info
