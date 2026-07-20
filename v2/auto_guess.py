"""Automatic starting vector for the steady-state search, derived from the
model itself (v2 addition). Used when uncertain_expansion is called with
initial_guess=None.

Method (all one-dimensional scan + brentq root finds — no fragile joint
solve, no hand-tuned numbers):
  - states  <- fixed points of the DETERMINISTIC state equations;
  - investment controls (those in the capital-growth equation) <- invert
    deterministic growth to a modest target (0.005 per period);
  - consumption controls <- absorb the static constraints;
  iterated Gauss-Seidel style, then the utility entries are computed from
  the model's own expressions (value from the deterministic recursion,
  multiplier = (1-beta) * d kappa / d consumption).
"""
import numpy as np
import sympy as sp
from scipy.optimize import brentq


def _scan_root(f, lo, hi, n=240):
    xs = np.linspace(lo, hi, n)
    px, pv = None, None
    for x in xs:
        try:
            v = f(x)
        except Exception:
            v = np.nan
        if not np.isfinite(v):
            px, pv = None, None
            continue
        if pv is not None and np.sign(v) != np.sign(pv) and pv != 0:
            try:
                return brentq(f, px, x, xtol=1e-12)
            except Exception:
                pass
        px, pv = x, v
    return None


def derive_initial_guess(ss_variables, control_variables, state_variables,
                         shock_variables, output_constraint, capital_growth,
                         state_equations, static_constraints,
                         parameter_names, args, g_target=0.005):
    pvals = dict(zip(parameter_names.keys(), args))
    sub0 = {sp.Symbol("q_t"): 0}
    for w in shock_variables:
        sub0[sp.Symbol(w)] = 0
        sub0[sp.Symbol(w + "p1")] = 0
    for k, v in pvals.items():
        sub0[sp.Symbol(k)] = float(v)

    det = lambda e: e.subs(sub0)
    growth_d = det(capital_growth)
    states_d = [det(e) for e in state_equations]
    cons_d = [det(e) for e in static_constraints]
    kappa_d = det(output_constraint)

    ctrl = list(control_variables)
    stat = list(state_variables)
    csym = {c: sp.Symbol(c) for c in ctrl}
    ssym = {s: sp.Symbol(s) for s in stat}
    inv_ctrl = [c for c in ctrl if csym[c] in growth_d.free_symbols]
    oth_ctrl = [c for c in ctrl if c not in inv_ctrl]

    def point():
        p = {csym[c]: cvals[c] for c in ctrl}
        p.update({ssym[s]: svals[s] for s in stat})
        return p

    def num(expr, sym=None, x=None):
        p = point()
        if sym is not None:
            p[sym] = x
        return complex(expr.subs(p).evalf()).real

    # step the target growth down if it admits no positive-consumption point
    for g_try in [g_target, 0.003, 0.002, 0.001, 0.0005, 0.0002]:
      g_target = g_try
      cvals = {c: 0.01 for c in ctrl}
      svals = {s: 0.0 for s in stat}
      feasible = True
      for _ in range(3):
          def fg(v):
              p = point()
              for c in inv_ctrl:
                  p[csym[c]] = v
              return complex(growth_d.subs(p).evalf()).real - g_target
          v = _scan_root(fg, 1e-8, 1.0) or _scan_root(fg, 1e-8, 10.0)
          if v is not None:
              for c in inv_ctrl:
                  cvals[c] = v
          rem = list(oth_ctrl)
          for con in cons_d:
              tgt = next((c for c in rem if csym[c] in con.free_symbols), None)
              if tgt is None:
                  continue
              r = _scan_root(lambda x: num(con, csym[tgt], x), 1e-8, 1.0)
              if r is not None:
                  cvals[tgt] = r
                  rem.remove(tgt)
          for e, s in zip(states_d, stat):
              r = _scan_root(lambda x: num(e, ssym[s], x) - x, -30.0, 30.0)
              if r is not None:
                  svals[s] = r
          pinned = set(inv_ctrl)
          for con in cons_d:
              t = next((c for c in oth_ctrl if csym[c] in con.free_symbols), None)
              if t:
                  pinned.add(t)
          for c in ctrl:
              if c in pinned:
                  continue
              host = next(((e, s) for e, s in zip(states_d, stat)
                           if csym[c] in e.free_symbols), None)
              if host is None:
                  continue
              e, s = host
              r = _scan_root(lambda x: num(e, csym[c], x) - svals[s], 1e-8, 1.0)
              if r is not None:
                  cvals[c] = r

      if feasible and all(v > 0 for v in cvals.values()):
          break

    beta = float(pvals["beta"])
    rho = float(pvals["rho"])
    c_log = num(kappa_d)
    if abs(rho - 1.0) < 1e-6:
        v_util = c_log + beta * g_target / (1 - beta)
    else:
        lam_g = beta * np.exp((1 - rho) * g_target)
        v_util = (np.log((1 - beta) * np.exp((1 - rho) * c_log)
                         / max(1 - lam_g, 1e-10)) / (1 - rho))
    ms = 0.1
    cc = (oth_ctrl or ctrl)[0]
    try:
        ms = float((1 - beta) * num(sp.diff(kappa_d, csym[cc])))
    except Exception:
        pass

    names = [str(v) for v in ss_variables]
    names = names[1:names.index("q_t")]
    out = []
    for nm in names:
        if nm in cvals:
            out.append(float(cvals[nm]))
        elif nm in svals:
            out.append(float(svals[nm]))
        elif nm == "log_gk_t":
            out.append(g_target)
        elif nm == "vmk_t":
            out.append(v_util)
        elif nm == "rmv_t":
            out.append(v_util)
        elif nm == "log_cmk_t":
            out.append(c_log)
        elif nm == "ms_t":
            out.append(ms)
        elif nm == "mg_t":
            out.append(1.0)
        elif nm.startswith("m") and nm.endswith("_t"):
            out.append(0.0)
        else:
            out.append(0.1)
    return np.array(out)
