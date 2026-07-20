"""Manual-free solving layer for the uncertain_expansion engine.

Three pieces, replacing the three manual steps observed in every existing
notebook:
  1. derive_guess(spec, params): a starting vector DERIVED from the model
     (states = deterministic fixed point; investment control inverts the
     growth equation to a modest target growth; consumption absorbs the
     resource constraint; the utility entries are computed from the model's
     own expressions). No hand-tuned numbers.
  2. autosolve(spec_fn, target): adaptive straight-line continuation from
     the anchor (= the model's create_args defaults) to the target
     parameters, bisecting the step on failure, warm-starting each solve
     from the previous solution. No hand-built warm-start chains.
  3. plain-language failure diagnostics: report where along the path the
     solve stopped instead of surfacing a raw numerical exception.
"""
import sys, os, io, signal, contextlib
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import numpy as np, sympy as sp
import uncertain_expansion_faisal_feb26 as robust
from uncertain_expansion_faisal_feb26 import compile_equations
from scipy.optimize import brentq, fsolve

class _TO(Exception): pass
def _alarm(s, f): raise _TO()

# ---------------------------------------------------------------------------
def _ss_names(spec, n_states, n_shocks):
    n_J = spec["n_controls"] + (n_states + 1) + 2
    _, ssv, _, _, _ = compile_equations(
        parameter_names=spec["parameter_names"], variables=spec["variables"],
        variables_tp1=spec["variables_tp1"],
        control_variables=spec["control_variables"],
        state_variables=list(spec["state_variables"]),
        output_constraint=spec["kappa"], capital_growth=spec["growth"],
        state_equations=spec["state_equations"],
        static_constraints=spec["static_constraints"],
        var_shape=[n_J, n_states + 1, n_shocks])
    names = [str(v) for v in ssv]
    return names[1:names.index("q_t")]

def _subs_deterministic(expr, spec, pvals):
    """Set q=0 and all shocks to 0; substitute numeric parameter values."""
    sub = {sp.Symbol("q_t"): 0}
    for w in spec["shock_variables"]:
        sub[sp.Symbol(w + "p1")] = 0
        sub[sp.Symbol(w)] = 0
    for k, v in pvals.items():
        sub[sp.Symbol(k)] = v
    return expr.subs(sub)

def _flatten(params):
    """Vector parameters become name1, name2, ... (the engine's convention)."""
    flat = {}
    for k, v in params.items():
        a = np.asarray(v, dtype=float)
        if a.ndim == 0:
            flat[k] = float(a)
        else:
            for i, x in enumerate(a):
                flat[f"{k}{i + 1}"] = float(x)
    return flat


def _scan_root(f, lo, hi, n=240):
    """1-d root by sign-change scan + brentq; None if no sign change."""
    xs = np.linspace(lo, hi, n)
    prev_x, prev_v = None, None
    for x in xs:
        try: v = f(x)
        except Exception: v = np.nan
        if not np.isfinite(v): prev_x, prev_v = None, None; continue
        if prev_v is not None and np.sign(v) != np.sign(prev_v) and prev_v != 0:
            try: return brentq(f, prev_x, x, xtol=1e-12)
            except Exception: pass
        prev_x, prev_v = x, v
    return None


def derive_guess(spec, params, n_states, n_shocks, g_target=0.005):
    """Model-derived starting vector, built by robust one-dimensional
    solves (scan + brentq) instead of a fragile joint fsolve:
      states  <- fixed points of the deterministic state equations,
      investment controls <- invert the growth equation to g_target,
      consumption controls <- absorb the static constraints,
    iterated Gauss-Seidel style; the utility entries are then computed
    from the model's own expressions. No hand-tuned numbers."""
    pvals = _flatten(params)
    ctrl = [c[:-2] for c in spec["control_variables"]]
    stat = [s[:-2] for s in spec["state_variables"]]
    csyms = {c: sp.Symbol(c + "_t") for c in ctrl}
    ssyms = {s: sp.Symbol(s + "_t") for s in stat}

    growth_d = _subs_deterministic(spec["growth"], spec, pvals)
    states_d = [_subs_deterministic(e, spec, pvals) for e in spec["state_equations"]]
    cons_d   = [_subs_deterministic(e, spec, pvals) for e in spec["static_constraints"]]
    kappa_d  = _subs_deterministic(spec["kappa"], spec, pvals)

    inv_ctrl = [c for c in ctrl if csyms[c] in growth_d.free_symbols]
    oth_ctrl = [c for c in ctrl if c not in inv_ctrl]

    def point():
        p = {csyms[c]: cvals[c] for c in ctrl}
        p.update({ssyms[s]: svals[s] for s in stat})
        return p

    def num(expr, sym=None, x=None):
        p = point()
        if sym is not None: p[sym] = x
        return complex(expr.subs(p).evalf()).real

    ok = True
    # if the target growth leaves no positive-consumption allocation, step
    # it down: the guess must be a feasible interior point, nothing more
    for g_target in [g_target, 0.003, 0.002, 0.001, 0.0005, 0.0002]:
      cvals = {c: 0.01 for c in ctrl}
      svals = {s: 0.0 for s in stat}
      ok = True
      for _ in range(3):                       # Gauss-Seidel passes
          # investment controls: common value v inverts growth to g_target
          def fg(v):
              p = point(); [p.update({csyms[c]: v}) for c in inv_ctrl]
              return complex(growth_d.subs(p).evalf()).real - g_target
          v = _scan_root(fg, 1e-8, 1.0)
          if v is None: v = _scan_root(fg, 1e-8, 10.0)
          if v is not None:
              for c in inv_ctrl: cvals[c] = v
          else: ok = False
          # consumption controls: one constraint each, solved 1-d
          rem = list(oth_ctrl)
          for con in cons_d:
              tgt = next((c for c in rem if csyms[c] in con.free_symbols), None)
              if tgt is None: continue
              r = _scan_root(lambda x: num(con, csyms[tgt], x), 1e-8, 1.0)
              if r is not None: cvals[tgt] = r; rem.remove(tgt)
              else: ok = False
          # states: each equation's own fixed point, wide scan
          for e, s in zip(states_d, stat):
              r = _scan_root(lambda x: num(e, ssyms[s], x) - x, -30.0, 30.0)
              if r is not None: svals[s] = r
              else: ok = False
          # leftover controls (e.g. a second capital's investment): close by
          # the equal-investment heuristic; their host state equation then
          # pins the corresponding ratio state in the states pass
          pinned = set(inv_ctrl)
          for con in cons_d:
              t = next((c for c in oth_ctrl if csyms[c] in con.free_symbols), None)
              if t: pinned.add(t)
          for c in ctrl:
              if c in pinned: continue
              if inv_ctrl: cvals[c] = cvals[inv_ctrl[0]]

      if ok and all(v > 0 for v in cvals.values()):
          break

    beta, rho = float(pvals["beta"]), float(pvals["rho"])
    c_log = num(kappa_d)
    g0 = g_target
    if abs(rho - 1.0) < 1e-6:
        v_util = c_log + beta * g0 / (1 - beta)
    else:
        lam_g = beta * np.exp((1 - rho) * g0)
        v_util = (np.log((1 - beta) * np.exp((1 - rho) * c_log)
                         / max(1 - lam_g, 1e-10)) / (1 - rho))
    ms = 0.1
    cc = (oth_ctrl or ctrl)[0]
    try:
        ms = float((1 - beta) * num(sp.diff(kappa_d, csyms[cc])))
    except Exception:
        pass

    names = _ss_names(spec, n_states, n_shocks)
    out = []
    for nm in names:
        base = nm[:-2] if nm.endswith("_t") else nm
        if base in cvals: out.append(float(cvals[base]))
        elif base in svals: out.append(float(svals[base]))
        elif nm == "log_gk_t": out.append(g0)
        elif nm == "vmk_t": out.append(v_util)
        elif nm == "rmv_t": out.append(v_util)
        elif nm == "log_cmk_t": out.append(c_log)
        elif nm == "ms_t": out.append(ms)
        elif nm == "mg_t": out.append(1.0)
        elif nm.startswith("m") and nm.endswith("_t"): out.append(0.0)
        else: out.append(0.1)
    return np.array(out), ok, names

# ---------------------------------------------------------------------------
def _residual(spec, params, ss_names, ss_vals):
    """Independent check: plug the returned steady state into the
    DETERMINISTIC state equations, growth consistency, and static
    constraints. Not the engine's own gate."""
    pvals = _flatten(params)
    point = {}
    vals = dict(zip(ss_names, ss_vals))
    for nm, v in vals.items():
        point[sp.Symbol(nm)] = v
        if nm.endswith("_t"):
            point[sp.Symbol(nm[:-2] + "_tp1")] = v
    errs = []
    for e, svar in zip(spec["state_equations"], spec["state_variables"]):
        d = _subs_deterministic(e, spec, pvals).subs(point)
        errs.append(complex(d.evalf()).real - vals.get(svar, 0.0))
    g = _subs_deterministic(spec["growth"], spec, pvals).subs(point)
    errs.append(complex(g.evalf()).real - vals.get("log_gk_t", 0.0))
    for c in spec["static_constraints"]:
        d = _subs_deterministic(c, spec, pvals).subs(point)
        errs.append(complex(d.evalf()).real)
    return float(np.max(np.abs(errs)))


def _solve_once(spec_at, guess, timeout=120):
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(timeout)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = robust.uncertain_expansion(
                spec_at["control_variables"], spec_at["state_variables"],
                spec_at["shock_variables"], spec_at["variables"],
                spec_at["variables_tp1"], spec_at["kappa"], spec_at["growth"],
                spec_at["state_equations"], spec_at["static_constraints"],
                np.asarray(guess, float), spec_at["parameter_names"],
                spec_at["args"], approach="1", iter_tol=1e-7, max_iter=500,
                ExternalHabit=spec_at.get("external_habit", False))
        signal.alarm(0)
        return r, None
    except _TO:
        return None, "TIMEOUT"
    except Exception as e:
        signal.alarm(0)
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def _solve_checked(spec_at, params, ss_names, guess, timeout=120, rtol=1e-6):
    r, err = _solve_once(spec_at, guess, timeout)
    if r is None:
        return None, err
    ss = np.asarray(r["ss"], float)
    try:
        res = _residual(spec_at, params, ss_names, ss)
    except Exception as e:
        return None, f"residual-check error: {type(e).__name__}"
    if not np.isfinite(res) or res > rtol:
        if np.isfinite(res) and res < 1e-4:
            # gray zone: the root is nearly right (a fallback path stopped
            # short of full precision) — polish by re-solving warm-started
            # AT the returned root, then re-judge once
            warm = np.concatenate([[guess[0]], ss])
            r2, err2 = _solve_once(spec_at, warm, timeout)
            if r2 is not None:
                ss2 = np.asarray(r2["ss"], float)
                try:
                    res2 = _residual(spec_at, params, ss_names, ss2)
                except Exception:
                    res2 = np.inf
                if np.isfinite(res2) and res2 <= rtol:
                    return r2, None
                res = min(res, res2)
        return None, f"REJECTED: returned root fails the model's own equations (residual {res:.1e})"
    return r, None

def autosolve(build_spec, anchor, target, n_states, n_shocks,
              timeout=90, max_bisect=7, log=None):
    """Coordinate-wise adaptive continuation from the anchor (the model's
    own default parameters) to the target: one changed parameter at a time,
    bisecting the step on failure, warm-starting every solve from the
    previous solution, and accepting a solve ONLY if the returned steady
    state satisfies the model's deterministic equations (independent
    residual gate)."""
    say = log or (lambda s: None)
    axes = [k for k in target
            if not np.allclose(np.asarray(anchor[k], float),
                               np.asarray(target[k], float))]
    p0 = dict(anchor)
    spec0 = build_spec(p0)
    guess, gok, names = derive_guess(spec0, p0, n_states, n_shocks)
    say(f"anchor guess derived (pre-solve ok={gok})")
    ss_names = names[1:]   # the engine returns the solved vector minus its first entry
    # fast path: derive a guess AT THE TARGET and try it directly
    pt = dict(anchor); pt.update(target)
    spect = build_spec(pt)
    gt, gtok, _ = derive_guess(spect, pt, n_states, n_shocks)
    rt, errt = _solve_checked(spect, pt, ss_names, gt, timeout)
    if rt is not None:
        say("target solved DIRECTLY from the derived guess (no continuation)")
        return rt, "OK (direct)"
    say(f"direct attempt failed ({errt}); falling back to continuation")
    r, err = _solve_checked(spec0, p0, ss_names, guess, timeout)
    if r is None:
        return None, ("anchor solve failed (" + str(err) + ") — the model's own "
                      "default parameters do not solve; check the model or the guess derivation")
    say("anchor solved and verified")
    head = guess[0]
    cur = dict(p0)
    ss = np.asarray(r["ss"], float)
    for ax in axes:
        a, b = np.asarray(anchor[ax], float), np.asarray(target[ax], float)
        t_done = 0.0
        say(f"axis {ax}: {a} -> {b}")
        while t_done < 1.0 - 1e-12:
            step = 1.0 - t_done
            for _ in range(max_bisect + 1):
                t_try = t_done + step
                trial = dict(cur)
                v = (1 - t_try) * a + t_try * b
                trial[ax] = float(v) if np.ndim(v) == 0 else v
                warm = np.concatenate([[head], ss])
                r2, err = _solve_checked(build_spec(trial), trial, ss_names, warm, timeout)
                if r2 is not None:
                    r, ss, cur, t_done = r2, np.asarray(r2["ss"], float), trial, t_try
                    say(f"  {ax}={trial[ax] if np.ndim(trial[ax])==0 else '(vec)'} ok (t={t_done:.3f})")
                    break
                step /= 2.0
            else:
                kind = ("solves are slow here (timeouts) — a finer path or a longer "
                        "budget may still reach it" if err == "TIMEOUT" else
                        "the steady state appears to become ill-behaved here")
                return None, (f"continuation stalled on axis '{ax}' at t={t_done:.3f}, "
                              f"value {cur[ax]} (last error: {err}); {kind}")
    return r, "OK"
