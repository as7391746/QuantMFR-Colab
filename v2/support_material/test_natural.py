"""Broader test economies for the natural-matching construction (v2.1).

Three additions beyond the six of models_sourced.py, each testing a
different corner of the class:

  GHK  Greenwood-Hercowitz-Krusell (AER 1997), two exogenous trends,
       zero-tax fixed-labor variant (our restrictions; see
       _sources_grid_papers/ghk_aer1997.pdf) - two coupled capital
       ratios, growth rates read from the declared trends.
  BY   Bansal-Yaron (JF 2004) Case II endowment economy, monthly
       (bansal_yaron_jf2004.pdf) - no choice margins at all: every
       coordinate is claimed by an equation; log-variance state is our
       flagged restatement of their Gaussian level-variance.
  GR   Gomme-Rupert (JME 2007 / Cleveland Fed WP 05-05) one-sector
       fixed-hours variant, quarterly (gomme_rupert_wp0505.pdf) - the
       profession's reference RBC calibration; frictionless beta
       re-derived because the printed one was calibrated with taxes.

Run:  python test_natural.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import sympy as sp
from models_sourced import spec_pack, MODELS
from natural_guess import construct, compile_system
from autosolve import _solve_checked, _ss_names


def _ghk(S):
    a = sp.exp(S("kh_t") * (S("ae") + S("as_") - 1)) * sp.exp(S("lx_t") * S("as_")) \
        * S("lbar") ** (1 - S("ae") - S("as_"))
    growth = sp.log(1 - S("de") + S("ies_t") * a) - sp.log(S("gq"))
    kappa = sp.log(S("cs_t")) + sp.log(a)
    khn = S("kh_t") + growth - sp.log(S("g")) + 0 * S("W1_tp1")
    lxn = sp.log((1 - S("ds")) * sp.exp(S("lx_t")) + S("iss_t") * a) - growth
    return kappa, growth, [khn, lxn], [1 - S("cs_t") - S("ies_t") - S("iss_t")]


def _by(S):
    sig = S("sigma") * sp.exp(S("s_t") / 2)
    growth = S("mu") + S("x_t") + sig * S("W1_tp1")
    xn = S("rho_x") * S("x_t") + S("phie") * sig * S("W2_tp1")
    sn = S("nu1") * S("s_t") + S("sigs") * S("W3_tp1")
    return sp.log(S("cs_t")), growth, [xn, sn], [1 - S("cs_t")]


def _gr(S):
    yk = sp.exp((1 - S("alpha")) * S("s1_t") + S("s2_t")) * S("hbar") ** (1 - S("alpha"))
    growth = sp.log(1 - S("delta") + S("uxs_t") * yk)
    kappa = sp.log(S("ucs_t")) + (1 - S("alpha")) * S("s1_t") + S("s2_t") \
        + (1 - S("alpha")) * sp.log(S("hbar"))
    s1n = S("s1_t") + S("lgam") - growth + 0 * S("W1_tp1")
    s2n = S("rho_m") * S("s2_t") + S("sig_m") * S("W1_tp1")
    return kappa, growth, [s1n, s2n], [1 - S("ucs_t") - S("uxs_t")]


EXTRA = {
    "GHK": dict(
        build=spec_pack(["cs_t", "ies_t", "iss_t"], ["kh_t", "lx_t"], 1, _ghk),
        params=dict(beta=1.0124 / 1.07, rho=1.001, gamma=1.001, ae=0.169957,
                    as_=0.130043, de=0.124, ds=0.056, gq=1.032, g=1.0124,
                    lbar=0.24),
        n_states=2, n_shocks=1,
        anchors={"log_gk_t": float(np.log(1.0124)), "lx_t": float(np.log(1.386019)),
                 "cs_t": 0.803712, "ies_t": 0.125693, "iss_t": 0.070595},
        tol=1e-4),
    "BY": dict(
        build=spec_pack(["cs_t"], ["x_t", "s_t"], 3, _by),
        params=dict(beta=0.998, gamma=10.0, rho=2 / 3, mu=0.0015, rho_x=0.979,
                    sigma=0.0078, phie=0.044, nu1=0.987, sigs=0.0378),
        n_states=2, n_shocks=3,
        anchors={"log_gk_t": 0.0015, "x_t": 0.0, "s_t": 0.0,
                 "vmk_t": 0.861296},
        tol=1e-5),
    "GR": dict(
        build=spec_pack(["ucs_t", "uxs_t"], ["s1_t", "s2_t"], 1, _gr),
        params=dict(beta=0.969762, gamma=1.001, rho=1.001, alpha=0.283,
                    delta=0.022633, lgam=0.0041912, hbar=1 / 3, rho_m=0.7518,
                    sig_m=0.0075),
        n_states=2, n_shocks=1,
        anchors={"log_gk_t": 0.0041912, "uxs_t": 0.1306, "s1_t": -1.108448},
        tol=1e-4),
}


def run():
    print("lineup (models_sourced):")
    for name in ["AK", "HABIT", "KL", "CROCE", "TALLARINI", "ACL"]:
        M = MODELS[name]
        T = dict(M["defaults"]); T.update(M["target"])
        spec = M["build"](T)
        t0 = time.time()
        x, err, info = construct(spec, T, M["n_states"], M["n_shocks"])
        print(f"  {name:10s} construct err={err:.1e} ({time.time()-t0:.0f}s)")
    print("additions:")
    for name, E in EXTRA.items():
        spec = E["build"](E["params"])
        t0 = time.time()
        x, err, info = construct(spec, E["params"], E["n_states"], E["n_shocks"])
        if x is None:
            print(f"  {name:4s} construct FAILED"); continue
        ctx = compile_system(spec, E["n_states"], E["n_shocks"])
        d = dict(zip(ctx["names"], np.asarray(x, float)))
        worst = max(abs(d[k] - v) for k, v in E["anchors"].items())
        sn = [str(s) for s in _ss_names(spec, E["n_states"], E["n_shocks"])][1:]
        t1 = time.time()
        r, msg = _solve_checked(spec, E["params"], sn, x, timeout=600)
        print(f"  {name:4s} construct err={err:.1e} ({t1-t0:.0f}s)  "
              f"worst anchor gap={worst:.1e}  engine "
              f"{'OK' if r else 'FAIL'} ({time.time()-t1:.0f}s)")


if __name__ == "__main__":
    run()
