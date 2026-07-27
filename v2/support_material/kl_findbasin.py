import sys, os, warnings; warnings.filterwarnings("ignore")
V2 = os.path.expanduser("~/Documents/MFR/quantmfr-ch11-demo/v2"); sys.path.insert(0, V2)
import numpy as np, sympy as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import uncertain_expansion_faisal_feb26 as robust
from uncertain_expansion_faisal_feb26 import compile_equations
from autosolve import derive_guess
from models_sourced import MODELS

# ---- compile the full residual vector (pre-solve; identical preprocessing) ----
M = MODELS["KL"]; T = dict(M["defaults"]); T.update(M["target"])
spec = M["build"](T); nS, nW = 1, 1
n_J = spec["n_controls"] + (nS+1) + 2
eqs, variables, vt, H, L = compile_equations(
    parameter_names=spec["parameter_names"], variables=spec["variables"], variables_tp1=spec["variables_tp1"],
    control_variables=spec["control_variables"], state_variables=list(spec["state_variables"]),
    output_constraint=spec["kappa"], capital_growth=spec["growth"], state_equations=spec["state_equations"],
    static_constraints=spec["static_constraints"], var_shape=[n_J, nS+1, nW])
subs = robust.automate_step_1(variables)
nG = variables.index(sp.Symbol("log_gk_t")); nQ = variables.index(sp.Symbol("q_t"))
subs[variables[0]] = variables[nG]; subs[vt[0]] = variables[nG]
subs[variables[nQ]] = 0.; subs[vt[nQ]] = 0.
for w in variables[nQ+1:nQ+nW+1]: subs[w] = 0.
for w in vt[nQ+1:nQ+nW+1]: subs[w] = 0.
eqs = [e.subs(subs) for e in eqs]; var_solve = variables[1:nQ]
pd = {v: val for v, val in zip(spec["parameter_names"], spec["args"])}
fl = sp.lambdify(var_solve, [sp.sympify(e).subs(pd) for e in eqs], "numpy")
names = [str(v) for v in var_solve]
def Fraw(x):
    try:
        v = np.array(fl(*x), dtype=complex).flatten()
        out = np.where(np.abs(v.imag) > 1e-10, np.nan, v.real)
        return out
    except Exception:
        return np.full(len(eqs), np.nan)

iv = {n: names.index(n) for n in names}
al, be, rho, dlt, xi_, mu, a1, a2 = (T["alpha"], T["beta"], T["rho"], T["delta"], T["xi"], T["mu"], T["a1"], T["a2"])
IM = [iv["ms_t"], iv["m0_t"], iv["mg_t"]]   # multiplier coordinates

def reconstruct(w, is_):
    """All 9 coordinates from closed-form algebra + one linear lsq. NO solver."""
    cs = 1.0 - is_
    if cs <= 0 or is_ <= 0: return None, np.nan
    ik = is_ * np.exp((1-al)*w)
    phi = a1 + a2/(1-1/xi_) * ik**(1-1/xi_)
    g = np.log(1 - dlt + phi)
    if not np.isfinite(g): return None, np.nan
    ck = np.log(cs) + (1-al)*w                       # kappa
    lam = be*np.exp((1-rho)*g)
    if lam >= 1: return None, np.nan                  # transversality violated
    vmk = np.log((1-be)*np.exp((1-rho)*ck)/(1-lam))/(1-rho)
    x = np.zeros(len(names))
    x[iv["vmk_t"]], x[iv["log_cmk_t"]], x[iv["cs_t"]], x[iv["is_t"]] = vmk, ck, cs, is_
    x[iv["log_gk_t"]], x[iv["w_t"]] = g, w
    # multipliers: FOC block is LINEAR in (ms, m0, mg) -> least squares
    F0 = Fraw(x)
    if np.any(~np.isfinite(F0)): return None, np.nan
    cols = []
    for j in IM:
        xj = x.copy(); xj[j] = 1.0
        cols.append(Fraw(xj) - F0)
    A = np.column_stack(cols)
    mstar, *_ = np.linalg.lstsq(A, -F0, rcond=None)
    x[IM] = mstar
    F = Fraw(x)
    if np.any(~np.isfinite(F)): return None, np.nan
    return x, float(np.max(np.abs(F)))

WG = np.linspace(-9.0, 2.0, 200); IG = np.linspace(0.03, 1.15, 200)
Z = np.full((len(IG), len(WG)), np.nan)
for a_, i_ in enumerate(IG):
    for b_, w_ in enumerate(WG):
        _, r = reconstruct(w_, i_)
        Z[a_, b_] = np.log10(r + 1e-16) if np.isfinite(r) else np.nan
SUP = os.path.expanduser("~/Documents/MFR/quantmfr-ch11-demo/v2/support_material")
np.savez(os.path.join(SUP, "kl_basin_grid.npz"), Z=Z, WG=WG, IG=IG)
am = np.unravel_index(np.nanargmin(Z), Z.shape)
w_min, is_min, z_min = WG[am[1]], IG[am[0]], np.nanmin(Z)
print(f"basin bottom found (no solver): w={w_min:.4f}, is={is_min:.4f}, log10 residual={z_min:.2f}")
print(f"paper closed forms (external check only): w*=-4.1256, is*=0.3506")

# guesses projected onto the slice
g_cold, _, _, _ = derive_guess(spec, T, nS, nW)
g_auto, _, _, _ = derive_guess(spec, T, nS, nW, state_overrides=M["seeds"])
wc, ic = g_cold[iv["w_t"]], g_cold[iv["is_t"]]
wa_, ia_ = g_auto[iv["w_t"]], g_auto[iv["is_t"]]

fig = plt.figure(figsize=(12.5, 8.5)); ax = fig.add_subplot(111, projection="3d")
WW, II = np.meshgrid(WG, IG)
Zm = np.ma.masked_invalid(Z)
ax.plot_surface(WW, II, Zm, cmap="viridis_r", alpha=0.92, linewidth=0, antialiased=True, rstride=2, cstride=2)
zf = z_min - 1.5
ax.contourf(WW, II, Zm, zdir="z", offset=zf, levels=22, cmap="viridis_r", alpha=0.6)
def zat(w, i_):
    _, r = reconstruct(w, i_); return np.log10(r+1e-16) if np.isfinite(r) else np.nan
def zown(g):
    F = Fraw(np.asarray(g, float)); return np.log10(np.nanmax(np.abs(F)) + 1e-16)
z_cold, z_auto = zown(g_cold), zown(g_auto)
ax.plot([w_min]*2, [is_min]*2, [zf, z_min], color="gold", lw=1.5)
ax.scatter([w_min],[is_min],[zf], marker="*", s=260, color="gold", edgecolor="k",
           label=f"lowest sampled residual on the reconstruction manifold  (w={w_min:.2f}, i$^s$={is_min:.2f})", zorder=9)
ax.plot([wc, wc], [ic, ic], [zf, z_cold], color="red", lw=1.0, ls=":")
ax.scatter([wc],[ic],[z_cold], marker="o", s=110, color="#c44e52", edgecolor="k",
           label="unseeded auto guess (ratio state unpinned, defaults to 0)", zorder=9)
ax.plot([wa_, wa_], [ia_, ia_], [zf, z_auto], color="k", lw=1.0, ls=":")
ax.scatter([wa_],[ia_],[z_auto], marker="D", s=100, color="white", edgecolor="k",
           label="auto guess + optional paper seed", zorder=9)
ax.set_xlabel(r"$\omega=\log(Z/K)$"); ax.set_ylabel(r"$i^{s}$  investment share")
ax.set_zlabel(r"$\log_{10}\|F\|_\infty$  (full system, reconstructed slice)")
ax.set_title("KL merit-function geometry on the reconstruction manifold (grid evaluation + linear multiplier least squares;\n"
             "no nonlinear root solve). Gaps: outside the admissible reconstruction domain (negative consumption,\n"
             "or the deterministic value recursion undefined). No legacy hand-tuned guess exists for KL.")
ax.view_init(elev=28, azim=-63); ax.legend(loc="upper left", fontsize=8)
axi = fig.add_axes([0.70, 0.10, 0.27, 0.30])
axi.contourf(WW, II, Zm, levels=20, cmap="viridis_r")
axi.scatter([w_min],[is_min], marker="*", s=90, color="gold", edgecolor="k")
axi.scatter([wc],[ic], marker="o", s=40, color="#c44e52", edgecolor="k")
axi.scatter([wa_],[ia_], marker="D", s=35, color="white", edgecolor="k")
axi.set_facecolor("#bbbbbb"); axi.set_xlabel("$\\omega$", fontsize=8); axi.set_ylabel("$i^s$", fontsize=8)
axi.tick_params(labelsize=7); axi.set_title("2D view: valley location", fontsize=8)
plt.savefig("kl_basin_found.png", dpi=115)
import shutil as _sh
_sh.copy("kl_basin_found.png", os.path.join(SUP, "kl_basin_found.png"))
_sh.copy(os.path.abspath(__file__), os.path.join(SUP, "kl_findbasin.py"))
import shutil; shutil.copy("kl_basin_found.png", os.path.expanduser("~/Documents/MFR/loss_landscapes/kl_basin_found.png"))
print("saved + copied kl_basin_found.png")
