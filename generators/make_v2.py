"""Generate v2_demo.ipynb — sourced-model edition.
Model: six economies (two from the book, four from the literature), each
with its processes in the paper's notation, calibration citations, and
frequency-conversion rules. Solve: all six to their published preference
targets + the 22-cell correlation battery. Plot: anchor table, invariance
checks, |mu0| pivot, elasticity paths."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}

cells = []

cells.append(md(r'''# `uncertain_expansion` — published production economies and the automatic initial guess

The engine in `v2/` is the book's expansion code (the robust steady-state variant) with one substantive addition: an **automatic initial guess** that constructs the solver's starting vector from the model declaration itself (`auto_guess.py`; the driver `autosolve.py` adds multi-start over the few states no one-dimensional equation can pin, optional paper-derived seeds, and residual gates that verify every returned steady state against the model's own equations). The solver mathematics is untouched; `v2/PROVENANCE.md` lists every difference from the branch copy, and `v2/AUTO_GUESS.md` defines the construction and its validation.

**Six economies.** Two from the book (the Section 11.7 AK economy and the computation appendix's habit economy) and four written directly from the literature — every calibration number traces to a printed table:

| model | source | preferences solved | solves cold? |
|---|---|---|---|
| AK | book, Section 11.7 | $\gamma = 8$ | yes (no seeds exist) |
| HABIT | book, computation appendix | $\gamma = 8$, $\lambda = 0.67$, $\tau = 0.01$ | yes (no seeds exist) |
| KL | Kaltenbrunner & Lochstoer, *RFS* 2010 (LRR II) | $\gamma = 5$, $\rho = 2/3$ | yes |
| ACL | Ai, Croce & Li, *RFS* 2013 (Extension 1) | $\gamma = 10$, $\rho = 0.5$ | yes (about 6 minutes) |
| CROCE | Croce, *JME* 2014 (fixed-labor WP version) | $\gamma = 30$, $\rho = 0.5$ | yes |
| TALLARINI | Tallarini, *JME* 2000 | $\chi = \gamma = 100$, $\rho \approx 1$ | yes |

"Cold" means the constructed guess alone, no seeds. **All six economies solve cold.** The two-capital ACL model required the paper's values until the initialization learned to read the balanced growth rate from the declared trends; it now solves cold in about six minutes, and the paper's values remain an optional accelerator (about 70 s). The Solve section reports the two modes separately for every model.

All models run at a **quarterly** frequency; conversions from each paper's native frequency use standard rules stated in each model's section. Runtime: about 60 minutes on Colab (the correlation battery is the bulk).'''))

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
import uncertain_expansion_faisal_feb26 as engine   # the book's expansion code (robust ss variant)
from autosolve import autosolve, _ss_names
from models_sourced import MODELS, with_loadings
from elasticity import exposure_elasticity, price_elasticity
print("engine:", os.path.relpath(engine.__file__))
print("models:", ", ".join(MODELS))'''))

# ---------------------------------------------------------------- MODEL
cells.append(md(r'''## Model

As in the computation appendix, every economy fits

$$
\begin{align}
X_{t+1}(\mathsf{q}) &= \psi^x\left[D_t(\mathsf{q}),\, X_t(\mathsf{q}),\, \mathsf{q}W_{t+1},\, \mathsf{q}\right] \\
\log G_{t+1}(\mathsf{q}) - \log G_t(\mathsf{q}) &= \psi^g\left[D_t(\mathsf{q}),\, X_t(\mathsf{q}),\, \mathsf{q}W_{t+1},\, \mathsf{q}\right] \\
\widehat{C}_t(\mathsf{q}) &= \kappa\left[D_t(\mathsf{q}),\, X_t(\mathsf{q})\right] + \widehat{G}_t(\mathsf{q}) \\
0 &= \phi\left[D_t(\mathsf{q}),\, X_t(\mathsf{q})\right]
\end{align}
$$

with growth variable $G_t = K_t$ and $W_{t+1} \sim N(0, I)$ i.i.d.

**A structural requirement discovered while testing.** The solver's first-order-condition assembly assumes the static constraint $\phi$ takes the form *constant minus the sum of all controls* — state-free, one constraint, every control in it. All the book's models have this form. A production economy with capital in the resource constraint (any Cobb–Douglas technology) must therefore be stated in **output shares** — controls $c^s_t + i^s_t = 1$ — with the state dependence moved into $\kappa$ and $\psi^g$, whose derivatives the solver handles generically. This is a pure change of decision variables; the economics is unchanged. Stated any other way, the compiled system can admit roots that are not the model's steady state, which is why every solution below is checked against paper-derived restrictions. (For the same reason, an endogenous labor choice — a second, time constraint — lies outside the current class; the two labor-choice papers below are handled as noted in their sections.)

Model declarations live in [`v2/models_sourced.py`](https://github.com/as7391746/QuantMFR-Colab/blob/main/v2/models_sourced.py) (about 15 lines each, in the pattern above); the sections below give each economy's processes and calibration provenance.'''))

cells.append(md(r'''### Shared notation

Symbols used across model sections. **Each model section below also carries its own variable dictionary** mapping the paper's notation to the engine's; this table covers what they share.

**Economic symbols**:

| symbol | meaning |
|---|---|
| $K_t$ | capital — the growth numeraire ($G_t = K_t$); every quantity is stationarized by dividing by $K$ |
| $C_t,\ I_t,\ J_t,\ Y_t$ | consumption, investment (physical; intangible for ACL), output |
| $\kappa$ | the consumption entry of the utility recursion: $\kappa = \widehat{C}_t - \widehat{K}_t = \log(C_t/K_t)$, a function of controls and states |
| $\psi^x,\ \psi^g,\ \phi$ | state laws of motion; the capital-growth equation; the static resource constraint |
| $\mathsf{q}$ | the perturbation parameter ($\mathsf{q}=0$ is the deterministic steady state; $\mathsf{q}^2$ terms are Ito corrections) |
| $W_{t+1}$ | i.i.d. $N(0, I)$ shocks; $\Sigma$ = loading matrix, $\Sigma_r$ = its $r$-th row |
| $\beta,\ \gamma,\ \rho$ | subjective discount factor, risk aversion, inverse IES — the recursive-utility parameters |
| $\xi$ *(inside model specs)* | the capital-adjustment-cost elasticity (Jermann). **Not** the robustness parameter: robustness $\xi_{\text{rob}} = 1/(\gamma-1)$ enters only through $\gamma$ |
| $c^s,\ i^s,\ j^s$ | controls stated as output shares $C/Y,\ I/Y,\ J/Y$, so the resource constraint reads $1 - \sum \text{shares} = 0$ |
| $\omega,\ \omega^a,\ \omega^x,\ s$ | endogenous ratio states $\log(Z/K),\ \log(A/K),\ \log(X/K),\ \log(S/K)$ |
| $x_t$ | the long-run-risk AR(1) state |
| $\mu^0$ | constant term of the first-order drift tilt of the uncertainty-adjusted probability — the implied worst-case "pessimism" |
| $a_1,\ a_2$ | Jermann normalization constants — **derived, never calibrated**: pinned by "no adjustment cost at the steady state" |

**Code objects** (used in the Solve and Plot cells):

| name | what it is |
|---|---|
| `engine` | the book's expansion code (`uncertain_expansion`, robust steady-state variant); solver mathematics unchanged |
| `MODELS[name]` | one economy: `build` (parameters $\to$ engine arguments), `defaults` (anchor values), `target` (published preference target), `seeds` (the paper's closed-form steady-state values, used as hints), `n_states`, `n_shocks` |
| `autosolve(build, defaults, target, ...)` | the solve driver: derives a starting vector from the model, solves, and accepts only if the model's own equations are satisfied; `state_seeds` passes the paper-derived hints |
| `_ss_names(...)` | coordinate names of the solved steady-state vector, in the engine's internal ordering |
| `with_loadings(build, L)` | replaces the shock vector $W$ by $L\,W$ (unit row norms) — the correlation battery |
| `directions(family, n)` | builds each correlation family's loading matrix |
| `log_sdf(sol, rho, beta)` | the log stochastic-discount-factor increment assembled from a solution — the appendix's own construction |
| `exposure_elasticity`, `price_elasticity` | Borovicka-Hansen shock elasticities, the construction behind Figures 11.1-11.3 |
| `T_ELAS = 160` | elasticity horizon (quarters) |
| `ANCH` | the paper anchors — closed-form steady-state values each solution is checked against |
| a solution `r` | dictionary: `r["ss"]` = steady state, `r["util_sol"]["mu_0"]` = the tilt $\mu^0$, `r["gc_tp1"]` = consumption growth, `X1_tp1`/`X2_tp1` = laws of motion |
'''))

cells.append(md(r'''### AK — the book, Ch. 11 (Borovicka, Hansen and Sargent), Section 11.7 planner's problem: the AK economy with stochastic volatility (Section 11.7.1)

**Source.** The book itself: Chapter 11, *Using Recursive Utility to Assess Uncertainty*, Section 11.7 "A planner's problem with recursive utility", Section 11.7.1 "An economy with long-run uncertainty" (source anchor `AK_model`), calibrated from the book's "Appendix to Chapter 11", section "Appendix D: Parameter values". Appendix D states the model as a discrete-time approximation to the continuous-time economy of Hansen, Khorrami and Tourre (2024), Section 4.4, with their annual parameters already converted there to a quarterly time unit. We solve exactly the version stated in Section 11.7, at the chapter's baseline configuration ($\gamma = 8$, $\rho = 1$, baseline $\alpha$) — the single base solve behind Figure 11.3; Figures 11.1 and 11.2 vary $\rho$ (re-pairing $\alpha$ by the Section 11.7.4 table) and, in Figure 11.1, also $\gamma$ around this same baseline. **Native frequency: quarterly** — no conversion by us, with one exception: the Section 11.7.4 table prints $\alpha$ in annual units (see the calibration table).

**Economics.** An AK planner economy: output is proportional to capital, $C_t + I_t = \alpha K_t$, and new capital is installed through the concave adjustment-cost technology $\frac{1}{\zeta}\log(1+\zeta I_t/K_t)$. Two exogenous states put the Bansal–Yaron channels inside a production economy: $Z_{1,t}$ moves the conditional growth rate of capital (long-run risk), and $Z_{2,t}$ is a log stochastic-volatility state — $\{\exp(Z_{2,t})\}$ approximates a Feller square-root process and scales every shock loading. It is in this lineup as the replication baseline: the book solves this economy itself and supplies a closed-form check at $\rho = 1$.

**Variable dictionary.**

| paper symbol | engine symbol | meaning | type | units / frequency |
|---|---|---|---|---|
| $K_t$, $\widehat{K}_t = \log K_t$ | growth numeraire ($G_t = K_t$, $\widehat{G}_t = \widehat{K}_t$) | capital; only its log growth enters, as $\psi^g$ | state | log units, quarterly |
| $C_t$, $\widehat{C}_t = \log C_t$; $D_{1,t} = C_t/K_t$ | `D1_t` | consumption (level) and its capital ratio — the paper's own first control | control | ratio to capital, quarterly flow |
| $I_t$; $D_{2,t} = I_t/K_t$ | `D2_t` | investment (level) and its capital ratio — the second control | control | ratio to capital, quarterly flow |
| $Z_{1,t}$ | `Z1_t` | long-run growth state (predictable component of capital growth) | state | log units, quarterly |
| $Z_{2,t}$ | `Z2_t` | log stochastic-volatility state | state | log units, quarterly |
| $W_{t+1}$ | `W1_t`, `W2_t`, `W3_t` | i.i.d. $N(0, I_3)$; component 1 = direct capital shock, 2 = growth-rate shock, 3 = volatility shock (identified by the loading rows) | shock | standard normal, quarterly |
| $\alpha$ | `alpha` | productivity of capital (output per unit of capital) | parameter | quarterly flow |
| $\delta$; $\beta = e^{-\delta}$ | `beta` | subjective discount rate; discount factor | parameter | per quarter |
| $\rho$ | `rho` | inverse elasticity of intertemporal substitution | parameter | unitless |
| $\gamma$ | `gamma` | risk aversion / robustness parameter | parameter | unitless |
| $\zeta$ | `zeta` | adjustment-cost curvature | parameter | unitless (quarterly-$I/K$ convention) |
| $\iota_k$ | `iota_k` | constant drift term in capital growth (the book leaves it unnamed; it enters as $-\iota_k$) | parameter | per quarter |
| $\nu_k$ | `nu_k` | loading of capital growth on $Z_{1,t}$ | parameter | unitless |
| $\nu_1$, $\nu_2$ | `nu_1`, `nu_2` | mean reversion of $Z_1$ and of the volatility state | parameter | per quarter |
| $\mu_2$ | `mu_2` | stationary level of $\exp(Z_2)$: $\exp(Z_2^0) = \mu_2$ | parameter | variance scale, unitless |
| $\sigma_k$, $\sigma_1$, $\sigma_2$ | `sigma_k`, `sigma_1`, `sigma_2` | shock-loading rows (each a 3-vector), all multiplied by $e^{Z_{2,t}/2}$ in the dynamics | parameter | quarterly loadings |

**Model as printed** (Section 11.7.1; the labeled displays carry MyST labels `equation2`–`equation4` in the source and print as equations (11.37)–(11.39) in the rendered book). Exogenous state dynamics (`equation2` = (11.37)):

$$Z_{1,t+1} - Z_{1,t} = -\nu_1 Z_{1,t} + \exp\left(\tfrac{1}{2} Z_{2,t}\right)\sigma_1 W_{t+1}$$

$$Z_{2,t+1} - Z_{2,t} = -\nu_2\left[1 - \mu_2 \exp(-Z_{2,t})\right] - \tfrac{1}{2}|\sigma_2|^2 \exp(-Z_{2,t}) + \exp\left(-\tfrac{1}{2} Z_{2,t}\right)\sigma_2 W_{t+1}$$

Resource constraint and capital evolution (both displayed in Section 11.7.1; unnumbered in the book — a label `equation1` is commented out in the source):

$$C_t + I_t = \alpha K_t$$

$$\widehat{K}_{t+1} - \widehat{K}_t = \left[\tfrac{1}{\zeta}\log\left(1 + \zeta \tfrac{I_t}{K_t}\right) + \nu_k Z_{1,t} - \iota_k\right] - \tfrac{1}{2}|\sigma_k|^2 \exp(Z_{2,t}) + \exp\left(\tfrac{1}{2} Z_{2,t}\right)\sigma_k W_{t+1}$$

The book's own static restatement in the controls $D_t = (C_t/K_t,\ I_t/K_t)$ (`equation3` = (11.38); the book writes it in terms of $\widehat{G}_t = \widehat{K}_t$):

$$0 = \alpha - D_{1,t} - D_{2,t}, \qquad \widehat{C}_t - \widehat{G}_t = \log D_{1,t}$$

At $\rho = 1$ the first-order conditions (`equation4` = (11.39), specialized in Section 11.7.2, "First-order approximation when $\rho = 1$") collapse to a scalar equation with closed-form solution:

$$\frac{1-\beta}{D_{2,t} - \alpha} + \frac{\beta}{1 + \zeta D_{2,t}} = 0 \quad\Longrightarrow\quad D_2^{*} = \frac{(\beta - 1) + \beta\alpha}{\beta + (1-\beta)\zeta}.$$

**Mapping to the engine form.** The book states this economy already in the engine's own coordinates — capital is the growth numeraire and both controls are ratios to it — so the change of variables is the identity (no output-share restatement needed):

- $X_t = (Z_{1,t}, Z_{2,t})$: $\psi^x$ is exactly the two exogenous displays above, with shocks entering as $\mathsf{q}W_{t+1}$ and the $-\frac{1}{2}|\sigma_2|^2$ correction carrying $\mathsf{q}^2$ ($\mathsf{q}$ is the perturbation parameter of the Model section; $\mathsf{q} = 1$ recovers the printed system).
- $G_t = K_t$: $\psi^g = \frac{1}{\zeta}\log(1 + \zeta D_{2,t}) + \nu_k Z_{1,t} - \iota_k - \frac{\mathsf{q}^2}{2}|\sigma_k|^2 e^{Z_{2,t}} + e^{Z_{2,t}/2}\sigma_k W_{t+1}$ — the capital display with $I_t/K_t = D_{2,t}$.
- $\kappa = \log D_{1,t}$ (second line of `equation3`) and $\phi = \alpha - D_{1,t} - D_{2,t}$ — already in the required constant-minus-sum-of-controls class.
- Engine names coincide with the paper symbols: `D1_t`, `D2_t`, `Z1_t`, `Z2_t`, shocks `W1_t`–`W3_t`.

**Calibration** (Appendix D "Parameter values", quarterly; $\alpha$, $\gamma$, $\rho$ from the Section 11.7.4 figure configuration):

| parameter | paper value (native freq) | conversion rule | engine value (quarterly) | status |
|---|---|---|---|---|
| $\delta$ | 0.025 (Appendix D table) | printed value is a $\times 10$ typo for the quarterly rate (annual 0.01) | 0.0025 | **calibrated** (typo; see limitations) |
| $\beta$ | — | $\beta = e^{-\delta}$ (the book's own map, Section 11.3.1: $\beta_\epsilon = e^{-\delta\epsilon}$) | $e^{-0.0025} \approx 0.997503$ | **derived** |
| $\iota_k$ | 0.01 (Appendix D table) | none | 0.01 | **calibrated** |
| $\zeta$ | 32 (Appendix D table) | none (already the quarterly-$I/K$ convention) | 32.0 | **calibrated** |
| $\nu_k$ | 0.01 (Appendix D table) | none | 0.01 | **calibrated** |
| $\nu_1$ | 0.014 (Appendix D table) | none | 0.014 | **calibrated** |
| $\nu_2$ | 0.0485 (Appendix D table) | none | 0.0485 | **calibrated** |
| $\mu_2$ | $6.3\times 10^{-6}$ (Appendix D table) | none | $6.3\times 10^{-6}$ | **calibrated** |
| $\sigma_k$ | $\sqrt{3}\,(0.92,\ 0.40,\ 0)$ (Appendix D matrix) | none | $\sqrt{3}\,(0.92,\ 0.40,\ 0)$ | **calibrated** |
| $\sigma_1$ | $\sqrt{3}\,(0,\ 5.70,\ 0)$ (Appendix D matrix) | none | $\sqrt{3}\,(0,\ 5.70,\ 0)$ | **calibrated** |
| $\sigma_2$ | $\sqrt{3}\,(0,\ 0,\ 0.00031)$ (Appendix D matrix) | none | $\sqrt{3}\,(0,\ 0,\ 0.00031)$ | **calibrated** |
| $\alpha$ | 0.092 at $\rho = 1$ (Section 11.7.4 $\rho$–$\alpha$ table; annual units) | annual flow $\div\ 4$: $0.092/4 = 0.023$ is the value the book's own driver uses for Figures 11.1–11.2 at $\rho = 1$; the engine carries the Figure 11.3 base value $0.0922/4$, where 0.0922 comes from the book's companion code, not the printed table | 0.02305 | **calibrated** (base value via the book's code — see limitations) |
| $\rho$ | 1 (the chapter's baseline case) | unit-EIS offset | 1.001 | **convention** |
| $\gamma$ | 8 (Section 11.7.4 figure configuration) | degeneracy offset | 8.001 (solve target; continuation default 1.001) | **convention** |

The loading rows look large but always multiply $e^{Z_{2,t}/2}$ with $\exp(Z_2^0) = \mu_2$: the effective steady-state quarterly capital-growth volatility is $\sqrt{\mu_2}\,|\sigma_k| \approx 0.0044$.

**Anchors.**

- **Independent closed form** (the book's own algebra for the $\rho = 1$ planner problem, Section 11.7.2 — independent of the expansion code): $D_2^{*} = \frac{(\beta-1)+\beta\alpha}{\beta+(1-\beta)\zeta} = 0.019023$ at the engine parameter values, with $D_1^{*} = \alpha - D_2^{*}$. The engine at $\rho = 1.001$ returns $D_2 = 0.019016$ — a $7.1\times 10^{-6}$ gap attributable to the $\rho$ offset.
- **Book-internal consistency checks** (the order-zero statements of Sections 11.7.1–11.7.2): $Z_1^0 = 0$ and $\exp(Z_2^0) = \mu_2$, i.e. $Z_2^0 = \log\mu_2 \approx -11.975$; order-zero capital growth $\frac{1}{\zeta}\log(1 + \zeta D_2^{*}) - \iota_k \approx 0.00486$ per quarter.

**Limitations / restrictions.**

- $\rho = 1.001$ and $\gamma = 8.001$ instead of exactly 1 and 8 — the book's own offsets around the unit-EIS degeneracy (the book's own Figure 11.3 image file is named `price_elasticity_alpha0.092_rho1.001.jpg`, and its figure driver solves the base case at $\gamma = 8.001$, $\rho = 1.001$); the anchor comparison above quantifies the effect.
- Appendix D prints $\delta = 0.025$; that is a $\times 10$ typo (the quarterly counterpart of the annual 0.01 is 0.0025), and the engine uses 0.0025. With the printed 0.025 the book's own closed form gives a negative $D_2^{*}$. The caption of Figure 11.3 corroborates the annual-0.01 reading: it prints $\beta = .99$, the annual discount factor $e^{-0.01}$.
- $\alpha$ does not appear in Appendix D; Section 11.7.4 prints $\alpha = 0.092$ in annual units alongside otherwise quarterly parameters. Quarterly $\alpha = 0.092/4 = 0.023$ is what the book's driver uses for Figures 11.1–11.2 at $\rho = 1$; the engine's $0.02305 = 0.0922/4$ is the Figure 11.3 base value, where 0.0922 comes from the book's companion code — it is not printed in the chapter's text or tables (the same number does appear as the output share $a$ of the habit extension of this economy in the book's uncertainty-expansion code notebook). The quarterly value is required for internal consistency: evaluating the $D_2^{*}$ formula with $\alpha = 0.092$ and $\zeta = 32$ gives 0.083, not the 0.019 behind the figures.
- No structural restrictions: the model is solved exactly as stated in Section 11.7.

**Solve status: solves cold — automatic guess alone, no seeds exist for this model (about 2 s); this is also the one model with a legacy hand-built guess in the book, which the automatic guess replaces.**'''))

cells.append(md(r'''### HABIT — the book, Chapter 11 planner section + computation appendix (internal-habit economy, solved as the external-habit variant)

**Source.** The book itself, twice over: the economics is stated in the habit subsection of Section 11.7 (Chapter 11, "Another model of intertemporal substitution/complementarity", MyST target `habits`), and the exact system solved here is stated in Section 2 ("Example") of the computation appendix notebook ("Uncertainty Expansion - Computation Process", `Appendix_UncertaintyExpansion_code.ipynb`): the Section 11.7 AK technology extended with a habit stock, written in **annualized units with discrete step $\epsilon = 0.25$** (one quarter) — the appendix's own convention; the $\epsilon$ and $\sqrt{\epsilon}$ factors below are its statement, not ours. The VERSION solved is the **external-habit variant**: the appendix states the internal-habit planner problem (its FOCs compile automatically) and provides a flag (its Section 2.6, "External Habit Model" — the notebook numbers two subsections 2.6) that switches one first-order condition to the external specification; the engine declaration sets that flag. No outside journal publication; native frequency quarterly.

**Economic meaning.** Consumption services are a CES bundle of current measured consumption and a habit stock — a geometric average of past measured consumption — so preferences are time-nonseparable: $0 < \lambda \le 1$ gives consumption durability/substitution, $\lambda < 0$ gives Ryder–Heal intertemporal complementarity ("habit persistence"). The planner splits a fixed output flow between feeding the habit/consumption bundle and feeding capital, on top of the AK long-run-risk and stochastic-volatility technology. It is in this lineup as the second book-replication baseline: the one model with an extra endogenous preference-side state, and the engine's only use of the external-habit FOC switch.

**Variable dictionary** (paper = chapter/appendix notation; App. D = the Chapter 11 appendix's quarterly parameter table):

| paper symbol | engine symbol | meaning | type | units / frequency |
|---|---|---|---|---|
| $K_t$ | — (growth numeraire) | capital stock; $\widehat K_t = \widehat G_t = \log K_t$ (hats denote logarithms), all quantities divided by $K_t$ | state | level, quarterly step |
| $H_t$ | — (enters via $X_t$) | habit stock, geometric average of past measured consumption | state | level |
| $Z_{1,t}$ (appendix code $Z_t$) | `Z_t` | long-run growth state shifting the capital drift | state | annualized units |
| $Z_{2,t}$ (appendix code $Y_t$) | `Y_t` | log-volatility state; the $\sigma_k, \sigma_1$ loadings scale by $e^{Z_{2,t}/2}$, its own $\sigma_2$ loading by $e^{-Z_{2,t}/2}$; steady state $e^{Y^*}=\mu_2$ at $\mathsf q = 0$ | state | log units |
| $X_{1,t} = \log H_t - \log K_t$ | `X_t` | log habit-to-capital ratio | state | log ratio |
| $I_{h,t}$; $D_{1,t} = I_{h,t}/K_t$ | `imh_t` | habit/consumption investment ("measured consumption") per unit capital | control | annualized ratio |
| $I_{k,t}$; $D_{2,t} = I_{k,t}/K_t$ | `imk_t` | physical investment per unit capital | control | annualized ratio |
| $C_t$ | — (given by $\kappa$) | consumption–habit CES bundle entering utility; pinned by `imh_t` and `X_t`, not a separate choice | derived | level |
| $W_{t+\epsilon}$ | `W1_t, W2_t, W3_t` | i.i.d. $N(0, I_3)$ innovations | shock | one draw per quarter |
| $\mathsf q$ | `q_t` | perturbation scale on the variance-correction terms ($\mathsf q = 1$ is the full model) | perturbation variable | unit-free |
| $\epsilon$ | `epsilon` | time step | parameter | $0.25$ yr $=$ 1 quarter |
| $\beta,\ \gamma,\ \rho$ | `beta`, `gamma`, `rho` | per-step discount factor; risk aversion; inverse EIS | parameter | $\beta$ per quarter |
| $\lambda,\ \tau$ | `llambda`, `tau` | habit weight and curvature of the CES bundle | parameter | unit-free |
| $\nu_h$ | `nu_h` | habit depreciation per step | parameter | per quarter |
| $\zeta$ | `phi_1` $= 1/\zeta$, `phi_2` $= \zeta$ | adjustment-cost curvature (appendix stores $1/\zeta$ and $\zeta$ separately) | parameter | annualized |
| $\alpha$ (chapter) $= a$ (appendix) | `a` | output–capital ratio in $I_{h,t} + I_{k,t} = a K_t$ | parameter | annualized |
| $\iota_k,\ \nu_k$ (App. D) | `alpha_k`, `beta_k` | capital drift intercept and $Z$-loading; App. D quarterly value $= \epsilon\,\times$ engine value | parameter | annualized rate |
| $\nu_1,\ \nu_2$ (App. D) | `beta_z`, `beta_2` | mean reversion of $Z_{1}$ and $Z_{2}$; App. D quarterly value $= \epsilon\,\times$ engine value | parameter | annualized rate |
| $\mu_2$ | `mu_2` | volatility level: steady state of $e^{Z_{2,t}}$ | parameter | unit-free |
| $\sigma_k,\ \sigma_1,\ \sigma_2$ | `sigma_k`, `sigma_z`, `sigma_y` | shock-loading rows; per-step loading is $\sqrt{\epsilon}\,\sigma$ | parameter | annualized ($\sqrt{12}\times$ base) |

**Equations (the book's own statement).** Chapter 11, `habits` subsection — habit stock (first display) and the CES bundle (their eq. `consumption_habit`) and output split (display after "We modify the output constraint"):

$$H_{t+1} = e^{-\nu_h} H_t + \left(1 - e^{-\nu_h}\right) I_{h,t}, \qquad C_t = \left[(1-\lambda)\, I_{h,t}^{\,1-\tau} + \lambda\, H_t^{\,1-\tau}\right]^{\frac{1}{1-\tau}}, \qquad I_{h,t} + I_{k,t} = \alpha K_t$$

Appendix notebook, Section 2.3, annualized with step $\epsilon$ (its displayed math keeps the $\epsilon$ factors but writes the drift coefficients with the App. D quarterly symbols $\nu_1, \nu_2, \iota_k, \nu_k, \zeta$ — an extra $\epsilon$ relative to App. D — and drops $\epsilon$ and $\mathsf q^2$ on some variance terms; its code cells are definitive and the engine transcribes them verbatim):

$$Z_{t+\epsilon} = Z_t - \epsilon\,\beta_z Z_t + \sqrt{\epsilon}\; e^{Y_t/2}\, \sigma_z\, W_{t+\epsilon}$$

$$Y_{t+\epsilon} = Y_t - \epsilon\,\beta_2\!\left(1 - \mu_2\, e^{-Y_t}\right) - \epsilon\, \mathsf q^2\, \tfrac{1}{2} |\sigma_y|^2 e^{-Y_t} + \sqrt{\epsilon}\; e^{-Y_t/2}\, \sigma_y\, W_{t+\epsilon}$$

$$\widehat K_{t+\epsilon} - \widehat K_t = \epsilon \left[ \tfrac{1}{\zeta} \log\!\left(1 + \zeta\, \tfrac{I_{k,t}}{K_t}\right) - \alpha_k + \beta_k Z_t - \mathsf q^2\, \tfrac{1}{2} |\sigma_k|^2 e^{Y_t} \right] + \sqrt{\epsilon}\; e^{Y_t/2}\, \sigma_k\, W_{t+\epsilon}$$

$$X_{1,t+\epsilon} = \log\!\left[ e^{-\nu_h + X_{1,t}} + \left(1 - e^{-\nu_h}\right) \tfrac{I_{h,t}}{K_t} \right] - \left(\widehat K_{t+\epsilon} - \widehat K_t\right)$$

Internal vs. external. With $MS_t$ the costate on the output constraint, $MX_{1,t+1}$ the costate on the habit ratio, $N^*_{t+1}$ the recursive-utility change of measure, and $\mathfrak A_t$ date-$t$ information, the internal FOC for $D_{1,t}$ — chapter eq. `marginal_internal_habit`, displayed there for $\rho = 1$, $\tau = 1$ — is

$$MS_t = (1-\beta)\left(\frac{1-\lambda}{D_{1,t}}\right) + \mathbb E\!\left[ N^*_{t+1}\, \frac{\beta K_t \left(1 - e^{-\nu_h}\right)}{e^{X_{1,t+1}}\, K_{t+1}}\; MX_{1,t+1} \,\Big|\, \mathfrak A_t \right]$$

(the chapter and the appendix's Section 2.6 display the expectation with a $1 \times 3$ selector row against the full costate vector $MX_{t+1}$, writing the habit-ratio component simply as $X_{t+1}$; we show the contracted scalar form), and the **external** variant drops the expectation term (appendix Section 2.6): equivalently, the derivative of the $X_{1,t}$ law with respect to $I_{h,t}/K_t$ is set to zero in the compiled FOC. The general-$\tau$ analogue is compiled automatically.

**Mapping to the engine form.** No change of variables is needed — the appendix already states everything in ratios to capital: $\kappa = \frac{1}{1-\tau}\log\!\left[(1-\lambda)\,\texttt{imh\_t}^{\,1-\tau} + \lambda\, e^{(1-\tau) X_t}\right]$ (the chapter display of $\log C_t - \log K_t$); $\psi^g$ is the capital-growth equation above with `imk_t`; $\psi^x$ stacks the $Z$, $Y$, $X$ laws, the $X$ law carrying $-\psi^g$ by substitution; $\phi = a - \texttt{imh\_t} - \texttt{imk\_t}$ with $a = 0.0922$. The engine declaration passes `external=True` to `spec_pack`, which stores it as the spec's `external_habit` field; the driver forwards it to the solver as the appendix's own `ExternalHabit=True` option — the Section 2.6 switch. The builder is a verbatim transcription of the appendix's Section 2.3 code cells.

**Calibration** — appendix notebook parameter cell (`create_args`), cross-checked against App. D ("Appendix D: Parameter values" plus its $\nu_h$ sentence). Note on the third column header: for this model the engine stores the appendix's **annualized** rates verbatim; the quarterly step is carried by the explicit $\epsilon = 0.25$ factors inside the equations, so App. D's quarterly numbers are recovered as $\epsilon \times$ (engine value) and $\sqrt{\epsilon} \times$ ($\sigma$ rows).

| parameter | paper value (native freq) | conversion rule | engine value (quarterly) | status |
|---|---|---|---|---|
| $\epsilon$ | $0.25$ yr per step (appendix cell) | none | 0.25 | **calibrated** |
| $\beta$ | $\delta = 0.01$/yr (appendix cell) | $\beta = e^{-\delta\epsilon}$ | $e^{-0.0025}$ | **derived** |
| $a$ | $0.0922$ annualized (appendix cell) | none; $\epsilon\, a = 0.02305$, the AK quarterly $\alpha$ | 0.0922 | **calibrated** |
| $1/\zeta,\ \zeta$ | $\zeta = 8$ annualized (appendix cell) | quarterly $\zeta_q = \zeta/\epsilon = 32$ (App. D) | `phi_1` $=0.125$, `phi_2` $=8.0$ | **calibrated** |
| $\alpha_k$ | $0.04$/yr (appendix cell) | $\epsilon\,\alpha_k = \iota_k = 0.01$ (App. D) | 0.04 | **calibrated** |
| $\beta_k$ | $0.04$/yr (appendix cell) | $\epsilon\,\beta_k = \nu_k = 0.01$ (App. D) | 0.04 | **calibrated** |
| $\beta_z$ | $0.056$/yr (appendix cell) | $\epsilon\,\beta_z = \nu_1 = 0.014$ (App. D) | 0.056 | **calibrated** |
| $\beta_2$ | $0.194$/yr (appendix cell) | $\epsilon\,\beta_2 = \nu_2 = 0.0485$ (App. D) | 0.194 | **calibrated** |
| $\mu_2$ | $6.3\times 10^{-6}$ (both sources) | none | 6.3e-6 | **calibrated** |
| $\sigma_k$ | $\sqrt{12}\,(0.92,\ 0.40,\ 0)$ (appendix cell) | $\sqrt{\epsilon}\,\sigma_k = \sqrt{3}\,(0.92, 0.40, 0)$ = App. D row | `sqrt(12)*[0.92, 0.40, 0.0]` | **calibrated** |
| $\sigma_1$ | $\sqrt{12}\,(0,\ 5.7,\ 0)$ (appendix cell) | $\sqrt{\epsilon}\,\sigma_1$ = App. D row | `sqrt(12)*[0.0, 5.7, 0.0]` | **calibrated** |
| $\sigma_2$ | $\sqrt{12}\,(0,\ 0,\ 0.00031)$ (appendix cell) | $\sqrt{\epsilon}\,\sigma_2$ = App. D row | `sqrt(12)*[0.0, 0.0, 0.00031]` | **calibrated** |
| $\nu_h$ | $0.025$ per quarter (App. D sentence; appendix cell as $0.1\,\epsilon$) | none | 0.025 | **calibrated** |
| $\gamma$ | figures at $\gamma = 8$ (chapter habit figure captions) | none | 1.001 anchor $\to$ 8.0 target | target **calibrated**; anchor **convention** |
| $\rho$ | chapter sets $\rho = 1$ throughout this section | unit-EIS convention | 1.001 | **convention** |
| $\lambda$ | figures at $\lambda = .67$ (chapter habit figure captions) | none | $-0.0$ (i.e. $0$, habit off) anchor $\to$ 0.67 target | target **calibrated**; anchor is the appendix's own start |
| $\tau$ | chapter text: strongest responses at $\tau = .01$ | $\tau = 1.01 \approx$ Cobb–Douglas $\tau = 1$ at the anchor | 1.01 anchor $\to$ 0.01 target | anchor **convention**; target **calibrated** |

**Anchors.** No independent paper anchor exists — nothing analogous to the AK closed form $D^{2*}$ is printed for the habit extension. The checks are: (i) the driver's residual gate: the returned $\mathsf q = 0$ steady state must satisfy the compiled equations to the driver's tolerance of $10^{-6}$ (the recorded HABIT residual is $1.1\times 10^{-11}$) — model-internal; (ii) the steady-state habit-ratio identity implied by the $X$ law, $e^{X^*}\!\left(e^{g^*} - e^{-\nu_h}\right) = \left(1 - e^{-\nu_h}\right) i_h^*$ with $g^*$ the steady capital growth and $i_h^*$ the steady $I_h/K$ — a **consistency check**, not independent; (iii) agreement with the appendix notebook's own stored solution, verified earlier to machine precision — **book replication**, book-internal by construction.

**Limitations / restrictions.** We solve only the external-habit variant (the appendix's Section 2.6 flag); the internal version is not run here. $\rho = 1.001$ rather than the chapter's $\rho = 1$, and $\tau = 1.01$ at the anchor — both conventions; the driver continues $(\gamma, \lambda, \tau)$ from the appendix's own starting configuration $(1.001,\ 0,\ 1.01)$ to the chapter-figure configuration $(8,\ 0.67,\ 0.01)$. The $\gamma$ target $8.0$ is the chapter-caption value; the appendix notebook's own external-habit solve loops instead use $\gamma \in \{1.001,\ 4.001,\ 8.001\}$ — a $0.001$ offset we do not reproduce. This model keeps the appendix's annualized-units-with-$\epsilon$ statement, so its stored parameter values differ by $\epsilon$-factors from the other five models, which are stated per quarter. One source discrepancy, flagged for honesty: App. D prints $\delta = 0.025$, while the appendix notebook uses $\delta = 0.01$/yr giving $\beta = e^{-0.0025}$ per quarter; we follow the notebook (this also matches the AK entry).

**Solve status: solves cold — automatic guess alone (about 12 s), reaching the habit target through the driver; no seeds exist for this model. Anchor type: book replication, no independent paper anchor.**'''))

cells.append(md(r'''### KL — Kaltenbrunner & Lochstoer (*RFS* 2010), permanent-shock benchmark "LRR II"

**Source.** Kaltenbrunner, G., and L. A. Lochstoer, "Long-Run Risk through Consumption Smoothing," *Review of Financial Studies* 23(8), 2010, pp. 3190–3224. Version solved: the permanent-shock benchmark **LRR II** — random-walk technology ($\varphi = 1$ in their Eq. (9)) with EIS $\psi = 1.5$, their Table 4 (p. 3210). Native frequency: quarterly ("one unit of time in the model corresponds to a quarter of a year," p. 3196) — no frequency conversion anywhere.

**Economic meaning.** A one-sector stochastic-growth economy with Epstein–Zin preferences in which long-run consumption risk is *endogenous*: technology growth is i.i.d., yet optimal consumption smoothing produces a highly persistent expected-consumption-growth component, so the price of risk is high at a low risk aversion ($\gamma = 5$). It sits in this lineup as the minimal production long-run-risk economy — one shock, one endogenous ratio state, closed-form deterministic steady state — the cleanest test of the engine's share-form constraint class.

**Variable dictionary** (their Section 1, pp. 3194–3196):

| paper symbol | engine symbol | meaning | type | units / frequency |
|---|---|---|---|---|
| $K_t$ | — (scaled out) | capital stock | state | level, quarterly |
| $Z_t = e^{\mu t + z_t}$ | — (scaled out) | labor-enhancing technology (the paper's term, p. 3194); $z_t$ its stochastic log component, a random walk at $\varphi = 1$ | state | level, quarterly |
| $\log(Z_t/K_t)$ | `w_t` ($\omega_t$) | the single stationary ratio state (our change of variables; the paper gives this ratio no symbol — its own $\omega$ denotes wages, p. 3195) | state | log ratio, quarterly |
| $H_t$ | — | hours worked; no disutility of labor, so $H_t \equiv 1$ (their Section 1.2, p. 3194) | parameter | — |
| $Y_t$ | — | aggregate output; eliminated by the share change of variables | derived | level, quarterly |
| $C_t$ | `cs_t` ($c^s_t = C_t/Y_t$) | consumption; the engine control is its output share | control | share of $Y_t$ |
| $I_t = Y_t - C_t$ | `is_t` ($i^s_t = I_t/Y_t$) | gross investment (p. 3195); the engine control is its output share | control | share of $Y_t$ |
| $\varepsilon_t \sim N(0, \sigma^2)$ | $\sigma\,$`W1_tp1` | technology shock; the engine shock is standard normal | shock | quarterly |
| $\mu,\ \sigma$ | `mu`, `sigma` | mean and volatility of log technology growth | parameter | quarterly |
| $\varphi$ | — | technology-shock persistence, their Eq. (9); $\varphi = 1$ (permanent) in LRR II | parameter | — |
| $\alpha$ | `alpha` | capital share | parameter | — |
| $\delta$ | `delta` | capital depreciation rate | parameter | quarterly |
| $\phi(\cdot)$ | — | weakly concave installation function, their Eq. (5) | derived | — |
| $\xi$ | `xi` | elasticity of the investment rate to Tobin's q; adjustment costs vanish as $\xi \to \infty$ | parameter | — |
| $a_1,\ a_2$ | `a1`, `a2` | installation constants: no adjustment cost at the steady state (their fn. 4) | derived constant | — |
| $\beta$ | `beta` | subjective discount factor | parameter | quarterly |
| $\gamma$ | `gamma` | coefficient of relative risk aversion | parameter | — |
| $\psi$ | `rho` $= 1/\psi$ | elasticity of intertemporal substitution; the engine uses the inverse EIS $\rho$ | parameter | — |

**Equations (paper notation).** Preferences are Epstein–Zin over $C_t$ with $(\beta, \gamma, \psi)$ — their Eq. (1), p. 3194 — supplied to the engine's recursive-utility layer as $(\beta, \gamma, \rho = 1/\psi)$, not re-declared per model. Production, their Eq. (3), p. 3194, with $H_t \equiv 1$, and the resource constraint $I_t = Y_t - C_t$ (p. 3195):

$$Y_t = (Z_t H_t)^{1-\alpha} K_t^{\alpha}, \qquad C_t + I_t = Y_t$$

Capital accumulation and installation, their Eqs. (4)–(5), p. 3195:

$$K_{t+1} = (1-\delta)\,K_t + \phi\!\left(I_t/K_t\right) K_t, \qquad \phi(X) = a_1 + \frac{a_2}{1 - 1/\xi}\, X^{1-1/\xi}$$

with their footnote 4 (p. 3195) normalization pinning $a_1, a_2$ so that adjustment costs vanish at the nonstochastic steady state:

$$a_2 = (e^{\mu} - 1 + \delta)^{1/\xi}, \qquad a_1 = \tfrac{1}{\xi-1}\,(1 - \delta - e^{\mu}) \;\;\Rightarrow\;\; \phi(\overline{I/K}) = \overline{I/K}, \;\; \phi^{\prime}(\overline{I/K}) = 1 \;\text{ at } \overline{I/K} = e^{\mu} - 1 + \delta$$

Technology, their Eqs. (8)–(9), p. 3196; at $\varphi = 1$ the two collapse to a random walk with drift:

$$Z_t = \exp(\mu t + z_t), \qquad z_t = \varphi z_{t-1} + \varepsilon_t \qquad \Longrightarrow \qquad \log Z_{t+1} - \log Z_t = \mu + \varepsilon_{t+1}$$

**Mapping to the engine form.** Capital enters the resource constraint through $Y_t(Z_t, K_t)$, so the constraint must be stated in output shares to fit the engine's state-free constraint class (see the Model note above). Divide the constraint by $Y_t$ and scale levels by $K_t$: with $\omega_t = \log(Z_t/K_t)$ we get $Y_t/K_t = e^{(1-\alpha)\omega_t}$ and $I_t/K_t = i^s_t\, e^{(1-\alpha)\omega_t}$, and the engine blocks are

$$\kappa_t = \log c^s_t + (1-\alpha)\,\omega_t \;\;\big(= \log(C_t/K_t)\big), \qquad \psi^g_t = \log\!\left(1 - \delta + \phi\!\left(i^s_t\, e^{(1-\alpha)\omega_t}\right)\right) \;\;\big(= \log(K_{t+1}/K_t)\big)$$

$$\psi^x:\;\; \omega_{t+1} = \omega_t + \mu + \sigma W_{t+1} - \psi^g_t, \qquad \phi:\;\; 1 - c^s_t - i^s_t = 0$$

Notation guard: $\phi(\cdot)$ *with an argument* is always the paper's installation function; the bare $\phi$ is the engine's constraint block. The paper's own $\omega$ denotes wages (p. 3195), which never appear in this section; our $\omega_t$ is only the ratio state defined above. $W_{t+1}$ is the engine's standard-normal shock, so $\varepsilon_{t+1} = \sigma W_{t+1}$.

**Parameters** (all native quarterly):

| parameter | paper value (native freq) | conversion rule | engine value (quarterly) | status |
|---|---|---|---|---|
| $\alpha$ | 0.36 (Table 1, p. 3197) | none | 0.36 | **calibrated** |
| $\delta$ | 0.021 (Table 1, p. 3197) | none | 0.021 | **calibrated** |
| $\mu$ | 0.4% (Table 1, p. 3197) | percent $\to$ decimal | 0.004 | **calibrated** |
| $\sigma$ | 4.11% (Table 4, p. 3210, LRR II column; Table 1, p. 3197, prints $\sigma = 4.1\%$ among parameters "constant across models," but Table 4's model-specific values — 4.05%, 4.05%, 1.61%, 4.11% — supersede it) | percent $\to$ decimal | 0.0411 | **calibrated** |
| $\gamma$ | 5 (Table 1, p. 3197) | none | 5.0 | **calibrated** |
| $\beta$ | 0.998 (Table 4, p. 3210, LRR II) | none | 0.998 | **calibrated** |
| $\xi$ | 18.0 (Table 4, p. 3210, LRR II) | none | 18.0 | **calibrated** |
| $\psi$ | 1.5 (Table 4, p. 3210, LRR II) | $\rho = 1/\psi$ | $\rho = 1/1.5$ | **derived** |
| $a_1$ | — | fn. 4, p. 3195: $(1-\delta-e^{\mu})/(\xi-1)$ | $-1.4711 \times 10^{-3}$ | **derived** |
| $a_2$ | — | fn. 4, p. 3195: $(e^{\mu}-1+\delta)^{1/\xi}$ | $0.81471$ | **derived** |
| solve-path start $\gamma = \rho$ | — | homotopy anchor near log utility | 1.001 | **convention** |

**Anchors.** The deterministic steady state is available in closed form from the paper's own equations; all three anchors are *independent closed forms* — derived from the paper, using no engine output:

1. balanced growth (p. 3196: all endogenous variables grow at the technology rate): $\psi^{g*} = \mu = 0.004$;
2. their fn. 4, p. 3195: $\overline{I/K} = e^{\mu} - 1 + \delta = 0.025008$;
3. the deterministic Euler equation — their Eqs. (2), (6)–(7) at the steady state, where fn. 4 gives $\phi^{\prime} = 1$ — reduces to $\alpha\, \overline{Y/K} = e^{\rho\mu}/\beta - (1-\delta)$, a closed form the paper does not print (it is our reduction of their equations); hence

$$\omega^{*} = \frac{1}{1-\alpha}\, \log\!\left[\frac{e^{\rho\mu}/\beta - (1-\delta)}{\alpha}\right] = -4.1256, \qquad i^{s*} = \left(e^{\mu} - 1 + \delta\right) e^{-(1-\alpha)\omega^{*}} = 0.3506, \qquad c^{s*} = 0.6494$$

These closed forms play two roles, and the distinction matters for what each run proves: they are the optional seed, so a *seeded* solve that lands on them is only a consistency check (the solver started there); they are also the acceptance test for the *cold* (unseeded) solve, and a cold solve that lands on them is an independent check. We run both.

**Limitations / restrictions.** (i) Fixed labor $H_t \equiv 1$ is the paper's own benchmark (Section 1.2: no disutility of labor), not our restriction. (ii) We solve only LRR II; LRR I ($\varphi = 0.95$, transitory shocks) would add $z_t$ as a second state and is not in this lineup. (iii) The paper solves globally by value-function iteration (p. 3196); the engine is a small-noise expansion around the deterministic steady state, so we verify the steady-state anchors above, not their simulated moments. (iv) The $\gamma = \rho = 1.001$ start is a solver convention only; the economy reported is at the published target $\gamma = 5$, $\psi = 1.5$.

**Solve status: solves cold — automatic guess alone (about 27 s; the ratio state $\omega$ is unpinned and multi-start finds it); the paper closed-form seed is optional and cuts this to about 3 s. Cold-solve agreement with the closed forms is an independent check; the seeded solve is a consistency check.**'''))

cells.append(md(r'''### ACL — Ai, Croce & Li (*RFS* 2013), Extension 1 (intangible-capital adjustment costs)

**Source.** Ai, Croce & Li, "Toward a Quantitative General Equilibrium Asset Pricing Model with Intangible Capital", *Review of Financial Studies* 2013. We solve **Extension 1** — their Section V.A, Jermann adjustment costs on the production of new blueprints — calibrated in their Table C.2, column "Extension 1"; **native frequency: annual**. Working parameter values and the closed-form steady-state chain come from the companion implementation in Borovička & Hansen, *Journal of Econometrics* 183 (2014) 67–90 (Section 7 and Appendix C Table 1, p. 89; they cite the paper as Ai et al. 2012, the advance-access version), who re-solved exactly this version in Dynare (their Section 7.2; code linked in their footnote 15). ACL page references below are to the manuscript version in our source folder; the journal pagination differs.

**Economics.** Two capital stocks: tangible capital $K$ (assets in place) and intangible capital $S$ (blueprints, i.e. growth options). New production units are assembled from tangible investment and blueprints through a CES aggregator, and the newest vintage is shielded from the current productivity shock by a vintage wedge — so intangible capital carries little long-run productivity risk while tangible capital carries a lot, which is the model's engine for the value premium. It is in this lineup as the hardest case: two capitals, three states, two shocks, and a steady state the solver cannot find unaided (see the solve status below).

| paper symbol | engine symbol | meaning | type | units / frequency |
|---|---|---|---|---|
| $K_t,\ A_t,\ S_t$ | — | trending levels: tangible capital (generation-0 equivalents), generation-0 labor productivity, blueprint stock; enter the engine only through the two log ratios below | state | levels, quarterly |
| $\omega^a_t = \log(A_t/K_t)$ | `wa_t` | productivity relative to tangible capital | state | log ratio, quarterly |
| $s_t = \log(S_t/K_t)$ | `s_t` | intangible relative to tangible capital | state | log ratio, quarterly |
| $x_t$ | `x_t` | long-run-risk component of productivity growth | state | log growth, quarterly |
| $C_t$ | `cs_t` | consumption; engine control is the output share $c^s_t = C_t/Y_t$ | control | share of $Y_t$ |
| $I_t$ | `is_t` | tangible investment; $i^s_t = I_t/Y_t$ | control | share of $Y_t$ |
| $J_t$ | `js_t` | intangible investment (new blueprints); $j^s_t = J_t/Y_t$ | control | share of $Y_t$ |
| $\varepsilon_{a,t+1}$ | $W_{1,t+1}$ | short-run (direct) productivity shock | shock | i.i.d. $N(0,1)$ |
| $\varepsilon_{x,t+1}$ | $W_{2,t+1}$ | long-run-risk shock | shock | i.i.d. $N(0,1)$ |
| $Y_t$ | — | output $K_t^{\alpha}(A_tN_t)^{1-\alpha}$; per unit of $K$: $Y_t/K_t = e^{(1-\alpha)\omega^a_t}$ | derived | trending level |
| $N_t$ | — | labor input, fixed at 1 in Extension 1 (endogenous labor is their Ext. 2) | parameter | — |
| $G(I_t,S_t)$ | — | CES aggregator = measure $M_t$ of new production units | derived | flow |
| $H(J_t,K_t)$ | — | new-blueprint production, Jermann form | derived | flow |
| $\varpi_{t+1}$ | — | vintage wedge on newly built capital | derived | multiplicative factor |
| $\alpha$ | `alpha` | capital share | parameter | unit-free |
| $\delta_K,\ \delta_S$ | `dK`, `dS` | depreciation, tangible / intangible | parameter | per quarter |
| $\nu,\ \eta$ | `nu`, `eta` | CES weight on $I$ and elasticity of substitution in $G$ | parameter | unit-free |
| $\xi$ | `xi` | Jermann curvature in $H$ | parameter | unit-free |
| $a_1,\ a_2$ | `a1`, `a2` | normalization constants of $H$ | derived constant | frequency-dependent |
| $\mu$ | `mu` | mean productivity growth | parameter | per quarter |
| $\sigma_a,\ \sigma_x$ | `sig_a`, `sig_x` | volatilities of the two shocks | parameter | per quarter |
| $\rho$ (their AR coefficient) | `rho_x` | persistence of $x_t$ — **not** the preference parameter $\rho$ | parameter | per quarter |
| $\beta,\ \gamma,\ \psi$ | `beta`, `gamma`, `rho` $=1/\psi$ | Epstein–Zin discount factor, risk aversion, IES; the chapter's $\rho$ is the inverse IES | parameter | $\beta$ per quarter; $\gamma,\psi$ unit-free |

Their Eq. (5), ms. p. 11, and Eq. (10), ms. p. 12 — technology and resources (Extension 1 fixes $N_t = 1$):

$$Y_t = K_t^{\alpha}(A_t N_t)^{1-\alpha}, \qquad C_t + I_t + J_t \le Y_t.$$

Their Eq. (21), ms. p. 24 (CES aggregator), Eq. (9), ms. p. 12 (tangible accumulation), and Eq. (3), ms. p. 8 (the wedge):

$$G(I,S) = \left(\nu I^{1-1/\eta} + (1-\nu)S^{1-1/\eta}\right)^{\frac{1}{1-1/\eta}}, \qquad K_{t+1} = (1-\delta_K)K_t + \varpi_{t+1}\,G(I_t,S_t), \qquad \varpi_{t+1} = e^{-\frac{1-\alpha}{\alpha}(x_t + \sigma_a\varepsilon_{a,t+1})}.$$

Their Eq. (29), ms. p. 41 — the equation that defines Extension 1 — with $H$ parameterized in the unnumbered display just below it, and Eq. (1), ms. p. 7 (productivity):

$$S_{t+1} = (1-\delta_S)\left(S_t - G(I_t,S_t)\right) + H(J_t,K_t), \qquad H(J,K) = \left[\tfrac{a_1}{1-1/\xi}\left(\tfrac{J}{K}\right)^{1-1/\xi} + a_2\right]K,$$

$$\Delta a_{t+1} \equiv \log(A_{t+1}/A_t) = \mu + x_t + \sigma_a\varepsilon_{a,t+1}, \qquad x_{t+1} = \rho\,x_t + \sigma_x\varepsilon_{x,t+1}.$$

$a_1, a_2$ are pinned down by the two steady-state conditions stated in Section V.A, ms. p. 41: $H(J,K) = J$ and $H_J(J,K) = 1$. (Subscripts on $G$ and $H$ denote partial derivatives, following the paper's own convention on ms. p. 41.)

**Mapping to the engine form.** Controls are the three output shares with $\phi:\ c^s_t + i^s_t + j^s_t = 1$ (Eq. (10) at equality); states are $\omega^a_t, s_t, x_t$. Since $G$ is homogeneous of degree 1, everything divides by $K_t$: $I_t/K_t = i^s_t\,e^{(1-\alpha)\omega^a_t}$, $J_t/K_t = j^s_t\,e^{(1-\alpha)\omega^a_t}$, $S_t/K_t = e^{s_t}$, and

$$\kappa_t = \log c^s_t + (1-\alpha)\,\omega^a_t \;\left(= \log(C_t/K_t)\right), \qquad \psi^g = \log\!\left(1 - \delta_K + \varpi_{t+1}\,G\!\left(I_t/K_t,\ e^{s_t}\right)\right),$$

$$\psi^x:\quad \omega^a_{t+1} = \omega^a_t + \mu + x_t + \sigma_a W_{1,t+1} - \psi^g, \qquad s_{t+1} = \log\!\left((1-\delta_S)\!\left(e^{s_t} - \tfrac{G_t}{K_t}\right) + \tfrac{H_t}{K_t}\right) - \psi^g, \qquad x_{t+1} = \rho_x x_t + \sigma_x W_{2,t+1},$$

with $G_t \equiv G(I_t,S_t)$, $H_t \equiv H(J_t,K_t)$, and $H_t/K_t = \tfrac{a_1}{1-1/\xi}(J_t/K_t)^{1-1/\xi} + a_2$.

| parameter | paper value (native freq) | conversion rule | engine value (quarterly) | status |
|---|---|---|---|---|
| $\beta$ | 0.97 annual (Table C.2, ms. p. 60); BH working value 0.971 (BH Table 1, p. 89) | $\beta_q = \beta_a^{1/4}$ | $0.971^{1/4} = 0.992670$ | **calibrated** |
| $\gamma$ | 10 (Table C.2) | unit-free | 10 (continuation target; the solve starts at 1.001) | **calibrated** |
| $\psi$ | 2.0 (Table C.2) | $\rho = 1/\psi$, unit-free | `rho` $= 0.5$ | **derived** |
| $\alpha$ | 0.3 (Table C.2) | unit-free | 0.3 | **calibrated** |
| $\delta_K = \delta_S$ | 11% annual (Table C.2) | $(1-\delta)_q = (1-\delta_a)^{1/4}$ | 0.0287132 | **calibrated** |
| $\nu$ | 0.88 (Table C.2) | unit-free | 0.88 | **calibrated** |
| $\eta$ | 2.50 (Table C.2) | unit-free | 2.5 | **calibrated** |
| $\xi$ | 5 (Table C.2) | unit-free | 5.0 | **calibrated** |
| $\mu$ | 2.0% annual (Table C.2) | $\mu_q = \mu_a/4$ | 0.005 | **calibrated** |
| $\sigma_a$ | 5.08% annual (Table C.2) | i.i.d. part: $\sigma_{a,q} = \sigma_{a,\text{ann}}/2$ | 0.0254 | **calibrated** |
| $\rho$ (AR of $x$) | 0.925 (Table C.2) | $\rho_{x,q} = \rho_a^{1/4}$ | 0.980698 | **calibrated** |
| $\sigma_x$ | 0.86% (Table C.2); BH working value 0.8636% | same unconditional variance of $x$: $\sigma_{x,q} = \sigma_{x,a}\sqrt{\tfrac{1-\rho_a^{1/2}}{1-\rho_a^{2}}}$ | 0.0044440 | **calibrated** |
| $\varphi_0, \varphi_1$ | 0, 1 (Table C.2) | none — already embodied in the functional form of $\varpi$, Eq. (3) | — | **calibrated** |
| $a_1$ | 0.6645 annual (BH Table 1, p. 89) | recomputed at the quarterly parameters from $H = J$, $H_J = 1$ | 0.535473 | **derived** |
| $a_2$ | $-0.0324$ annual (BH Table 1, p. 89) | same | $-0.0110059$ | **derived** |

Status refers to the native value; every quarterly entry follows mechanically from its stated rule. Table C.2 prints $\beta$ and $\sigma_x$ rounded (0.97, 0.86%); we use the BH working values 0.971 and 0.008636 — with $\beta = 0.971$ (not 0.97) the normalization below reproduces the BH printed $a_1, a_2$ exactly.

**Anchors.** The cold solve reaches these values without them (an independent check); a seeded solve is a consistency check. At the deterministic steady state $\varpi = 1$, $H_J = 1$ so the blueprint price is $q_S = 1/H_J = 1$ (their modified conditions, ms. p. 41), and — a by-product of the normalization — $H_K = 0$, so the $H_{K,t}\,q_{S,t}$ term in the Extension-1 modified $p_{K,t}$ (ms. p. 41) vanishes and the baseline tangible-capital pricing applies at the steady state. The BH Dynare file solves the entire steady state in closed form; the chain is frequency-generic (evaluated at the annual parameters it reproduces BH Table 1's $a_1 = 0.6645$, $a_2 = -0.0324$ to all printed digits). With $\bar G/\bar S$ and $G_I$ evaluated from Eq. (21) at $\bar I/\bar S$:

$$\frac{\bar I}{\bar S} = \left[\frac{\nu}{1-\nu}\left(\frac{e^{\rho\mu}}{\beta} - 1 + \delta_S\right)\right]^{\eta}\ \text{(their intangible return, ms. p. 41, at } \bar r_S = e^{\rho\mu}/\beta\text{)}; \qquad \frac{\bar J}{\bar S} = e^{\mu} - (1-\delta_S)\left(1 - \frac{\bar G}{\bar S}\right)\ \text{(ss of Eq. (29) with } H = J\text{)};$$

$$\bar q_K = 1 - \delta_S + \frac{1}{G_I}\ \text{(ss of the exercise margin, their Eq. (13), ms. p. 15, at } \bar q_S = 1\text{)}; \qquad \alpha\left(\frac{\bar K}{\bar A}\right)^{\alpha-1} = \bar q_K\left(\frac{e^{\rho\mu}}{\beta} - 1 + \delta_K\right)\ \text{(ss of their Eq. (15), ms. p. 16, at } \bar r_K = e^{\rho\mu}/\beta\text{)}; \qquad \frac{\bar G}{\bar K} = e^{\mu} - 1 + \delta_K\ \text{(ss of Eq. (9))},$$

which then delivers $\bar S/\bar K$, $\bar J/\bar K$, and finally $a_1 = (\bar J/\bar K)^{1/\xi}$, $a_2 = -(\bar J/\bar K)/(\xi-1)$. The engine steady state — the three shares, $\omega^{a*}$, $s^*$, and the growth rate — matches this chain to about $10^{-10}$. **But the same chain supplies the solver's seeds** ($\omega^{a*} = -\log(\bar K/\bar A)$, $s^* = \log(\bar S/\bar K)$), so agreement confirms the implementation is internally consistent with the Borovička–Hansen algebra; it is not a closed form the solver found on its own.

**Limitations / restrictions.** (i) We solve the aggregate planner formulation as written by BH; the cross-sectional content of the paper (vintage portfolios, book-to-market sorts) is outside this representation. (ii) Fixed labor $N_t = 1$ is Extension 1 itself, not our restriction. (iii) The annual-to-quarterly conversion is ours — both papers calibrate annually, the lineup runs quarterly — and $a_1, a_2$ are therefore recomputed at the quarterly parameters, not converted. (iv) $\beta = 0.971$ and $\sigma_x = 0.8636\%$ are BH working values rather than the rounded Table C.2 prints. (v) $\gamma = 10$ is reached by the engine's continuation from 1.001.

**Solve status: solves cold in about six minutes — the automatic guess alone, once the initialization reads the balanced growth rate from the declared trends; the paper-derived closed-form seeds remain an optional accelerator (about 70 s). Cold agreement with the Borovička–Hansen steady-state chain is an independent check; a seeded solve is a consistency check.**'''))

cells.append(md(r'''### CROCE — Croce (*JME* 2014), fixed-labor benchmark (2008 working-paper version)

**Source.** Croce, "Long-Run Productivity Risk: A New Hope for Production-Based Asset Pricing?", published in the *Journal of Monetary Economics* 66 (2014), 13–31. The version solved here is the **2008 working paper** (draft dated January 15, 2008, circulated in the IGIER working-paper series): its Section 3 benchmark is a consumption-only, fixed-labor production economy, calibrated at a native **monthly** frequency (Table 3, Panel A, p. 37). To make the version choice impossible to misread: we solve the 2008 working-paper benchmark and use nothing from the 2014 publication, because the 2014 baseline puts leisure inside the utility bundle — an elastic labor choice, outside the engine class of a single intratemporal constraint on the output shares — and recalibrates it (risk aversion 10 and adjustment-cost elasticity 7 there, versus 30 and 0.98 here; its Section 3, pp. 20–22).

**Economics.** This is the canonical one-capital long-run-risk production economy: productivity growth carries a small, highly persistent predictable component, and an Epstein–Zin agent who fears news about that component prices capital accordingly. Combined with Jermann-type adjustment costs, long-run productivity risk delivers a sizable equity premium (4.8% annualized in his Table 3) at risk aversion 30 — the result that made production-based long-run risk viable. In the lineup it is the minimal long-run-risk member: two shocks, two states, one constraint.

**Variable dictionary.** Paper variables are monthly levels; engine variables are quarterly shares and log ratios.

| paper symbol | engine symbol | meaning | type | units / frequency |
|---|---|---|---|---|
| $C_t$ | `cs_t` $= C_t/Y_t$ | consumption; engine carries the output share | control | level, monthly (paper); share, unit-free, quarterly (engine) |
| $I_t$ | `is_t` $= I_t/Y_t$ | investment; engine carries the output share | control | level, monthly (paper); share, unit-free, quarterly (engine) |
| $Y_t$ | eliminated: $Y_t/K_t = e^{(1-\alpha)\omega^a_t}\,\bar n^{\,1-\alpha}$ | output | derived | level |
| $K_t$ | — (numeraire: engine variables are ratios to $K_t$) | capital stock | state | level |
| $A_t$ | `wa_t` $= \omega^a_t \equiv \log(A_t/K_t)$ | productivity level | state | log ratio, unit-free, quarterly |
| $x_t$ | `x_t` | long-run component of expected productivity growth | state | log growth per month (paper) / per quarter (engine) |
| $\varepsilon_{a,t+1}$ | `W1_tp1` | short-run productivity shock | shock | $N(0,1)$ per period |
| $\varepsilon_{x,t+1}$ | `W2_tp1` | long-run (expected-growth) shock | shock | $N(0,1)$ per period |
| $U_t$ | handled internally by the engine | continuation utility | derived | — |
| $n_t,\ \bar n$ | `nbar` | labor input, at the corner $n_t=\bar n$; leisure $l_t=\bar n-n_t$ never enters $U_t$ | parameter | share of time endowment |
| $\mu$ | `mu` | mean productivity growth | parameter | per month → per quarter |
| $\sigma$ | `sig_a` | short-run volatility of $\Delta a$ | parameter | per month → per quarter |
| $\rho$ | `rho_x` | persistence of $x_t$ (his $\rho$ is NOT the preference $\rho$ below) | parameter | per month → per quarter |
| $\sigma_x$ | `sig_x` | volatility of the long-run shock | parameter | per month → per quarter |
| $\delta$ | `beta` | subjective discount factor (his Table prints the annualized $\delta^{12}$) | parameter | per month → per quarter |
| $\Psi$ | `rho` $= 1/\Psi$ | intertemporal elasticity of substitution; engine solves in the book inverse-EIS $\rho$ | parameter | unit-free |
| $\gamma$ | `gamma` | relative risk aversion | parameter | unit-free |
| $\alpha$ | `alpha` | capital share | parameter | unit-free |
| $\delta_k$ | `dk` | capital depreciation rate | parameter | per month → per quarter |
| $\tau$ | `tau` | elasticity of the adjustment-cost (capital-supply) function | parameter | unit-free |
| $G(\cdot)$ | built into $\psi^g$ | Jermann installation technology | derived | per period |
| $a_1, a_2$ | `a1`, `a2` | adjustment-cost constants, pinned at $\bar x$ | derived constant | quarterly normalization |
| $\bar x$ | — | normalization point $e^{\mu}-1+\delta_k$ (quarterly values) | derived constant | quarterly |

**Equations (paper notation).** Every display used below is unnumbered in the working paper; the only numbered equation nearby is the stochastic discount factor, their Eq. (7), p. 13, which is not needed to state the planner problem. Preferences, technology, and the productivity process (Section 3, p. 11, where the paper defines $\Delta a_{t+1}\equiv\log(A_{t+1}/A_t)$):

$$U_t=\left[(1-\delta)\,C_t^{1-\frac{1}{\Psi}}+\delta\left(E_t\!\left[U_{t+1}^{1-\gamma}\right]\right)^{\frac{1-1/\Psi}{1-\gamma}}\right]^{\frac{1}{1-1/\Psi}}, \qquad Y_t=K_t^{\alpha}\left[A_t n_t\right]^{1-\alpha}$$

$$\Delta a_{t+1}=\mu+x_t+\sigma\,\varepsilon_{a,t+1}, \qquad x_t=\rho\,x_{t-1}+\sigma_x\,\varepsilon_{x,t}, \qquad (\varepsilon_{a},\varepsilon_{x})\sim \text{iid } N(0,I_2)$$

Resource constraint and capital accumulation (p. 12):

$$C_t+I_t\le Y_t, \qquad K_{t+1}=(1-\delta_k)K_t+G\!\left(\tfrac{I_t}{K_t}\right)K_t, \qquad G\!\left(\tfrac{I_t}{K_t}\right)=\left[\frac{a_1}{1-\frac{1}{\tau}}\left(\tfrac{I_t}{K_t}\right)^{1-\frac{1}{\tau}}+a_2\right]$$

Labor (p. 12): the agent is endowed with $\bar n$ units of time, $n_t+l_t\le\bar n$; since leisure does not appear in the utility function, offering $n_t=\bar n$ is always optimal — his statement, not ours — and his footnote 14 (p. 12) imposes $\bar n=0.18$ (total employment times average weekly hours over the civilian non-institutional population 16+, as in Tallarini 2000). **Fixed labor is a property of the working-paper model itself, not a restriction we add.**

**Mapping to the engine form.** Divide by $K_t$ and pass to shares: states $\omega^a_t=\log(A_t/K_t)$ and $x_t$; controls $c^s_t=C_t/Y_t$, $i^s_t=I_t/Y_t$; then with $Y_t/K_t=e^{(1-\alpha)\omega^a_t}\bar n^{\,1-\alpha}$, the book's log consumption–capital ratio $\kappa_t=\log(C_t/K_t)$ and log capital growth $\psi^g_{t+1}=\log(K_{t+1}/K_t)$ become

$$\kappa_t=\log c^s_t+(1-\alpha)\left(\omega^a_t+\log\bar n\right), \qquad \psi^g_{t+1}=\log\!\left(1-\delta_k+G\!\left(i^s_t\,e^{(1-\alpha)\omega^a_t}\bar n^{\,1-\alpha}\right)\right)$$

$$\omega^a_{t+1}=\omega^a_t+\mu+x_t+\sigma W^1_{t+1}-\psi^g_{t+1}, \qquad x_{t+1}=\rho_x x_t+\sigma_x W^2_{t+1}, \qquad \phi_t = 1-c^s_t-i^s_t=0,$$

where all parameters take their quarterly engine values and $\rho_x$ is the persistence (the preference $\rho=1/\Psi$ keeps the book meaning).

**Parameters.** Monthly values from Table 3, Panel A, p. 37 (headed $\mu,\ \sigma,\ \sigma_x,\ \rho,\ \delta_k,\ \alpha,\ \tau,\ \delta^{12},\ \gamma,\ \Psi$); one quarter = three months.

| parameter | paper value (native freq) | conversion rule | engine value (quarterly) | status |
|---|---|---|---|---|
| $\mu$ | $0.165\%$ per month | $\mu_q=3\mu$ | $0.00495$ | **calibrated** (T3A p. 37) |
| $\sigma$ | $0.60\%$ per month | $\sigma_q=\sigma\sqrt{3}$ (three iid monthly shocks) | $0.010392$ | **calibrated** (T3A p. 37) |
| $\sigma_x$ | $5.5\%\,\sigma$ per month | $\sigma_{x,q}=\sigma_x\sqrt{(1-\rho^6)/(1-\rho^2)}$ — exact 3-month skip-sample of the AR(1), equivalently unconditional-variance matching | $0.000560$ | **calibrated** (T3A p. 37) |
| $\rho$ | $0.98$ per month | $\rho_{x,q}=\rho^3$ | $0.941192$ | **calibrated** (T3A p. 37) |
| $\delta_k$ | $0.5\%$ per month | $1-\delta_{k,q}=(1-\delta_k)^3$ | $0.014925$ | **calibrated** (T3A p. 37) |
| $\alpha$ | $0.33$ | unit-free | $0.33$ | **calibrated** (T3A p. 37) |
| $\tau$ | $0.98$ | unit-free | $0.98$ | **calibrated** (T3A p. 37) |
| $\delta$ | $\delta^{12}=0.98$ (annualized) | $\beta_q=(\delta^{12})^{1/4}$ | $0.98^{1/4}=0.994962$ | **calibrated** (T3A p. 37) |
| $\gamma$ | $30$ | unit-free | $30.0$ | **calibrated** (T3A p. 37) |
| $\Psi$ | $2$ | engine solves in $\rho=1/\Psi$ | $0.5$ | **calibrated** (T3A p. 37) |
| $\bar n$ | $0.18$ | level, unchanged | $0.18$ | **calibrated** (fn. 14, p. 12) |
| $a_1$ | not printed in the WP | $a_1=\bar x^{\,1/\tau}$ at $\bar x=e^{\mu_q}-1+\delta_{k,q}=0.019887$ | $0.018359$ | **derived** (Jermann normalization) |
| $a_2$ | not printed in the WP | $a_2=-\bar x/(\tau-1)$ | $0.994370$ | **derived** (Jermann normalization) |

The Jermann normalization sets $G(\bar x)=\bar x$ and $G^{\prime}(\bar x)=1$ — no adjustment costs on the deterministic balanced path — which pins $a_1,a_2$ as above; the WP states only that $G$ follows Jermann (1998), and the same rule (with $a_1,a_2$ named $b,c$) appears in the author replication file for the 2014 version.

**Anchors** (deterministic balanced path, $x=0$, shocks off):
1. **Growth:** $\psi^g=\mu_q=0.00495$ — independent closed form (balanced growth of any correct solution).
2. **Investment rate:** $I/K=\bar x=e^{\mu_q}-1+\delta_{k,q}=0.019887$ — a consistency check of the normalization ($a_1,a_2$ are constructed to make $G(\bar x)=\bar x$), not an independent restriction.
3. **Euler equation:** with $q_t=1/G^{\prime}(I_t/K_t)$ (his p. 13) equal to 1 at $\bar x$, the deterministic Euler equation gives $\alpha\,Y/K=e^{\mu_q/\Psi}/\beta_q-(1-\delta_{k,q})=0.022479$, hence $Y/K=0.068119$, $\bar\omega^a=\frac{1}{1-\alpha}\log\!\left[(Y/K)\,\bar n^{-(1-\alpha)}\right]=-2.294907$, and shares $i^s=\bar x/(Y/K)=0.291952$, $c^s=0.708048$ — independent closed form. (His Table 3, Panel B mean $I_t/Y_t=25\%$ is a simulated stochastic mean at $\gamma=30$; the anchor checks the deterministic limit, not that moment.)

**Limitations.** (i) Frequency: the paper solves monthly and time-aggregates simulated data to quarterly statistics; we re-index the model at quarterly frequency with the conversions above. Mean growth, the short-run innovation variance, discounting, depreciation, and the skip-sampled law of $x_t$ convert exactly, but the quarterly model is not the exact time aggregation of the monthly one (decisions reset quarterly here, monthly there), so simulated moments can differ at the margin from his Table 3. (ii) We solve the planner allocation only; his financial-leverage block ($B/S=2/3$, levered return, p. 13) is a pricing add-on that does not affect the allocation and is not part of the solved system. (iii) Labor is fixed in the paper itself (fn. 14) — nothing is restricted there. No parameter is changed from Table 3, Panel A.

**Solve status: solves cold — automatic guess alone (about 7 s); the paper closed-form seed is optional (about 5 s). Cold-solve agreement with the anchors is an independent check.**'''))

cells.append(md(r'''### TALLARINI — Tallarini (*JME* 2000), risk-sensitive stochastic growth

**Source.** Tallarini, T.D., Jr., "Risk-sensitive real business cycles", *Journal of Monetary Economics* 45 (2000) 507–532. Version solved: his **production economy**, Eqs. (25)–(29) (p. 520), at $\chi = 100$ (his $\chi$ is the coefficient of relative risk aversion) — the largest value on his baseline calibration grid $\chi \in \{1, 10, 25, 100\}$ (p. 523), though not the paper's extreme: his Table 9 matching experiment (p. 529) reaches $\chi = 180.2$ with $\sigma_\varepsilon$ readjusted — with $\beta = 0.9926$ ($\beta$ = subjective discount factor). Native frequency: quarterly — no frequency conversion anywhere in this model.

**Economic meaning.** The paper separates risk aversion from intertemporal substitution (EIS fixed at one) in the one-sector stochastic growth model of Christiano–Eichenbaum (1992): raising risk aversion barely moves the business-cycle moments but pushes the risk-free rate and the market price of risk toward the data, and raises the welfare cost of fluctuations. It is in this lineup as the extreme-preference stress test: the simplest technology of the six (one shock, no adjustment costs, random-walk productivity) paired with the largest risk aversion, probing the expansion farthest from log utility.

**Variable dictionary** (timing: the engine dates the capital stock by its period of use, so engine $K_t$ = his $K_{t-1}$):

| paper symbol | engine symbol | meaning | type | units / frequency |
|---|---|---|---|---|
| $C_t$; $c^s_t \equiv C_t/Y_t$ | `cs_t` | consumption; its output share (Eq. (1) writes consumption $c_t$) | control | share, quarterly |
| $I_t$; $i^s_t \equiv I_t/Y_t$ | `is_t` | gross investment; its output share | control | share, quarterly |
| $Y_t$ | eliminated: $Y_t/K_t = e^{\alpha\omega_t}\bar N^{\alpha}$ | output, from his Eq. (25) | derived | level, quarterly |
| $K_{t-1}$ | folded into `wx_t` | capital used in production at $t$ | state | level |
| $X_t$ | folded into `wx_t` | random-walk productivity level | state | level |
| — (ours) | `wx_t` | $\omega_t \equiv \log(X_t/K_{t-1})$, the single engine state | state | log ratio, quarterly |
| $N_t$ | `nbar` | labor input — a **choice** in his model; frozen by us at $\bar N = 0.2305$ (see limitations) | parameter | fraction of time endowment |
| $L_t$ | — | leisure, $1-N_t$; the constant $1-\bar N$ once labor is fixed | derived constant | fraction |
| $\varepsilon_t$ | `sig` $\times$ `W1_t` | technology innovation; engine $W_{1,t}$ is standard normal | shock | $\mathrm N(0,\sigma_\varepsilon^2)$, quarterly |
| $\sigma_\varepsilon$ | `sig` | innovation standard deviation | parameter | per quarter |
| $\gamma$ | `g` | mean growth rate of productivity (his Table 4 caption) | parameter | log growth per quarter |
| $\alpha$ | `a` | **labor** income share — his exponent is on $N_tX_t$, so the capital share is $1-\alpha = 0.339$; do not read 0.661 as a capital share | parameter | unit-free |
| $\delta$ | `delta` | depreciation rate of capital | parameter | per quarter |
| $\beta$ | `beta` | subjective discount factor | parameter | per quarter |
| $\chi$ | `gamma` | CRRA for atemporal wealth gambles, Eq. (1) | parameter | unit-free |
| $\theta$ | — | leisure weight in his Eq. (22); with labor fixed it no longer affects the consumption choice, but it still rescales the risk sensitivity through $1+\theta$ (his Eq. (23); see limitations) | parameter | unit-free |
| $\sigma$ | — | risk-sensitivity parameter, his Eqs. (2) and (23) | derived constant | unit-free |

**Notation warning:** his $\gamma$ is the productivity drift (engine `g`); the engine parameter named `gamma` is his $\chi$. His $\sigma$ is a preference parameter, not a volatility.

**Equations (his notation).** Preferences, with $U_t$ the date-$t$ continuation utility and $\mathrm E_t$ the conditional expectation (his Eq. (1) and Eq. (2), both p. 510):

$$U_t = \log c_t + \beta\,\frac{1}{(1-\beta)(1-\chi)}\,\log\!\big(\mathrm E_t\big[\exp\{(1-\beta)(1-\chi)\,U_{t+1}\}\big]\big), \qquad \sigma \equiv 2(1-\beta)(1-\chi).$$

His production-economy baseline (Eq. (22), p. 519) adds $\theta\log L_t$ to the flow and rescales the risk sensitivity to $\sigma \equiv 2(1-\beta)(1-\chi)/(1+\theta)$ (his Eq. (23), p. 519). Technology and constraints (all p. 520):

$$Y_t = K_{t-1}^{1-\alpha}(N_tX_t)^{\alpha} \;\;\text{(25)}, \qquad \log X_t = \gamma + \log X_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d. } \mathrm N(0,\sigma_\varepsilon^2) \;\;\text{(26)},$$

$$K_t = (1-\delta)K_{t-1} + I_t \;\;\text{(27)}, \qquad L_t + N_t \le 1 \;\;\text{(28)}, \qquad C_t + I_t \le Y_t \;\;\text{(29)}.$$

**Mapping to the engine form.** He stationarizes by dividing by $X_t$ (his Eq. (30), p. 520); the engine divides by capital instead — an equivalent change of variables. With labor frozen at $\bar N$ and (28)–(29) at equality, the state is $\omega_t = \log(X_t/K_t)$ (engine timing), so $Y_t/K_t = e^{\alpha\omega_t}\bar N^{\alpha}$ and

$$\kappa_t = \log(C_t/K_t) = \log c^s_t + \alpha\,\omega_t + \alpha\log\bar N, \qquad \psi^g_t = \log(K_{t+1}/K_t) = \log\!\big(1-\delta+i^s_t\,e^{\alpha\omega_t}\bar N^{\alpha}\big),$$

$$\psi^x:\;\; \omega_{t+1} = \omega_t + \gamma + \sigma_\varepsilon W_{1,t+1} - \psi^g_t, \qquad \phi:\;\; 1 - c^s_t - i^s_t = 0.$$

With leisure fixed, his Eq. (22) collapses (up to an additive constant) to an Eq.-(1)-form recursion — but with risk-aversion coefficient $(\chi+\theta)/(1+\theta)$, not $\chi$, because his Eq. (23) keeps the $1+\theta$ rescaling. The engine instead runs his consumption-only preferences, Eq. (1) read through Eq. (2), setting $\gamma_{\text{eng}} = \chi = 100$ (see the risk-aversion limitation); his unit EIS is run as $\rho = 1.001$ (book convention).

**Parameters.** His Table 4 (p. 524) is quarterly and taken from Christiano–Eichenbaum (1992) except $\sigma_\varepsilon$, which he chose to match the variance of consumption growth under expected utility (p. 523); $\beta = 0.9926 = 1.03^{-0.25}$ is the Christiano–Eichenbaum value (text, pp. 523–524).

| parameter | paper value (native freq) | conversion rule | engine value (quarterly) | status |
|---|---|---|---|---|
| $\alpha$ (`a`) | 0.661 (Table 4, p. 524) | none — already quarterly | 0.661 | **calibrated** |
| $\delta$ (`delta`) | 0.021 (Table 4, p. 524) | none | 0.021 | **calibrated** |
| $\gamma$ (`g`) | 0.004 (Table 4, p. 524) | none | 0.004 | **calibrated** |
| $\sigma_\varepsilon$ (`sig`) | 0.0115 (Table 4, p. 524) | none | 0.0115 | **calibrated** |
| $\beta$ (`beta`) | 0.9926 (pp. 523–524; Table 5 heading, p. 525) | none | 0.9926 | **calibrated** |
| $\chi$ (`gamma`) | 100 (largest value of his baseline grid $\chi \in \{1,10,25,100\}$, p. 523; Tables 5, 7, 8) | $\gamma_{\text{eng}} = \chi$, his Eq. (2) | 100.0 | **calibrated** |
| $\bar N$ (`nbar`) | 0.2305 (Table 4, p. 524; fn. 9, p. 522) | labor frozen at this value | 0.2305 | **our restriction** |
| $\rho$ (`rho`) | EIS $= 1$ (Eq. (1), p. 510) | unit EIS run as $\rho = 1.001$ | 1.001 | **convention** |

**Anchors.** All three are closed forms on the deterministic balanced path of his Eqs. (25)–(29). The first two follow immediately from Eqs. (26)–(27) (he states the balanced growth rate on p. 520); the Euler line is derived by us from the deterministic planner steady state — it is not printed as an equation in the paper:

$$\text{growth} = \gamma, \qquad I/K = e^{\gamma}-1+\delta = 0.02501, \qquad (1-\alpha)\,\frac{Y}{K} = \frac{e^{\rho\gamma}}{\beta} - (1-\delta) \;\Rightarrow\; \frac{Y}{K} = 0.09585,$$

hence $i^{s*} = 0.2609$, $c^{s*} = 0.7391$, $\omega^* = \alpha^{-1}\log\big((Y/K)\,\bar N^{-\alpha}\big) = -2.0801$ (evaluated at $\rho = 1$; the $\rho = 1.001$ convention shifts these only in the fourth to fifth decimal). Consistency check against his own numbers: his Table 5 (p. 525; $\chi=1$, $\beta=0.9926$) prints steady-state means implying $\bar k/\bar y = 8.052/0.768 = 10.484$ and $\bar i/\bar y = 0.2617$, versus the anchor values $e^{\gamma}K/Y = 10.475$ and $i^{s*} = 0.2609$ — agreement to 0.1–0.3 percent, the slack being his log-LQ approximation and three-decimal table rounding. This is an approximate consistency check, not an exact independent target.

**Limitations / restrictions.**
- **Fixed labor is our restriction.** His baseline chooses $N_t$: leisure enters Eq. (22) with weight $\theta = 2.9869$ at $\beta = 0.9926$, set so that mean labor is 0.2305 in the $\chi=1$ case (fn. 9, p. 522; Table 5, p. 525). A labor choice lies outside the engine constraint class, so we freeze $N_t = \bar N = 0.2305$ at every $\chi$; his own $\chi=100$ economy has mean labor 0.2331 (Table 5).
- **Risk-aversion reading.** With labor fixed, his preferences are exactly an Eq.-(1) recursion and we set $\gamma_{\text{eng}} = \chi = 100$ via Eq. (2). In his leisure economy, Eq. (23) divides the risk sensitivity by $1+\theta$, so the effective CRRA for consumption gambles is $(\chi+\theta)/(1+\theta) = 25.8$ at $\chi=100$ (his pp. 520, 525). Our run therefore prices consumption risk as his consumption-only $\chi=100$ agent — more risk-averse over consumption than his production economy at the same $\chi$.
- $\rho = 1.001$ replaces exact unit EIS (book convention, as in the other unit-EIS models).
- **Solution concept.** He solves a risk-sensitive LEQG approximation centered at a risk-adjusted stochastic steady state (fixed point in mean capital, Section 6, pp. 522–523); his Table 5 means shift roughly 10 percent from $\chi=1$ to $\chi=100$ (p. 524). The engine expands around the deterministic balanced path, so the anchors above are deterministic-path objects.
- Constraints (28)–(29) are imposed at equality (standard under monotone preferences).

**Solve status: solves cold — automatic guess alone (about 2 s) at his largest baseline risk aversion $\chi = 100$; the closed-form seed is optional. Cold-solve agreement with the anchors is an independent check.**'''))

cells.append(md(r'''### Shock-correlation structures

For the battery, the shock vector $W$ is replaced by $L\,W$ with **unit row norms**: each equation keeps its published marginal volatility exactly; only the correlations between the innovations feeding different equations change. $L$ orthogonal is a pure rotation (the economy is unchanged — an exact invariance to test against). Families: `diagonal` (the paper's own layout), `dense`, `leverage` (capital vs. last row correlation $\approx -0.9$), `collinear` ($\approx 0.99$), `rankdef` (singular — for the two-shock models this is the polar case of perfectly correlated short- and long-run innovations). Single-shock models (KL, TALLARINI) enter at their published loading with the sign-flip twin, the only orthogonal transformation in one dimension.'''))
cells.append(code('''def directions(family, n, seed=23):
    rng = np.random.default_rng(seed)
    if family == "diagonal":
        D = np.eye(n)
    elif family == "dense":
        D = rng.standard_normal((n, n))
    elif family == "leverage":
        D = np.eye(n)
        D[0] = np.ones(n) / np.sqrt(n)
        D[-1] = -0.9 * D[0] + 0.45 * (np.arange(n) == n - 1)
        for i in range(1, n - 1):
            D[i] = rng.standard_normal(n)
    elif family == "collinear":
        D = rng.standard_normal((n, n))
        D[1] = D[0] + 0.14 * rng.standard_normal(n)
    elif family == "rankdef":
        D = rng.standard_normal((n, n))
        D[-1] = D[0] if n == 2 else 0.6 * D[0] + 0.4 * D[1]
    return D / np.linalg.norm(D, axis=1, keepdims=True)

FAMILIES = ["diagonal", "dense", "leverage", "collinear", "rankdef"]
ORDER = ["AK", "HABIT", "KL", "ACL", "CROCE", "TALLARINI"]'''))

# ---------------------------------------------------------------- SOLVE
cells.append(md(r'''## Solve

**First pass** — every model to its published preference target, in two separately reported modes:

- **cold**: the automatically constructed guess alone — no seeds. A cold solve whose steady state then matches the paper's closed forms is an **independent check** (the paper's numbers were never given to the solver).
- **seed-assisted**: the constructed guess with the paper's closed-form steady-state values passed as seeds (only the four literature models provide them). Agreement here is a **consistency check** — the paper's values enter the initialization, so they cannot also count as independent validation.

Each accepted solution passes two residual gates — the model's own deterministic equations and the engine's complete steady-state system, both to $10^{-6}$ — and is then compared against its paper anchors. The gates certify convergence, not specification; the anchors are the external check.

ACL is the slow one: its cold solve takes about six minutes, most of the first-pass runtime.

**Second pass** — the correlation battery: every multi-shock model under the five $\Sigma$ structures plus a rotated twin per cell; single-shock models at their published loading plus the sign-flip twin. The battery uses the seeds throughout for speed — what the construction does on its own is established by the first pass. For the first-pass solutions we also compute consumption-growth exposure and price elasticities (first shock, median state, 160 quarters — the construction of Figures 11.1–11.3).'''))

cells.append(code('''T_ELAS = 160

# log SDF increment (appendix construction): log(beta) - rho*(cons growth) + (rho-1)*(vmr1+vmr2/2) + log N-tilde
def log_sdf(sol, rho_v, beta_v):
    vmr = sol["vmr1_tp1"] + 0.5 * sol["vmr2_tp1"]
    return (np.log(beta_v) - rho_v * sol["gc_tp1"] + (rho_v - 1) * vmr
            + sol["log_N_tilde"])

# paper anchors: closed-form steady-state values from each PAPER (growth rate,
# ratio-state levels); every solved steady state is compared against these
ANCH = {
    "AK": {"D2_t": 0.019023},
    "HABIT": {},
    "KL": {"log_gk_t": 0.004, "w_t": MODELS["KL"]["seeds"]["w"]},
    "ACL": {"log_gk_t": MODELS["ACL"]["defaults"]["mu"],
            "wa_t": MODELS["ACL"]["seeds"]["wa"], "s_t": MODELS["ACL"]["seeds"]["s"]},
    "CROCE": {"log_gk_t": MODELS["CROCE"]["defaults"]["mu"],
              "wa_t": MODELS["CROCE"]["seeds"]["wa"]},
    "TALLARINI": {"log_gk_t": MODELS["TALLARINI"]["defaults"]["g"],
                  "wx_t": MODELS["TALLARINI"]["seeds"]["wx"]},
}

def _anchor_err(r, M, T, name):
    ss = np.asarray(r["ss"], float)
    names = [str(n) for n in _ss_names(M["build"](T), M["n_states"], M["n_shocks"])][1:]
    d = dict(zip(names, ss))
    return max((abs(d[k] - v) for k, v in ANCH[name].items()), default=float("nan"))

first = {}
for name in ORDER:
    M = MODELS[name]
    T = dict(M["defaults"]); T.update(M["target"])
    row = {"model": name}

    # --- cold: constructed guess alone, no seeds -------------------------
    t1 = time.time()
    r_cold, msg = autosolve(M["build"], M["defaults"], T,
                            M["n_states"], M["n_shocks"], timeout=600)
    row.update(cold=r_cold is not None, cold_secs=round(time.time() - t1))
    if r_cold is None:
        row["cold_note"] = msg
    else:
        row["cold_anchor_err"] = _anchor_err(r_cold, M, T, name)

    # --- seed-assisted: only where the paper provides closed forms -------
    r_seed = None
    if M["seeds"]:
        t1 = time.time()
        r_seed, msg = autosolve(M["build"], M["defaults"], T,
                                M["n_states"], M["n_shocks"],
                                timeout=600, state_seeds=M["seeds"])
        row.update(seeded=r_seed is not None, seed_secs=round(time.time() - t1))
        if r_seed is not None:
            row["seed_anchor_err"] = _anchor_err(r_seed, M, T, name)

    # elasticities from the accepted solution: cold where cold works
    r = r_cold if r_cold is not None else r_seed
    row["solved"] = r is not None
    if r is not None:
        row["basis"] = "cold" if r_cold is not None else "seed-assisted"
        row["expo"] = exposure_elasticity(r["gc_tp1"], r["X1_tp1"], r["X2_tp1"],
                                          T_ELAS, shock=0, percentile=0.5).flatten()
        row["price"] = price_elasticity(r["gc_tp1"], log_sdf(r, T["rho"], T["beta"]),
                                        r["X1_tp1"], r["X2_tp1"],
                                        T_ELAS, shock=0, percentile=0.5).flatten()
    first[name] = row
    ce = row.get("cold_anchor_err", float("nan"))
    se = row.get("seed_anchor_err", float("nan"))
    print("%-9s cold: %s %5ds  anchor %s   | seeded: %s  anchor %s" % (
        name, "OK  " if row.get("cold") else "FAIL", row.get("cold_secs", 0),
        ("%.1e" % ce) if ce == ce else "  --   ",
        ("OK %3ds" % row.get("seed_secs", 0)) if row.get("seeded") else "  --  ",
        ("%.1e" % se) if se == se else "  --   "), flush=True)'''))

cells.append(code('''grid, t0 = [], time.time()
for name in ORDER:
    M = MODELS[name]
    T = dict(M["defaults"]); T.update(M["target"])
    nw = M["n_shocks"]
    if nw == 1:
        fams = [("published", np.array([[1.0]]), np.array([[-1.0]]))]
    else:
        Q, _ = np.linalg.qr(np.random.default_rng(99).standard_normal((nw, nw)))
        fams = [(f, directions(f, nw), directions(f, nw) @ Q) for f in FAMILIES]
    for fam, L0, L1 in fams:
        cell = {"model": name, "family": fam}
        t1 = time.time()
        r, msg = autosolve(with_loadings(M["build"], L0), M["defaults"], T,
                           M["n_states"], nw, state_seeds=M["seeds"])
        cell["solved"] = r is not None
        if r is not None:
            cell["ss"] = np.asarray(r["ss"], float)
            u = r["util_sol"]["\\u03bc_0"]
            cell["mu0"] = float(np.linalg.norm(np.asarray(u, float).flatten()))
            r2, _ = autosolve(with_loadings(M["build"], L1), M["defaults"], T,
                              M["n_states"], nw, state_seeds=M["seeds"])
            cell["ss_rot"] = np.asarray(r2["ss"], float) if r2 is not None else None
        cell["secs"] = round(time.time() - t1)
        grid.append(cell)
        print("  %-9s x %-9s: %s %4ds" % (name, fam,
              "OK  " if cell["solved"] else "FAIL", cell["secs"]), flush=True)
print("battery finished in %.1f min" % ((time.time() - t0) / 60))'''))

# ---------------------------------------------------------------- PLOT
cells.append(md(r'''## Plot

Everything below only reads `first` and `grid`.'''))

cells.append(code('''import pandas as pd
tab = pd.DataFrame([{ "model": n,
    "cold solve": ("OK (%ds)" % first[n]["cold_secs"]) if first[n].get("cold")
                  else ("FAIL (%ds)" % first[n].get("cold_secs", 0)),
    "cold anchor err (independent)": first[n].get("cold_anchor_err", float("nan")),
    "seed-assisted": ("OK (%ds)" % first[n]["seed_secs"]) if first[n].get("seeded")
                     else ("-" if not MODELS[n]["seeds"] else "FAIL"),
    "seed anchor err (consistency)": first[n].get("seed_anchor_err", float("nan")),
    "reported solution": first[n].get("basis", "-")} for n in ORDER]).set_index("model")
display(tab)

df = pd.DataFrame(grid)
def spread(g):
    ref = g.iloc[0]
    return max((float(np.max(np.abs(s - ref))) for s in g.iloc[1:]), default=0.0)
summary = pd.DataFrame({
    "cells": [f"{df[df.model == n].solved.sum()} / {len(df[df.model == n])}" for n in ORDER],
    "order-0 spread across families": [spread(df[df.model == n].dropna(subset=["ss"]).ss) for n in ORDER],
    "max rotation diff": [max((float(np.max(np.abs(c.ss - c.ss_rot)))
                               for _, c in df[df.model == n].iterrows() if c.ss_rot is not None),
                              default=float("nan")) for n in ORDER],
}, index=ORDER)
display(summary)
piv = df[df.family != "published"].pivot(index="model", columns="family", values="mu0")
display(piv.reindex(index=[n for n in ORDER if n in piv.index], columns=FAMILIES)
          .style.format("{:.4f}").set_caption("|mu0| — first-order drift tilt, by correlation structure"))'''))

cells.append(code('''import matplotlib.pyplot as plt, seaborn as sns
sns.set_style("darkgrid")
yrs = np.arange(1, T_ELAS + 1) / 4
fig, axes = plt.subplots(2, len(ORDER), figsize=(17, 5.6), sharex=True)
for j, n in enumerate(ORDER):
    if first[n]["solved"]:
        axes[0, j].plot(yrs, first[n]["expo"], lw=1.5)
        axes[1, j].plot(yrs, first[n]["price"], lw=1.5, color="#c44e52")
    axes[0, j].set_title(n)
    axes[1, j].set_xlabel("years")
axes[0, 0].set_ylabel("exposure elasticity")
axes[1, 0].set_ylabel("price elasticity")
fig.suptitle("consumption-growth elasticities at each model's published target, first shock, median state", y=1.02)
fig.tight_layout(); plt.show()'''))

cells.append(md(r'''**Reading the results.**

- **All six economies solve cold** — the constructed guess alone, no seeds — at their published preference targets, risk aversion up to $\chi = 100$, and their cold steady states match the papers' own restrictions (independent checks: the papers' numbers were never given to the solver). The AK and TALLARINI anchor entries reflect the $\rho = 1.001$ convention against $\rho = 1$ closed forms; HABIT has no paper anchor and is validated as a book replication.
- **ACL was the exception until the initialization read the growth rate from the declared trends**: it now solves cold in about six minutes to the Borovička–Hansen steady state, an independent check; the paper's values remain an optional accelerator (about 70 s).
- The correlation battery: order 0 is identical, digit for digit, across all $\Sigma$ structures and under rotations (both columns exactly zero), while $|\mu^0|$ moves with the correlations — largest where bad capital shocks bundle with bad long-run news. Elasticity term structures are smooth and settle in every economy.
- Two structural facts about the engine's model class were established while building this notebook and are stated in the Model section: the resource constraint must be of the share form (hence the output-share declarations), and an endogenous labor choice lies outside the current class. Both are exactly the kind of thing an entry layer can check or restate automatically — which is the point of `v2`.'''))

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
