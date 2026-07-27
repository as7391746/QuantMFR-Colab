# The automatic starting point — definition, algorithm, and limits

This document defines precisely what the automatic starting point ("auto
guess") is and is not. The one-sentence summary:

> **Auto guess constructs a feasible, model-consistent starting vector; it
> does not solve the complete equilibrium system.** Multi-start,
> continuation, and the nonlinear root solve are separate stages of the
> driver, documented below under "What auto guess is *not*."

## 1. Definition

Auto guess is a map

$$G:(\mathcal{M},\theta)\ \longmapsto\ (x_0,\ \mathcal{U},\ \text{feasibility flags})$$

where $\mathcal{M}$ is the user-declared model (the four pieces
$\kappa,\ \psi^g,\ \psi^x,\ \phi$), $\theta$ the parameter vector, $x_0$ the
constructed starting vector, and $\mathcal{U}$ the set of states the model's
own equations cannot pin (they are handed to multi-start).

Everything is evaluated under the deterministic operator

$$\mathcal{D}_0:\qquad \mathsf{q}=0,\qquad W_{t+1}=0,\qquad x_{t+1}=x_t .$$

## 2. Coordinate-wise construction

For a candidate growth rate $\bar g$ (default $0.005$), three groups of
one-dimensional equations are solved in Gauss–Seidel order (scan for a sign
change, then Brent's method — never a joint multi-dimensional solve):

1. **Investment controls** invert the growth equation:
   $\psi^g(u_0, s_0;\theta) = \bar g$.
2. **Consumption controls** absorb the static constraint:
   $\phi_j(u_0, s_0;\theta) = 0$.
3. **States** sit at their own deterministic fixed points:
   $\psi^x_j(u_0, s_0;\theta) - s_{0j} = 0$.

The three passes repeat (three sweeps). If no positive-consumption interior
point exists at $\bar g$, the **growth ladder** lowers
$\bar g \in \{0.005, 0.003, 0.002, 0.001, 0.0005, 0.0002\}$ and retries: the
guess must be a feasible interior point, nothing more.

## 3. Utility-block entries (closed forms, not solves)

With $\kappa_0 = \kappa(u_0, s_0;\theta)$, the deterministic value recursion
has a closed form:

$$v_0=\frac{1}{1-\rho}\log\!\left[\frac{(1-\beta)e^{(1-\rho)\kappa_0}}{1-\beta e^{(1-\rho)\bar g}}\right],\qquad \rho\neq 1,$$

valid only under the transversality condition
$\beta e^{(1-\rho)\bar g} < 1$; and for $\rho = 1$:

$$v_0=\kappa_0+\frac{\beta}{1-\beta}\,\bar g .$$

The static-constraint multiplier is initialized by the envelope condition

$$m_s^0=(1-\beta)\,\frac{\partial \kappa}{\partial c}(u_0, s_0).$$

## 4. The three classes of coordinates (read this table carefully)

| class | coordinates | origin |
|---|---|---|
| **derived from the model's equations** | controls, states with self-pinning laws of motion, growth $\bar g$, $\kappa_0$, $v_0$, $m_s^0$ | Sections 2–3 above |
| **unpinned** ($\mathcal{U}$) | ratio states whose own law of motion cancels (e.g. $\log(Z/K)$): only the Euler equation fixes their scale | handed to multi-start / optional paper seeds |
| **normalized initial values** | growth costate $m_g = 1$, other costates $= 0$ | conventions, *not* equation-derived |

## 5. What auto guess is *not* (the driver's other stages)

These live in `autosolve.py` and are separate from $G$:

- **Multi-start**: grid restarts over $\mathcal{U}$, re-deriving the whole
  guess at each pinned value (10-minute budget).
- **Optional paper seeds** (`state_seeds`): closed-form steady-state values
  a paper provides, used as overrides for $\mathcal{U}$. Results obtained
  with seeds are **consistency checks against the paper**, not independent
  discoveries.
- **Globalized root solve** (the engine's, unchanged): Newton (`hybr`),
  with an $\|F\|^2$ descent fallback and re-polish.
- **Coordinate-wise continuation**: one parameter axis at a time from the
  model's defaults, bisecting the step on failure, warm-starting each step.
- **Acceptance gates**: a returned point is accepted only if it satisfies
  both the model's deterministic equations and the engine's complete
  steady-state system to $10^{-6}$. The gates certify **convergence**, not
  **specification** — see `MODIFICATIONS.md`, Known limitations.

## 6. Known limitations

- $\bar g$ and its ladder are heuristics (any feasible interior growth
  target works; none is "the" right one).
- Unpinned states genuinely require multi-start or a seed: their scale is
  invisible to every one-dimensional pass (only the Euler equation carries
  it).
- Leftover investment controls (a second capital's) close by an
  equal-investment rule — a heuristic.
- On the reconstruction manifold near degenerate regions (e.g. the
  $\log(Z/K)=0$ plateau of KL), the transversality condition can fail and
  $v_0$ is undefined; the flags report this.
