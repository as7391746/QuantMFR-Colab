# Automatic Initial Guess and Numerical Validation

`uncertain_expansion` requires an `initial_guess` for its deterministic
steady-state solve. The v2 entry layer constructs that vector from the model
declaration when `initial_guess=None`. The expansion and root-solver
mathematics are unchanged.

## 1. Construction in the engine's notation

The engine takes the model in the notation of the computation appendix:

$$ X_{t+1}(\mathsf q) = \psi^x[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q], $$
$$ \widehat G_{t+1}(\mathsf q)-\widehat G_t(\mathsf q) = \psi^g[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q], $$
$$\widehat C_t(\mathsf q)-\widehat G_t(\mathsf q) = \kappa[D_t(\mathsf q),X_t(\mathsf q)],$$
$$0 =\phi[D_t(\mathsf q),X_t(\mathsf q)]. $$

Set $\mathsf q=0$, $W_{t+1}=0$, and impose time invariance; write
$g^0=\widehat G_{t+1}^0-\widehat G_t^0$ for the trial growth rate
(default $0.005$). The construction determines the steady-state objects
one coordinate at a time — every solve below is a one-dimensional sign
scan followed by bisection, never a joint system.

**Pass 1 — investment components of $D^0$.** The components of $D^0$
that enter $\psi^g$ take a common value solving

$$ \psi^g[D^0,X^0,0,0] = g^0 . $$

**Pass 2 — consumption components of $D^0$.** Each static constraint
pins one still-free component of $D^0$:

$$ \phi[D^0,X^0]=0 . $$

**Pass 3 — self-pinning components of $X^0$.** Each state solves its own
deterministic fixed-point equation

$$ X_j^0=\psi_j^x[D^0,X^0,0,0] . $$

A component that cancels out of its own equation — typically a ratio
state, whose law of motion determines only its increment — cannot be
pinned here; it is reported and handled in Section 2.

**Pass 4 — leftover components of $D^0$.** A component that enters
neither $\psi^g$ nor $\phi$ (a second capital good's investment) is
solved through the state equation it enters, so that the corresponding
component of $X^0$ keeps its fixed-point value. (The solve driver
instead closes such components by the equal-investment rule listed under
Limits.)

The four passes repeat three times, Gauss–Seidel style, so that each
pass sees the others' updated values. If no positive-consumption
interior allocation exists at $g^0$ — or the value recursion below
diverges there — the growth ladder lowers
$g^0\in\{0.003,0.002,0.001,0.0005,0.0002\}$ and the passes rerun; the
guess must be a feasible interior point, nothing more.

**Utility entries (closed forms, no solves).** From the constructed
$(D^0,X^0)$:

$$ \widehat C^0-\widehat G^0=\kappa[D^0,X^0], $$

and, for $\rho\ne1$, the deterministic value recursion has the closed
form

$$ \widehat V^0-\widehat G^0 = \widehat C^0-\widehat G^0 + \frac{1}{1-\rho} \log \frac{1-\beta}{1-\beta\exp\left((1-\rho)g^0\right)} , $$

valid exactly when $\beta\exp\left((1-\rho)g^0\right)<1$ — the
transversality condition, which the growth ladder enforces. For
$\rho=1$,

$$ \widehat V^0-\widehat G^0 = \widehat C^0-\widehat G^0 + \frac{\beta}{1-\beta}\, g^0 . $$

**Multiplier and normalizations.** The static multiplier starts at the
envelope value

$$ MS^0 = (1-\beta)\,\frac{\partial\kappa}{\partial D_c}[D^0,X^0], $$

where $D_c$ is the component of $D^0$ that enters $\kappa$ (the
consumption margin). The normalizations are $MG^0=1$ and $MX^0=0$.
These objects are placed directly into the `initial_guess` ordering
already used by `uncertain_expansion`.

**Worked example (Kaltenbrunner–Lochstoer).** The declaration has
controls $(c^s,i^s)$, state $\omega=\log(Z/K)$. Pass 1 solves
$\log\left(1-\delta+\phi_J(i^s e^{(1-\alpha)\omega})\right)=g^0$ for
$i^s$; Pass 2 solves $1-c^s-i^s=0$ for $c^s$; Pass 3 finds that
$\omega$ cancels out of its own law of motion
($\omega_{t+1}-\omega_t=\mu-\psi^g$ fixes only the increment), so
$\omega$ is reported unpinned and goes to the Section 2 grid — only the
Euler equation, which lives inside the engine's compiled system, can
determine its level. The utility entries then follow from the closed
forms. Started this way, the model solves cold in about 27 s; the
paper's closed-form $\omega$ is an optional accelerator (3 s).

| entry | construction |
|---|---|
| $D^0$, self-pinning components of $X^0$, $\widehat G_{t+1}^0-\widehat G_t^0$, $\widehat C^0-\widehat G^0$, $\widehat V^0-\widehat G^0$, $MS^0$ | model equations |
| components of $X^0$ not pinned by their own law of motion | multi-start or optional paper value |
| $MX^0$, $MG^0$ | normalization |

## 2. Solve and acceptance

If a component of $X^0$ cancels from its own fixed-point equation, the
driver tries a fixed grid of values and reconstructs all other entries of
`initial_guess` at every grid point. A paper value, when available, is an
optional alternative starting value. It is always reported separately:
a cold solve followed by comparison with the paper is an independent
check; a solve started from the paper value is a consistency check.

The engine then runs its existing root solve. A result is reported only
when both the model equations and the complete compiled steady-state
system have residual below $10^{-6}$. Continuation from the default
parameters is used only when the target parameters do not solve directly.

## 3. Numerical geometry

Each surface varies one component of $X^0$ and one investment component
of $D^0$. At every grid point the remaining entries are reconstructed as
above; the multiplier block is completed by linear least squares. No joint
nonlinear solve is used to construct the surface. Height and color show
the base-10 logarithm of the infinity norm of the complete steady-state
residual.

On the floor, green circles reach the verified steady state, orange squares
reach another root, and red crosses do not converge within the probe
budget. The large markers are ● unseeded `initial_guess`, ◇ optional paper
value, ★ lowest sampled residual, and + verified reference steady state.

![AK three-dimensional loss landscape](support_material/landscape_ak.png)

*Figure 1a. AK: the residual is relatively flat in $Z^2$ and sharply
localized in the investment decision $D^2$; the unseeded
`initial_guess` solves without a paper value.*

![HABIT three-dimensional loss landscape](support_material/landscape_habit.png)

*Figure 1b. HABIT: the trough links the habit state $X^0$ to the capital
investment decision; the unseeded `initial_guess` reaches the appendix
solution.*

![KL three-dimensional loss landscape](support_material/landscape_kl.png)

*Figure 1c. KL: the cold starting point lies on the plateau, while the
optional paper value places $X^0$ close to the sharp low-residual valley.*

![ACL three-dimensional loss landscape](support_material/landscape_acl.png)

*Figure 1d. ACL: the admissible region and attraction basin are narrow.
The cold attempt fails within budget; the paper value is required and the
result is a consistency check.*

![CROCE three-dimensional loss landscape](support_material/landscape_croce.png)

*Figure 1e. Croce: every sampled admissible starting point reaches the same
steady state; the paper value only accelerates the solve.*

![Tallarini three-dimensional loss landscape](support_material/landscape_tallarini.png)

*Figure 1f. Tallarini: most sampled starting points reach the verified
steady state at $\chi=100$; the paper value is optional.*

The surfaces show residual geometry; the floor probes show attraction.
They do not assert convexity. The grids and probe outcomes are stored in
`support_material/landscape_*.npz` and reproduced by
`support_material/make_landscapes.py`.

## 4. Numerical record

| model | cold `initial_guess` | optional paper value | validation |
|---|---|---|---|
| AK | residual $4.0\times10^{-2}$; solves in 2 s | none | independent; anchor error $7.1\times10^{-6}$ |
| HABIT | residual $9.3\times10^{-2}$; solves in 12 s | none | appendix replication |
| KL | residual $8.8\times10^{-1}$; solves in 27 s | solves in 3 s | independent cold check; seeded consistency error $3.3\times10^{-11}$ |
| ACL | residual $1.3$; cold and grid attempts fail | solves in 70 s | consistency only; error $5.0\times10^{-10}$ |
| CROCE | residual $3.1\times10^{-1}$; solves in 7 s | solves in 5 s | independent cold check; seeded consistency error $1.6\times10^{-11}$ |
| TALLARINI | residual $3.3\times10^{-1}$; solves in 2 s | solves in 1 s | independent cold check; $\rho$-convention gap $1.9\times10^{-4}$ |

Five of the six economies solve from the constructed `initial_guess`
without paper values. ACL is the exception and is labeled accordingly.
The full ablation record is in `support_material/ablation.json`.

## 5. Limits

1. The growth ladder and the equal-investment closing rule for extra
   capital goods are heuristics.
2. A component of $X^0$ that cancels from its own fixed-point equation
   requires multi-start or a supplied value.
3. Residual checks certify convergence of the compiled system, not correct
   model specification; the model-class restriction and paper checks remain
   necessary.
