# `v2/` — an automatic starting-point and verification layer for `uncertain_expansion`

This folder is the book's expansion engine with the solver mathematics **left
exactly as it is**, wrapped by a thin layer that lets a user state a new model
and solve it **without hand-tuning a starting point**. `PROVENANCE.md` lists
every byte-level difference from the upstream branch copy.

The design borrows standard tools from numerical optimization — a constructed
starting point, merit-function globalization, multi-start, and parameter
continuation. These are the techniques that convex-analysis-based solvers
developed to behave well on the systems that steady states actually are:
smooth but **nonconvex**. Nothing here assumes convexity; it borrows the
machinery convex theory built for the nonconvex case, and — importantly —
**checks the numerical properties of the point it returns** before accepting it.

---

## The problem, in optimization terms

The order-0 step of the expansion solves a nonlinear system for the
deterministic steady state `x`:

```
F(x) = 0            (equilibrium: recursive-utility, FOCs, costates, laws of motion)
```

equivalently, it minimizes the **merit function**

```
m(x) = ‖F(x)‖²   ≥ 0,   with m(x) = 0 at a steady state.
```

`m` is smooth but nonconvex: it can have several stationary points, and the
attractive one for a fast local method (Newton) depends on where you start.
The book shipped a single hand-tuned starting point calibrated to the
Section 11.7 AK model; move away from it and Newton stalls or converges to the
wrong basin. v2 replaces that one point with four standard responses:

1. **construct a feasible starting point from the model itself** (Algorithm 1);
2. **globalize the local solve** — descend on `m` when the Newton step fails (Algorithm 3);
3. **multi-start** over the coordinates a single guess cannot pin (Algorithm 2);
4. **continue** in the parameters when a target is far from any default (Algorithm 2).

---

## Algorithm 1 — `ConstructStartingPoint`  (`auto_guess.py`)

A good starting point is *feasible* (positive consumption, interior controls)
and *cheap*. We never solve the full system jointly here — a joint `fsolve`
from a flat guess is exactly what fails (a log-volatility state whose true value
is near −12 has ~zero gradient from 0). Instead we solve **one equation at a
time** — block-coordinate / Gauss–Seidel — and wrap it in a **feasibility
restoration** loop that lowers the target growth rate until consumption is
positive.

```
input : model (κ, ψ^g, ψ^x, φ), parameters θ, optional state overrides S̄
output: starting vector x₀, feasible flag, list of states the model could not pin

for g_target in [0.005, 0.003, 0.002, 0.001, 0.0005, 0.0002]:   # feasibility ladder
    controls ← 0.01 ;  states ← S̄ (else 0)
    repeat 3 times:                                             # Gauss–Seidel sweeps
        # (i) investment controls: invert the growth law to hit g_target
        d ← root over (0, 1] of  [ ψ^g(controls, states) − g_target ]     # 1-D scan + brentq
        set every investment control ← d
        # (ii) consumption controls: absorb the static constraint, one unknown each
        for each constraint φ:
            c ← root over (0, 1] of  φ(controls, states) = 0              # 1-D
        # (iii) states: each state at the fixed point of its OWN law of motion
        for each state s:
            x_s ← root over [−30, 30] of  ψ^x_s(states) − s = 0           # 1-D wide scan
            if no root found:  mark s as UNPINNED                          # only Euler pins it
        # (iv) close any leftover control by equal split of investment
    if feasible (all consumption > 0):  break

# utility block from closed forms, not from a solve:
v  ← deterministic value-function recursion (closed form in g_target, κ, β, ρ)
ms ← (1 − β) · ∂κ/∂c                                                       # envelope condition
assemble x₀ in the engine's internal ordering ; return x₀, feasible, UNPINNED
```

The `UNPINNED` list is the key output: a ratio state such as `log(Z/K)` cancels
out of its own law of motion, so **only the Euler equation fixes its scale** —
Algorithm 2 handles those by multi-start.

---

## Algorithm 2 — `Solve`  (`autosolve.py`)

The driver. Warm-starts, and escalates only as far as needed. `state_seeds` are
optional **model-provided closed-form** steady-state values (e.g. a paper's own
`ω*` from its Euler equation); they are hints, not required — the multi-start
path reaches the same root without them, only slower.

```
input : model builder, anchor θ_a, target θ_t, optional state_seeds
output: verified steady state, or a diagnostic

x₀, feasible, U ← ConstructStartingPoint(model(θ_t), θ_t, overrides = state_seeds)

# fast path: solve the target directly
sol ← GlobalizedRootFind(model(θ_t), x₀) ; if Verify(sol): return sol

# multi-start: restart over states the model could not pin (Alg. 1's U)
for s in U, for v in {−4,−2,−6,−1,−0.5, 0.5, 1, 2} (≤10-min budget):
    x ← ConstructStartingPoint(model(θ_t), θ_t, overrides = {s: v})
    sol ← GlobalizedRootFind(model(θ_t), x) ; if Verify(sol): return sol

# continuation: walk one parameter at a time from a solved anchor to the target
sol ← solve-and-verify at the anchor θ_a          # (returns a diagnostic if even this fails)
for each parameter axis k with θ_a[k] ≠ θ_t[k]:
    t ← 0
    while t < 1:
        step ← remaining ; warm ← sol
        for tries in 0 … max_bisect:
            θ ← θ interpolated to t+step on axis k
            sol' ← GlobalizedRootFind(model(θ), warm)
            if Verify(sol'):  sol ← sol' ; t ← t+step ; break
            step ← step / 2                          # bisect on failure
        else: return "continuation stalled at t (likely no steady state along this path)"
return sol
```

Multi-start is the textbook response to nonconvexity; coordinate-wise
continuation is a homotopy that keeps every intermediate solve warm-started from
a nearby solved point.

---

## Algorithm 3 — `GlobalizedRootFind`  (the engine's `ss_solver`, unchanged)

This is Faisal's Feb-2026 robust steady-state solve, kept intact. A fast local
root-finder, **globalized** by descending on the merit function `m = ‖F‖²`
when the local step does not converge — the standard damped/least-squares
safeguard for Newton on a nonconvex system.

```
input : system F, starting point x₀
1  sol ← Newton(F, x₀, method = hybr)                    # fast; needs a good x₀
2  if sol.success:  x* ← sol.x
3  else:                                                 # globalize
4      x_ls ← argmin ‖F(x)‖²  (L-BFGS-B from x₀)         # descent on the merit function
5      x*   ← Newton(F, x_ls, method = hybr).x            # polish from the descent point
6  return x*     # acceptance is decided by Verify, not here
```

---

## Algorithm 4 — `Verify`  (acceptance gates)

We accept `x*` only if it satisfies the model to tolerance — the numerical
analogue of checking that the merit function is actually at zero, not merely
that the solver *reported* success. Two residuals are checked, plus one
well-posedness property of the expansion itself.

```
input : returned point x*, model, parameters
r₁ ← ‖ model's own deterministic equations at x* ‖         # feasibility of the stated model
r₂ ← ‖ engine's complete steady-state system at x* ‖       # FOC/costate rows included
r  ← max(r₁, r₂)
if tol < r < 1e-4:                                         # gray zone: re-polish once
    x* ← Newton(F, warm = x*) ; recompute r
accept ⟺ r ≤ tol   (tol = 1e-6)
```

A separate numerical-property check lives *inside* the expansion and surfaces
automatically: the exponential-linear-quadratic expectation
`log E[exp(Y)]` is well defined only when the effective covariance

```
Σ = I − 2·sym(mat(ww)) ≻ 0
```

is positive definite. A mis-scaled or ill-posed model fails there rather than
returning a plausible-looking wrong number.

**What the gates do and do not certify.** They certify **convergence** — that
`x*` is a solution of the *compiled* system to tolerance. They do **not**
certify **specification** — that the compiled system is the intended model.
Both gates evaluate the same system the solver just solved, so a model whose
first-order conditions are assembled incorrectly can converge to a point that
passes both. What guards against that today is stating the model inside the
engine's supported class (one share-form resource constraint), not the check
itself. Making the compiled system correct by construction — deriving the
first-order conditions by symbolic differentiation of the Lagrangian, one
multiplier per constraint — is the next step, and is what would let this
verification stand on its own.

---

## What is unchanged

The solver mathematics, the second-order expansion, the change-of-measure
iteration, and the elasticity computations are the upstream engine, untouched.
The layer only decides **where to start**, **how far to escalate**, and
**whether to accept** — it never changes how a given system is solved once a
starting point is fixed. See `PROVENANCE.md` for the exact diff and
`models_sourced.py` for the six worked economies these algorithms are tested on.
