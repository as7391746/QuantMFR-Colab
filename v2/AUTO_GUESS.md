# Automatic Initial Guess and Numerical Validation

`uncertain_expansion` requires an `initial_guess` for its deterministic
steady-state solve. The v2 entry layer constructs that vector from the model
declaration when `initial_guess=None`. The expansion and root-solver
mathematics are unchanged.

The construction uses the natural matching the model's structure
provides. Whatever a single equation claims is taken from that
equation: each state from its own evolution equation, one decision from
the resource constraint, the value entries in closed form, the
multipliers by a linear solve. The remaining choice margins — whose
information lives in the first-order conditions — are solved against
their own first-order conditions, as one small joint block. On the six
test economies the constructed point satisfies the complete
steady-state system to machine precision.

The whole algorithm at a glance:

<img src="support_material/flowchart.png" alt="flow chart of the automatic initial guess" width="540">

## 1. Construction in the appendix's notation

We take the model in the notation of the computation appendix:

$$ X_{t+1}(\mathsf q) = \psi^x[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q], $$
$$ \widehat G_{t+1}(\mathsf q)-\widehat G_t(\mathsf q) = \psi^g[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q], $$
$$\widehat C_t(\mathsf q)-\widehat G_t(\mathsf q) = \kappa[D_t(\mathsf q),X_t(\mathsf q)],$$
$$0 =\phi[D_t(\mathsf q),X_t(\mathsf q)]. $$

Set $\mathsf q=0$, set the shock vector to zero, and treat the variables
as time invariant. Given $D^0$ and $X^0$, the second block defines the
growth rate $g^0=\psi^g[D^0,X^0,0,0]$ and the third defines
$\widehat C^0-\widehat G^0=\kappa[D^0,X^0]$. The unknowns are $D^0$
and $X^0$, determined by the first and fourth blocks together with the
first-order conditions $Q^0H^0+P^0L^0-M^0=0$.

**What a single equation claims.**

- Each component of $X^0$ with its own evolution equation is solved from
  it: $X_j^0=\psi_j^x[D^0,X^0,0,0]$, a one-variable bracketed solve. A
  first-block equation that depends on its own state only through the
  growth equation claims no state: it degenerates into the restriction
  $\psi^g[D^0,X^0]=$ drift, and we keep it as that restriction.
- One component of $D^0$ is solved from the resource constraint
  $\phi[D^0,X^0]=0$; within the accepted class the constraint is linear
  in the decisions, so which component it claims cannot change the
  allocation.
- The growth rate and $\widehat C^0-\widehat G^0$ follow by definition
  (second and third blocks). With
  $Q^0 = \beta\exp\left[(1-\rho)\,g^0\right]$, the recursive
  utility updating gives, for $\rho\ne1$,

$$ \widehat V^0-\widehat G^0 = \widehat C^0-\widehat G^0 + \frac{1}{1-\rho} \log \frac{1-\beta}{1-Q^0} \qquad (Q^0<1), $$

  and for $\rho=1$,
  $\widehat V^0-\widehat G^0=\widehat C^0-\widehat G^0+\frac{\beta}{1-\beta}g^0$.
- The multipliers enter the first-order conditions linearly, so given
  $(D^0,X^0)$ they are recovered exactly by a linear least-squares solve:
  $MS^0$, $MX^0$, $MG^0$ all come from the system, not from
  normalizations.

**The remaining margins.** The components of $D^0$ the constraint does
not claim, and the states whose first-block equations degenerate, carry
the information of the first-order conditions. We solve them against
their own first-order conditions as one small joint block (dimension
one to four in the test economies): at each candidate for these
margins, the claims above rebuild every other entry, and the block is
solved by least squares on the complete steady-state system from a few
starting values.

**Initial values.** Every component starts from a neutral default
($0.01$ for decisions, $0$ for states). Wherever an equation determines
a coordinate, the bracketed solve erases the default; the joint block's
coordinates are searched from several starting values rather than a
single default.

**Result.** The constructed point is not an approximation:

| model | construction error | cold solve, total |
|---|---|---|
| AK | $1.5\times10^{-15}$ | 8 s |
| HABIT | $1.5\times10^{-15}$ | 17 s |
| KL | $2.2\times10^{-16}$ | 9 s |
| CROCE | $3.5\times10^{-15}$ | 15 s |
| TALLARINI | $1.1\times10^{-16}$ | 5 s |
| ACL | $9.0\times10^{-15}$ | 91 s |

The root solve of `uncertain_expansion` — unchanged — confirms each
point and the expansion proceeds; final residuals improve to
$10^{-15}$–$10^{-17}$. No trial growth rate and no supplied values are
involved anywhere in this table.

## 2. Solve and acceptance

If the joint block does not land — no candidate satisfies the
steady-state system — we fall back to the previous construction: a grid
of trial values for the unclaimed states, constructing all other
entries of `initial_guess` afresh at each value. The user can also
supply a starting value for such a component directly.

We then run the existing root solve of `uncertain_expansion` and report
a solution only when the model equations and the complete compiled
steady-state system hold to within $10^{-6}$. If the target parameter
values do not solve directly, we move one parameter at a time from the
default values, restarting each solve from the previous solution.

In the validation below, the supplied values are the papers' own
steady-state values, and we report the two starting points separately:
a solve without them, checked against the paper, is an independent
check; a solve started from them is a consistency check.

## 3. The steady-state equations over a grid

Each figure varies one component of $X^0$ and one investment component
of $D^0$ over a grid. At every grid point we construct the remaining
entries as in Section 1 and complete the multiplier and co-state entries
by linear least squares — no joint solve is used to construct the
surface. Height and color show the base-10 logarithm of the largest
error in the complete steady-state system at that point.

The figures below show the geometry and the previous construction's
starting point (●); the natural-matching construction lands on the
verified steady state (+) by construction.

On the floor, green circles mark grid points from which
`uncertain_expansion` reaches the verified steady state, orange squares
a different solution, and red crosses no convergence within the time
budget. The large markers are ● the constructed `initial_guess` without
supplied values, ◇ the constructed `initial_guess` with a supplied
starting value (here, the paper's), ★ the lowest sampled error, and +
the verified steady state.

![AK steady-state error surface](support_material/landscape_ak.png)

*Figure 1a. AK: the error is nearly flat in $Z_2$ and sharply localized
in the investment choice $D_2$; the constructed `initial_guess` solves
without a paper value.*

![HABIT steady-state error surface](support_material/landscape_habit.png)

*Figure 1b. HABIT: the trough links the habit state $X_1$ to the
capital-investment choice; the constructed `initial_guess` reaches the
appendix solution.*

![KL steady-state error surface](support_material/landscape_kl.png)

*Figure 1c. KL: the constructed `initial_guess` sits on the flat region;
the optional paper value places the state next to the narrow trough.*

![ACL steady-state error surface](support_material/landscape_acl.png)

*Figure 1d. ACL: the region where the construction is defined, and the
set of starting points from which the solver reaches the verified steady
state, are both narrow. With the growth rate read from the declared
trends, the constructed `initial_guess` solves without paper values in
about six minutes; the paper's values cut this to about 74 s.*

![CROCE steady-state error surface](support_material/landscape_croce.png)

*Figure 1e. Croce: every sampled starting point reaches the same steady
state; the paper's value only speeds up the solve.*

![Tallarini steady-state error surface](support_material/landscape_tallarini.png)

*Figure 1f. Tallarini: most sampled starting points reach the verified
steady state at $\chi=100$; the paper's value is optional.*

The surfaces show the size of the steady-state errors; the floor markers
show where the solver converges. They do not assert convexity. The
grids and outcomes are stored in `support_material/landscape_*.npz` and
reproduced by `support_material/make_landscapes.py`.

## 4. Numerical record

| model | constructed `initial_guess` (natural matching) | optional supplied value | validation |
|---|---|---|---|
| AK | steady state to $1.5\times10^{-15}$; cold in 8 s | none | independent; anchor error $7.1\times10^{-6}$ ($\rho$ convention) |
| HABIT | steady state to $1.5\times10^{-15}$; cold in 17 s | none | appendix replication |
| KL | steady state to $2.2\times10^{-16}$; cold in 9 s | optional | independent check without supplied values |
| ACL | steady state to $9.0\times10^{-15}$; cold in 91 s | optional (88 s) | independent check without supplied values |
| CROCE | steady state to $3.5\times10^{-15}$; cold in 15 s | optional | independent check without supplied values |
| TALLARINI | steady state to $1.1\times10^{-16}$; cold in 5 s | optional | independent check without supplied values |

All six economies solve cold, and the constructed point is the
steady state itself: the previous knife-edge behavior of the
two-capital model is gone because the solve now starts at the root.
Earlier stage-by-stage records are in
`support_material/ablation.json`.
