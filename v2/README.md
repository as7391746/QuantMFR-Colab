# `v2/` — an automatic `initial_guess` for `uncertain_expansion`

The appendix writes the model as

$$ \begin{aligned} X_{t+1}(\mathsf q) &=\psi^x[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q],\\ \widehat G_{t+1}(\mathsf q)-\widehat G_t(\mathsf q) &=\psi^g[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q],\\ \widehat C_t(\mathsf q)-\widehat G_t(\mathsf q) &=\kappa[D_t(\mathsf q),X_t(\mathsf q)],\\ 0&=\phi[D_t(\mathsf q),X_t(\mathsf q)]. \end{aligned} $$

`uncertain_expansion` requires starting values for the corresponding
steady-state objects

$$ \widehat V^0-\widehat G^0,\quad \widehat C^0-\widehat G^0,\quad D^0,\quad MS^0,\quad MX^0,\quad MG^0,\quad \widehat G_{t+1}^0-\widehat G_t^0,\quad X^0. $$

The only substantive change in v2: when `initial_guess=None`, we
construct these values from $\kappa,\psi^g,\psi^x,\phi$. The expansion
and the root solver are unchanged.

If components of $X^0$ are not determined by their own equations

$$ X^0=\psi^x[D^0,X^0,0,0], $$

we supply a grid of trial values to the solver; the user can also supply
a starting value for such a component directly. If the target parameter
values do not solve directly, we move one parameter at a time from the
defaults, restarting from the previous solution. We report a solution
only when the model equations and the complete steady-state system hold
to within $10^{-6}$.

In the validation, the supplied values are the papers' own steady-state
values, and we report the two starting points separately: a solve
without them, checked against the paper, is an independent check; a
solve started from them is a consistency check.

We include six economies: the appendix's AK and habit models,
Kaltenbrunner–Lochstoer, Ai–Croce–Li, Croce, and Tallarini
($\chi=100$). All six solve without paper values — Ai–Croce–Li in
about six minutes, once the initialization reads the balanced growth
rate from the declared trends; the papers' values remain optional
accelerators, and runs with them are reported as consistency checks.

## Files

| file | contents |
|---|---|
| [`AUTO_GUESS.md`](AUTO_GUESS.md) | construction, steady-state error surfaces, and numerical record |
| [`PROVENANCE.md`](PROVENANCE.md) | differences from the upstream appendix code and frozen hashes |
| `../v2_demo.ipynb` | model variables, paper equations, calibrations, solves, and paper checks |
| `support_material/` | six error surfaces, overview figure, the script and numerical arrays that reproduce them, the solve record, and hashes |

Start with `support_material/landscapes_overview.png`. Each panel varies
one component of $X^0$ and one investment component of $D^0$ and
constructs the remaining appendix objects from the model equations. The
surface reports the error in the complete steady-state system; the floor
markers report which starting points converge. It is numerical
validation, not a convexity claim.

The separately tested extension to the first-order-condition assembly is
not part of v2.
