# `v2/` — an automatic `initial_guess` for `uncertain_expansion`

The appendix writes the model as

$$ \begin{aligned} X_{t+1}(\mathsf q) &=\psi^x[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q],\\ \widehat G_{t+1}(\mathsf q)-\widehat G_t(\mathsf q) &=\psi^g[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q],\\ \widehat C_t(\mathsf q)-\widehat G_t(\mathsf q) &=\kappa[D_t(\mathsf q),X_t(\mathsf q)],\\ 0&=\phi[D_t(\mathsf q),X_t(\mathsf q)]. \end{aligned} $$

`uncertain_expansion` requires starting values for the corresponding
steady-state objects

$$ \widehat V^0-\widehat G^0,\quad \widehat C^0-\widehat G^0,\quad D^0,\quad MS^0,\quad MX^0,\quad MG^0,\quad \widehat G_{t+1}^0-\widehat G_t^0,\quad X^0. $$

The only substantive change in v2 is that these values are constructed
from $\kappa,\psi^g,\psi^x,\phi$ when `initial_guess=None`; the expansion
and root solver are unchanged. Components of $X^0$ not determined by

$$ X^0=\psi^x[D^0,X^0,0,0] $$

are handled by a grid of trial values. Paper values, when available,
remain optional and are reported separately. When the target parameter
values do not solve directly, the code moves one parameter at a time
from the defaults toward the target, restarting from the previous
solution; every reported solution must satisfy the model equations and
the complete steady-state system to within $10^{-6}$.

Six economies are included: the appendix's AK and habit models,
Kaltenbrunner–Lochstoer, Ai–Croce–Li, Croce, and Tallarini
($\chi=100$). Five solve without paper values. Ai–Croce–Li is the
exception and is therefore reported as a consistency check.

## Files

| file | contents |
|---|---|
| [`AUTO_GUESS.md`](AUTO_GUESS.md) | construction, steady-state error surfaces, and numerical record |
| [`PROVENANCE.md`](PROVENANCE.md) | differences from the upstream appendix code and frozen hashes |
| `../v2_demo.ipynb` | model variables, paper equations, calibrations, solves, and paper checks |
| `support_material/` | six error surfaces, overview figure, the script and numerical arrays that reproduce them, the solve record, and hashes |

Start with `support_material/landscapes_overview.png`. Each panel varies
one component of $X^0$ and one investment component of $D^0$ while
constructing the remaining appendix objects from the model equations.
The surface reports the error in the complete steady-state system; the
floor markers report which starting points converge. It is numerical
validation, not a convexity claim.

The separately tested extension to the first-order-condition assembly is
not part of v2.
