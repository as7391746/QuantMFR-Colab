# `v2/` — an automatic `initial_guess` for `uncertain_expansion`

The appendix writes the model as

$$ \begin{aligned} X_{t+1}(\mathsf q) &=\psi^x[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q],\\ \widehat G_{t+1}(\mathsf q)-\widehat G_t(\mathsf q) &=\psi^g[D_t(\mathsf q),X_t(\mathsf q),\mathsf qW_{t+1},\mathsf q],\\ \widehat C_t(\mathsf q)-\widehat G_t(\mathsf q) &=\kappa[D_t(\mathsf q),X_t(\mathsf q)],\\ 0&=\phi[D_t(\mathsf q),X_t(\mathsf q)]. \end{aligned} $$

`uncertain_expansion` requires starting values for the corresponding
steady-state objects

$$ \widehat V^0-\widehat G^0,\quad \widehat C^0-\widehat G^0,\quad D^0,\quad MS^0,\quad MX^0,\quad MG^0,\quad \widehat G_{t+1}^0-\widehat G_t^0,\quad X^0. $$

The only substantive change in v2 is that these values are constructed
from $\kappa,\psi^g,\psi^x,\phi$ when `initial_guess=None`; the expansion
and root solver are unchanged. Components of $X^0$ not pinned by

$$ X^0=\psi^x[D^0,X^0,0,0] $$

are handled by multi-start. Paper values, when available, remain optional
and are reported separately. Continuation is used only for difficult
parameter targets, and every reported solution must pass both residual
checks.

Six economies are included: the appendix's AK and habit models,
Kaltenbrunner–Lochstoer, Ai–Croce–Li, Croce, and Tallarini
($\chi=100$). Five solve without paper values. Ai–Croce–Li is the
exception and is therefore reported as a consistency check.

## Files

| file | contents |
|---|---|
| [`AUTO_GUESS.md`](AUTO_GUESS.md) | construction, residual surfaces, captions, and numerical record |
| [`PROVENANCE.md`](PROVENANCE.md) | differences from the upstream appendix engine and frozen hashes |
| `../v2_demo.ipynb` | model variables, paper equations, calibrations, solves, and paper checks |
| `support_material/` | six residual surfaces, overview figure, reproducibility script, numerical arrays, ablation record, and hashes |

Start with `support_material/landscapes_overview.png`. Each panel varies
one component of $X^0$ and one investment component of $D^0$ while
reconstructing the remaining appendix objects. The surface reports the
complete steady-state residual; the floor markers report which starting
points converge. It is numerical validation, not a convexity claim.

The separately tested extension to the first-order-condition assembly is
not part of v2.
