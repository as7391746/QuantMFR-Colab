"""A user-facing layer over the RiskUncertaintyValue small-noise expansion
engine (frozen at bookshelf/_engines/ruv-planners-ext-09ca5df).

The user declares a model in his or her own notation and supplies parameters
as a plain dictionary; the engine's calling conventions (canonical variable
names, ordered argument lists, steady-state initial guesses) are handled
here.

    m = Model(states=["Z1", "Z2"], controls=["D1", "D2"], shocks=3)
    ... write the chapter's equations with m["Z1"], m.p.alpha, m.W, m.q ...
    sol = m.solve(PARAMS)
    curve = sol.price_elasticity(shock=1, T=160, quantile=0.5)

Conventions (matching how the chapter prints its equations):
  - state equations give NEXT-period states as expressions in current
    states/controls and next-period shocks m.W[i];
  - shock loadings are written WITHOUT the noise scale; Ito-correction
    terms carry m.q**2 explicitly — exactly as displayed in the chapter;
  - the parameter dictionary must include 'beta', 'rho', 'gamma' (the
    engine reads the preference parameters by these names);
  - vector parameters (e.g. sigma_k) are passed as length-n_shocks arrays
    and used in equations through m.dot("sigma_k") / m.norm2("sigma_k").
"""

import os
import re
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (os.path.join(HERE, "engine"),                  # local copy
                   os.path.join(os.path.dirname(HERE), "src")):   # repo-root src/
    if os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
        break
else:                                     # inside the bookshelf: frozen snapshot
    sys.path.insert(0, os.path.dirname(HERE))
    from _tools.paths import engine as _engine_path
    _engine_path("ruv-planners-ext")

from elasticity import exposure_elasticity, price_elasticity  # noqa: E402
from lin_quad_util import next_period  # noqa: E402
from uncertain_expansion import (  # noqa: E402
    approximate_fun, compile_equations, get_parameter_value,
    uncertain_expansion)


class _ParamNamespace:
    """Attribute access to parameter symbols: p.alpha, p.zeta, ..."""

    def __getattr__(self, name):
        return sp.Symbol(name)


# Names the engine creates internally; a user variable with one of these
# names (or matching the costate/shock patterns) would silently collide.
_RESERVED = {"log_gk", "q", "log_cmk", "vmk", "rmv", "ms", "mg"}
_RESERVED_PATTERNS = re.compile(r"m\d+|W\d+")


def _check_variable_name(name):
    if "_t" in name:
        raise ValueError(
            f"variable name '{name}' contains '_t': the engine rewrites "
            f"every '_t' inside a name to '_tp1', which would silently "
            f"corrupt it. Pick a name without '_t' (e.g. 'Xtilde').")
    if name in _RESERVED or _RESERVED_PATTERNS.fullmatch(name):
        raise ValueError(
            f"variable name '{name}' collides with an engine-internal "
            f"symbol. Reserved: {sorted(_RESERVED)}, m<digit>, W<digit>.")


class Model:
    """Declare states, controls and equations in the user's own notation."""

    def __init__(self, states, controls, shocks=3):
        self.state_names = list(states)
        self.control_names = list(controls)
        for name in self.state_names + self.control_names:
            _check_variable_name(name)
        self.n_shocks = int(shocks)
        self.p = _ParamNamespace()
        self.q = sp.Symbol("q_t")
        self.W = [sp.Symbol(f"W{i + 1}_tp1") for i in range(self.n_shocks)]
        self.state = {}          # name -> expression for next-period state
        self.growth = None       # expression: log K_{t+1} - log K_t
        self.consumption = None  # expression: log(C_t / K_t)
        self.constraint = None   # expression(s) equal to zero
        self._sym = {n: sp.Symbol(f"{n}_t")
                     for n in self.state_names + self.control_names}

    def __getitem__(self, name):
        return self._sym[name]

    def dot(self, vec_name):
        """<vec_name> . W  as sigma1*W1 + sigma2*W2 + ..."""
        return sum(sp.Symbol(f"{vec_name}{i + 1}") * self.W[i]
                   for i in range(self.n_shocks))

    def norm2(self, vec_name):
        """|vec_name|^2 as a symbolic sum of squares."""
        return sum(sp.Symbol(f"{vec_name}{i + 1}") ** 2
                   for i in range(self.n_shocks))

    # ------------------------------------------------------------------
    def _engine_inputs(self, params):
        controls = [f"{n}_t" for n in self.control_names]
        states = [f"{n}_t" for n in self.state_names]
        shocks = [f"W{i + 1}_t" for i in range(self.n_shocks)]
        variables = controls + states + ["log_gk_t"] + ["q_t"] + shocks
        variables_tp1 = [v + "p1" for v in variables]

        flat = {}
        for name, value in params.items():
            if name.endswith(("_t", "_tp1")):
                raise ValueError(
                    f"parameter name '{name}' ends in '_t'/'_tp1' and would "
                    f"collide with the engine's variable symbols.")
            arr = np.asarray(value, dtype=float)
            keys = ([name] if arr.ndim == 0
                    else [f"{name}{i + 1}" for i in range(len(arr))])
            for key, v in zip(keys, np.atleast_1d(arr)):
                if key in flat:
                    raise ValueError(
                        f"parameter '{key}' appears twice (a vector "
                        f"parameter expands to name1, name2, ...).")
                flat[key] = float(v)
        parameter_names = {name: sp.Symbol(name) for name in flat}
        args = list(flat.values())

        constraints = self.constraint
        if not isinstance(constraints, (list, tuple)):
            constraints = [constraints]

        return {
            "control_variables": controls,
            "state_variables": states,
            "shock_variables": shocks,
            "variables": variables,
            "variables_tp1": variables_tp1,
            "kappa": self.consumption,
            "growth": self.growth,
            "state_equations": [self.state[n] for n in self.state_names],
            "static_constraints": list(constraints),
            "parameter_names": parameter_names,
            "args": args,
        }

    def _default_guess(self, spec, start):
        """Build the engine's steady-state starting vector by introspecting
        the compiled variable list; `start` maps user variable names to
        rough starting values (states default to 0, controls to 0.02)."""
        n_J = len(self.control_names) + (len(self.state_names) + 1) + 2
        var_shape = [n_J, len(self.state_names) + 1, self.n_shocks]
        _, ss_variables, _, _, _ = compile_equations(
            parameter_names=spec["parameter_names"],
            variables=spec["variables"],
            variables_tp1=spec["variables_tp1"],
            control_variables=spec["control_variables"],
            state_variables=list(spec["state_variables"]),
            output_constraint=spec["kappa"],
            capital_growth=spec["growth"],
            state_equations=spec["state_equations"],
            static_constraints=spec["static_constraints"],
            var_shape=var_shape)
        names = [str(v) for v in ss_variables]
        n_q = names.index("q_t")
        start = dict(start or {})
        guess = []
        for name in names[1:n_q]:        # the vector fsolve actually solves
            base = name[:-2] if name.endswith("_t") else name
            if base in start:
                guess.append(float(start[base]))
            elif base in self.control_names:
                guess.append(0.02)
            elif base in self.state_names:
                guess.append(0.0)
            elif name == "log_gk_t":
                guess.append(float(start.get("growth", 0.005)))
            elif name == "vmk_t":
                guess.append(-2.0)
            elif name == "log_cmk_t":
                guess.append(-4.0)
            elif name == "mg_t":
                guess.append(1.0)
            elif re.fullmatch(r"m\d+_t|ms_t", name):
                guess.append(0.0)
            else:
                guess.append(0.1)
        return np.array(guess)

    # ------------------------------------------------------------------
    def solve(self, params, start=None, guess=None, tol=1e-10, max_iter=500):
        """Solve the model by second-order small-noise expansion.

        params : dict of parameter values (scalars or length-n_shocks arrays);
                 must include 'beta', 'rho', 'gamma'.
        start  : optional dict of rough steady-state starting values, keyed by
                 the user's own variable names (plus 'growth').
        guess  : optional full engine-format starting vector (overrides start).
        """
        for required in ("beta", "rho", "gamma"):
            if required not in params:
                raise ValueError(f"params must include '{required}'")
        spec = self._engine_inputs(params)
        if guess is None:
            guess = self._default_guess(spec, start)
        raw = uncertain_expansion(
            spec["control_variables"], spec["state_variables"],
            spec["shock_variables"], spec["variables"],
            spec["variables_tp1"], spec["kappa"], spec["growth"],
            spec["state_equations"], spec["static_constraints"],
            np.asarray(guess, dtype=float), spec["parameter_names"],
            spec["args"], approach="1", iter_tol=tol, max_iter=max_iter)
        return Solution(self, raw)


class Solution:
    """Wraps the engine output; every request uses the user's notation."""

    def __init__(self, model, raw):
        self.model = model
        self.raw = raw

    def parameter(self, name):
        return get_parameter_value(name, self.raw["parameter_names"],
                                   self.raw["args"])

    @property
    def consumption_growth(self):
        """LinQuad increment of log C (per period)."""
        return self.raw["gc_tp1"]

    def increment(self, expr):
        """LinQuad increment of ANY expression in the model's variables,
        e.g. sol.increment(sp.log(m['D2'])) for log(I/K) growth."""
        level, _, _, _ = approximate_fun(
            fun=expr, ss=self.raw["ss"],
            ss_variables=self.raw["ss_variables"],
            ss_variables_tp1=self.raw["ss_variables_tp1"],
            parameter_names=self.raw["parameter_names"],
            args=self.raw["args"], var_shape=self.raw["var_shape"],
            JX1_t=self.raw["JX1_t"], JX2_t=self.raw["JX2_t"],
            X1_tp1=self.raw["X1_tp1"], X2_tp1=self.raw["X2_tp1"],
            recursive_ss=self.raw["recursive_ss"], second_order=True)
        return next_period(level, self.raw["X1_tp1"],
                           self.raw["X2_tp1"]) - level

    @property
    def log_sdf(self):
        """log S_{t+1} - log S_t (recursive utility, as in the chapter)."""
        beta = self.parameter("beta")
        rho = self.parameter("rho")
        vmr = self.raw["vmr1_tp1"] + 0.5 * self.raw["vmr2_tp1"]
        return (np.log(beta) - rho * self.raw["gc_tp1"]
                + (rho - 1) * vmr + self.raw["log_N_tilde"])

    def exposure_elasticity(self, shock, T, quantile=0.5, process=None):
        inc = self.consumption_growth if process is None else process
        return exposure_elasticity(
            inc, self.raw["X1_tp1"], self.raw["X2_tp1"], T,
            shock=shock, percentile=quantile).flatten()

    def price_elasticity(self, shock, T, quantile=0.5, process=None):
        inc = self.consumption_growth if process is None else process
        return price_elasticity(
            inc, self.log_sdf, self.raw["X1_tp1"], self.raw["X2_tp1"], T,
            shock=shock, percentile=quantile).flatten()
