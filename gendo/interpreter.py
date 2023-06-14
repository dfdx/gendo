import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import random
# Importing Jax functions useful for tracing/interpreting.
import numpy as np
from functools import wraps

from jax import core
from jax import lax
from jax._src.util import safe_map


def examine_jaxpr(closed_jaxpr):
    jaxpr = closed_jaxpr.jaxpr
    print("invars:", jaxpr.invars)
    print("outvars:", jaxpr.outvars)
    print("constvars:", jaxpr.constvars)
    for eqn in jaxpr.eqns:
        print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
    print()
    print("jaxpr:", jaxpr)


def foo(x):
    return x + 1


def bar(w, b, x):
    return jnp.dot(w, x) + b + jnp.ones(5), x


def f(x):
    return jnp.exp(jnp.tanh(x))


def eval_jaxpr(jaxpr, consts, *args):
    # Mapping from variable -> value
    env = {}

    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # Bind args and consts to environment
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Loop through equations and evaluate primitives using `bind`
    for eqn in jaxpr.eqns:
        # Read inputs to equation from environment
        invals = safe_map(read, eqn.invars)
        # `bind` is how a primitive is called
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        # Write the results of the primitive into the environment
        safe_map(write, eqn.outvars, outvals)
    # Read the final result of the Jaxpr from the environment
    return safe_map(read, jaxpr.outvars)