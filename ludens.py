# %% Imports
import jax.numpy as jnp  # <- the core of jax are jnp.ndarrays
from jax import jit, grad, random

# %% ludens
jnp.array([1, 2, 3])  # <- jnp.array


def f(x):
    return x ** 2


x = jnp.array(3.0)  # <- jnp.array
f(x), grad(f)(x)  # <- jnp.array


#########################################################################

rngs = random.split(random.PRNGKey(0), 10)
fs = [f(rng) for rng in rngs]  # <- list comprehension
