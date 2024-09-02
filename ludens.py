# %% Imports
from jax import random, grad, tree
import jax.numpy as jnp


def identity(x):
    return x
