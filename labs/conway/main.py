# %% conway.py
#   conway game of life in jax
# by: Noah Syrkis

# %% Imports
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
from utils import animate


# %% Functions
def init(rng, shape):
    return random.bernoulli(rng, 0.5, shape)


def step(board):
    raise NotImplementedError("You need to implement the step function.")


# %% State
rng, key = random.split(random.PRNGKey(0))
shape = (50, 50)
board = init(rng, shape)
boards = [board, init(key, shape)]
animate(boards, "conway.svg")
