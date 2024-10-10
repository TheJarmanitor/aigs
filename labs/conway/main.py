# %% conway.py
#   conway game of life in jax
# by: Noah Syrkis

# %% Imports
from jax import random, lax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %% Functions
def init(rng, shape):
    return random.bernoulli(rng, 0.5, shape).astype(jnp.int32)


def conv(input, stride: int = 1):
    kernel = jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).reshape(1, 1, 3, 3)
    return lax.conv(
        input.reshape(1, 1, input.shape[0], input.shape[1]), kernel, (stride, stride), padding="SAME"
    ).reshape(input.shape[0], input.shape[1])


def step(board):
    neighs = conv(board)
    new_cells = neighs == 3
    survived = (neighs == 2) & board
    return new_cells | survived


# %% State
rng = random.PRNGKey(0)
shape = (50, 50)
board = init(rng, shape)


boards = [board]
for i in range(100):
    board = step(board)
    boards.append(board)


fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i, ax in enumerate(axes.flat):
    ax.imshow(boards[i])
    ax.axis("off")
