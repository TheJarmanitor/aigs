#Let's make a full on neural network now

# %% import libraries
from jax import grad, random, nn, tree
import jax.numpy as jnp
from jaxtyping import Array
import chex
from typing import Callable, Dict, List
import equinox as eqx
import os
from tqdm import tqdm

import tensorflow_datasets as tfds  # <- for data

import seaborn as sns  # <- for plotting
import matplotlib.pyplot as plt  # <- for plotting


# %%
