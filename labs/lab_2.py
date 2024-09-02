# %% lab_2.py
#     play two (autograd) and neural networks
# by: Noah Syrkis

#########################################################################
# %% Imports (add as needed) ############################################
from jax import grad, random, nn, tree
from jax import grad, random, nn, tree
import jax.numpy as jnp
from jaxtyping import Array
import chex
import equinox as eqx
import optax

from typing import Callable, Dict, List
from tqdm import tqdm
import os
from tqdm import tqdm

import tensorflow_datasets as tfds  # <- for data

import seaborn as sns  # <- for plotting
import matplotlib.pyplot as plt  # <- for plotting


#########################################################################
# %% Neural Networks ####################################################
# Implement a simple neural network using jnp, grad and random.
# Use the MNIST dataset from tensorflow_datasets.

# %% Data
mnist = tfds.load("mnist", split="train")
x_data = jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).reshape(-1,28*28)
y_data = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])

# %% Test
print(x_data.shape)
# %% random keys

rng = random.PRNGKey(seed := 1331)  # Random number generator


# %% params
@chex.dataclass
class Params:
    w1 = random.normal(rng, (x_data.shape[1], 64))
    b1 = random.normal(rng, 64)
    w2 = random.normal(rng, (64, 10))
    b2 = random.normal(rng, 10)
params = Params()


# %% Model (define model as a function that takes params and x_data)
def model(params: Params, x_data: Array):
    z = x_data @ params.w1 + params.b1
    z = nn.relu(z)
    z = z @ params.w2 + params.b2
    z = nn.softmax(z)

    return jnp.argmax(z, axis=1)


# %% Loss (define loss as a function that takes params, x_data, y_data)
def loss(params: chex.dataclass, x_data: Array, y_data):
    y_hat = model(params, x_data)
    return jnp.mean((y_hat - y_data)**2)

# %% Train loop

def train(loss_fn: Callable, params: Params, steps: int, learning_rate: float, x_data: Array, y_data: Array):
    for i in tqdm(range(steps)):
        grads = grad(loss_fn)(params, x_data, y_data)
        params = tree.map(lambda p, g: p - learning_rate * g, params, grads)

train(loss, params, 10000, 0.001, x_data, y_data)
# %% test

y_hat = model(params, x_data)

(y_hat == y_data).mean()

    # params -= gradient_of_loss(params)
# %%
y_hat = model(params, x_data)
(y_hat.argmax(axis=1) == y_data).astype(jnp.int32).mean()
# %% play with JIT and vmap to speed up training and simplify code


#########################################################################
# %% Equinox (optional) #################################################
# Do the same in Equinox (high-level JAX library)


#########################################################################
# %% Autograd (optional) ################################################
# Implement a simple autograd system using chex and jax (no grad)
# Structure could be something like this:


@chex.dataclass
class Value:
    value: Array
    parents: List["Value"] | None = None
    gradient_fn: Callable | None = None


def add(x: Value, y: Value) -> Value:
    """Add two values."""
    return Value(x.value + y.value, parents=[x, y], gradient_fn=lambda: 1)


def mul(x: Value, y: Value) -> Value:
    """Multiply two values."""
    raise NotImplementedError


def backward(x: Value, gradient: Array) -> Dict[Value, Array]:
    """Backward pass."""
    raise NotImplementedError


def update(x: Value, gradient: Array) -> Value:
    """Apply the gradient to the value."""
    raise NotImplementedError


# %%
jnp.array(1)
