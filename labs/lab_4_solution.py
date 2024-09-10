# %% lab_4.py
#   evolve neural networks (no grad) with ES/GA
# by: Noah Syrkis

# %% Imports ############################################################
from jax import random, grad, jit, vmap
from jax._src.api import vmap
from jaxtyping import Array
import jax.numpy as jnp
import evosax  # <- use this (https://github.com/RobertTLange/evosax)
import evojax  # <- or this  (https://github.com/google/evojax)
import plotly.graph_objects as go
import os
import plotly.express as px
import plotly.offline as pyo
from typing import Callable

# %% Helper functions ###################################################
def plot_fn(fn: Callable, steps=100, radius = 4) -> None:   # plot a 3D function

    # create a grid of x and y values
    x = jnp.linspace(-radius, radius, steps)  # <- create a grid of x values
    y = jnp.linspace(-radius, radius, steps)  # <- create a grid of y values
    Z = fn(*jnp.meshgrid(x, y))  # <- evaluate the function on the grid

    # create a 3D surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])  # <- create a 3D surface plot
    pyo.plot(fig, filename=f"{fn.__name__}.html")   # <- save the plot to an html file


# %% Evolution as optimization ##########################################
# Implement a simple evolutionary strategy to find the minimum of a
# [function](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
# 1. Select a function and implement it in jax.
# 2. Implement a simple ES algorithm.
# 3. Find the minimum of the function.
rng = random.PRNGKey(1331)
rng, *keys = random.split(rng, 4)  # Key for each operation

# %% 1.
def ackley_fn(x: Array, y: Array) -> Array:
    return -20 * jnp.exp(-0.2 * jnp.sqrt(0.5 * (x**2 + y**2))) - jnp.exp(0.5 * (jnp.cos(2 * jnp.pi * x) + jnp.cos(2 * jnp.pi * y))) + jnp.e + 20

def mccormick_fn(x: Array, y:Array) -> Array:
    return jnp.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1

def three_hump_fn(x: Array, y: Array) -> Array:
    return 2 * x**2 - 1.05 * x**4 + ((x**6)/6) + x * y + y**2
# plot_fn(mccormick_fn)

# %% 2.
def init_population(rng, size=100, dim=2):  # for 2d optimization problems size=100, dim=2
    population = random.normal(rng, (size, dim))
    return population

test_pop = init_population(rng)

def mutate(rng, parents, std=0.1):
    return random.normal(rng, parents.shape) * std

def evaluate(fn, population):
    return fn(population[:,0], population[:,1])

def es_algorithm(rng, fn, init_pop=None, steps=10000) -> Array:
    if init_pop is None:
        init_pop = init_population(rng)
    population = init_pop
    for _ in range(steps):
        rng, key = random.split(rng)
        fitness = evaluate(fn, population)
        indices = fitness.argsort()
        population = population[indices][:10].repeat(10, axis=0)
        population += mutate(key, population)
    return population[evaluate(fn, population).argsort()]

print(es_algorithm(rng, mccormick_fn, test_pop, steps=1000))

# %% Basic Neuroevolution ###############################################
# Take the code from the previous weeks, and replace the gradient
# descent with your ES algorithm.


# %% (Bonus) Growing topologies #########################################
# Implement a simple genetic algorithm to evolve the topology of a
# neural network. Start with a simple network and evolve the number of
# layers, the number of neurons, and the activation functions.
