# %% one.py
#   play one (math refresher)
# by: Noah Syrkis

# Exercise 1: Math Refresher
# Below are a series of exercises to help you refresh your math skills.
# They touch on statistics, linear algebra, and calculus, while introducing you to JAX.
# It is important that all of you have your environments set up correctly.
# Ideally we should have worked through all your issues in this session.

# %% Imports ############################################################
from jax import random, grad, Array
import jax.numpy as jnp
from typing import Callable, Dict, List
import chex

# %% Stats exercise [markdown]  #########################################
# You're given the data set `x` below.

# %% Setup
rng = random.PRNGKey(seed := 0)  # Random number generator
rng, *keys = random.split(rng, 4)  # Key for each operation

sigma = jnp.abs(random.normal(keys[0], (1,)))  # Standard deviation
mu = random.normal(keys[1], (1,))  # Mean

x = random.normal(keys[2], (10,)) * sigma + mu
print(x)

print(type(x))

# %% Question One [markdown]
# Compute the mean, variance, median, and standard deviation of the data set,
# without using any the built-int functions for doing so (i.e. `jnp.mean`, `jnp.var`, etc.).
# Write your own functions to do this.

# %% Solution
def mean(x: Array) -> Array:
    return x.sum()/x.size

print(mean(x))

def median(x: Array) -> Array:
    return x.sort()[x.size//2]
print(median(x))

def variance(x: Array) -> Array:
    return jnp.sum(jnp.power(x - mean(x),2))/x.size
print(variance(x))
# %% Question two [markdown]
# We are randomly setting the mean and the standard deviation from which the dataset is drawn.
# How close are the values you computed to the true values of `mu` and `sigma`?
# If we were to increase the size of the dataset,
# would the values you computed get closer to the true values?

# %% Solution
sample_sizes = [10, 100, 1000, 10000, 100000]
print(f"original values: mean {mu} sd {sigma}")
for n in sample_sizes:
    x = random.normal(keys[2], (n,)) * sigma + mu

    print(f"sample size {n}: mean {mean(x)} sd {jnp.sqrt(variance(x))}")

# %% Vector exercise [markdown] #########################################
# You're given two vectors `a` and `b` below.

# %% Setup
rng, *keys = random.split(rng, 3)  # Key for each operation

a = random.normal(keys[0], (10,))
b = random.normal(keys[1], (10,))

# %% Question three [markdown]
# write a function that computes the dot product of two vectors without using the built-in function `jnp.dot`.

# %% Solution

def dot_product(a: Array, b: Array) -> Array:
    return


# %% Question four [markdown]
# write a function that computes the outer product of two vectors without using the built-in function `jnp.outer`.

# %% Solution


# %% Question five [markdown]
# write a function that computes the element-wise product of two vectors without using the built-in functions.

# %% Solution


# %% Question six [markdown]
# write a function that computes the element-wise sum of two vectors without using the built-in functions.

# %% Solution


# %% Calculus exercise [markdown] #######################################
# You're given a function `f` below.
# It is a simple function that takes a scalar input and returns a scalar output.
# $$ f(x) = x^2 + 2x + 1 $$
# You're also given a value `x` at which to evaluate the function.
# $$ x = 3 $$
# Write a function `f_prime` that takes in a function `fn`, a scalar `x`,
# and a step size `h` as arguments, and returns the derivative of the function at `x`.


# %% Setup
x = jnp.array(3.0)


def f(x: Array) -> Array:
    return x**2 + 2 * x + 1


def f_prime(fn: Callable, x: Array, h: float) -> Array:
    raise NotImplementedError


grad(f)(x)  # <-- Your solution here instead of this cheating line

# %% Question seven [markdown]
# `g` takes in two vectors parameters `a` and `b` and a vector `x`.
# It returns the linear combination of `a` and `b` with `x`.
# manually tweak the values of `a` and `b` so that the output of `g` is as close to `y` as possible.


# %% Setup
def g(a: Array, b: Array, x: Array) -> Array:
    return a * x + b


x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([2.0, 3.0, 4.0])

a = jnp.array(0.0)
b = jnp.array(0.0)

y_hat = g(a, b, x)
