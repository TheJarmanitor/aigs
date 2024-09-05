# %% lab_4.py
#    deep learning with JAX
# by: Noah Syrkis

# %% Imports ############################################################
from jax import grad, lax, random
import jax.numpy as jnp
from jaxtyping import Array
import chex
from typing import Callable, Dict, List, Tuple
import equinox as eqx
import sklearn.datasets as skd
import tensorflow_datasets as tfds

# %%
# faces = skd.fetch_lfw_people()
# data = jnp.array(faces.data)  # type: ignore
# target = jnp.array(faces.target)  # type: ignore

mnist = tfds.load("mnist", split="train")
x_data = jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).reshape(-1,1,28,28).astype(jnp.float32)
y_data = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])
rng = random.PRNGKey(1331)

# %% Convolutional neural networks ######################################
# Make a CNN classifying faces using JAX or Equinox.
# Compare the performance of your model to a simple neural network.
# define params
@chex.dataclass
class Params:
    kernel_1: Array
    kernel_2: Array
    fc_w_1: Array
    fc_w_2: Array
    fc_b_1: Array
    fc_b_2: Array

# %% initialize

def init(rng) -> Params:
    rng, *keys = random.split(rng, 10)
    weights = Params(
        kernel_1=random.normal(keys[0], (32,1,3,3)),
        kernel_2 = random.normal(keys[1], (16,32,3,3)),
        fc_w_1=random.normal(keys[1], (16, 16)),
        fc_b_1=random.normal(keys[2],16),
        fc_w_2=random.normal(keys[3], (16,10)),
        fc_b_2=random.normal(keys[4],10)
    )
    return weights

weights = init(rng)


# %% define layers

def conv2D(input: Array, kernel: Array, strides: Tuple=(1,1)) -> Array:
    return lax.conv(input, kernel, strides, padding='SAME')

def fc(input: Array, weights: Array, biases:Array, activation: Callable) -> Array:
    z = input @ weights + biases

    return activation(z)

# %% define function




# %% Autoencoders #######################################################
# Implement an autoencoder using JAX or Equinox.
# Look at the reconstruction error and the latent space.


# %% Variational Autoencoders ###########################################
# Implement a variational autoencoder using JAX or Equinox.
# Compare the performance of your model to a simple autoencoder.


# %% Bonus ###############################################################
# Take a selfie as your target image.
# Decode a random latent space vector to generate a new face.
# Compute the loss.
# Optimize the latent space vector to minimize the loss.
# Display the optimized latent space vector next to the target image of yourself.
