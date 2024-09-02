# %% lab_4.py
#    deep learning with JAX
# by: Noah Syrkis

# %% Imports ############################################################
from jax import grad
import jax.numpy as jnp
from jaxtyping import Array
import chex
from typing import Callable, Dict, List
import equinox as eqx
import sklearn.datasets as skd

# %%
faces = skd.fetch_lfw_people()
data = jnp.array(faces.data)  # type: ignore
target = jnp.array(faces.target)  # type: ignore

# %% Convolutional neural networks ######################################
# Make a CNN classifying faces using JAX or Equinox.
# Compare the performance of your model to a simple neural network.


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
