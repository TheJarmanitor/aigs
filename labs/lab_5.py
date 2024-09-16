# %% lab_5.py
#     content generation lab
# by: Noah Syrkis

# %% Imports
import jax.numpy as jnp
from jax import random, grad, jit, vmap, lax, tree, nn
import gymnasium as gym, gymnax, pgx
import seaborn as sns
import matplotlib.pyplot as plt
# what ever other imports you need

# %% Step 1: choose any environment from gymnasium or gymnax that spits out an image like state

# %% Step 2: crate a convolutional model that maps the image to a latent space (like our MNIST classifier, except we won't classify anything)

# %% Step 3: create a deconvolutional model that maps the latent space back to an image

# %% Step 4: train the model to minimize the reconstruction error

# %% Step 5: generate some images by sampling from the latent space

# %% Step 6: visualize the images

# %% Step 7: (optional) try to interpolate between two images by interpolating between their latent representations

# %% Step 8: (optional) try to generate images that are similar to a given image by optimizing the latent representation

# %% Step 9: instead of mapping the image to a latent space, map the image to a distribution over latent spaces (VAE)

# %% Step 10: sample from the distribution over latent spaces and generate images

# %% Step 11: (optional) try to interpolate between two images by interpolating between their distributions over latent spaces

# %% Step 12: (optional) try to generate images that are similar to a given image by optimizing the distribution over latent spaces

# %% Step 13: (optional) try to switch out the VAE for a GAN
