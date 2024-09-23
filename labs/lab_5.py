# %% lab_5.py
#     content generation lab
# by: Noah Syrkis

# %% Imports
import jax.numpy as jnp
from jax import random, grad, jit, vmap, lax, tree, nn
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
# what ever other imports you need

# %% Step 1: choose any environment from gymnasium or gymnax that spits out an image like state
env = gym.make("CarRacing-v2", domain_randomize=True)  # select environment
env.reset()  # reset environment (have to do this for some reason)
obss = [env.step(env.action_space.sample())[0] for _ in range(1000)]  # take n steps
data = jnp.array(obss)  # put it in jnp.array
data.shape


# %% Step 2: crate a convolutional model that maps the image to a latent space (like our MNIST classifier, except we won't classify anything)
def init_model(rng, chans, shaps):
    keys = random.split(rng, 3)

    def aux(key, shp, fn):
        return [(fn(key, i, o), jnp.zeros((o,))) for i, o in zip(shp[:-1], shp[1:])]

    c_fn = lambda rng, c_in, c_out: random.normal(rng, (c_out, c_in, 3, 3))
    f_fn = lambda rng, c_in, c_out: random.normal(rng, (c_in, c_out))
    return [aux(*a) for a in zip(keys, [chans, shaps, chans[::-1]], [c_fn, f_fn, c_fn])]


def model(params, image):
    # for param, fn in zip(params, [conv, jnp.dot, devonv]):
    # for w, b in param:
    # image = nn.tanh(fn(w, image) + b)
    for w, b in params[0]:
        image = nn.relu(conv(w, image) + b)
    image = image.reshape((image.shape[0], -1))
    for w, b in params[1]:
        image = nn.relu(jnp.dot(image, w) + b)
    image = image.reshape((image.shape[0], 8, 8, 128))
    for w, b in params[2]:
        image = nn.relu(devonv(w, image) + b)
    return nn.sigmoid(image)


def conv(kernel, image):
    return lax.conv(image, kernel, (2, 2), "SAME")


def devonv(kernel, image):
    return lax.conv_transpose(image, kernel, (1, 1), "SAME")


params = init_model(random.PRNGKey(0), [3, 32, 64, 128], [128, 64, 32, 3])
recon = model(params, data)

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
