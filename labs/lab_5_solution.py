# %% lab_5.py
#     content generation lab
# by: Noah Syrkis

# %% Imports
import jax.numpy as jnp
from jax import random, grad, jit, vmap, lax, tree, nn
from jax.image import resize
import gymnasium as gym, gymnax, pgx
import seaborn as sns
import matplotlib.pyplot as plt
import ale_py

# what ever other imports you need

# %% Step 1: choose any environment from gymnasium or gymnax that spits out an image like state
gym.register_envs(ale_py)
env = gym.make("ALE/Frogger-v5", obs_type="grayscale")
obs, info = env.reset()

print(obs.shape)
plt.imshow(obs)
# %% Step 2: crate a convolutional model that maps the image to a latent space (like our MNIST classifier, except we won't classify anything)
class Autoencoder():
    def __init__(self, rng) -> None:
        self.rng = rng

    def init_kernel(self, key, in_size, out_size, kernel_shape=3):
        kernel = nn.initializers.glorot_uniform()(key, (out_size, in_size, kernel_shape, kernel_shape))
        return kernel

    def init_encoder(self, rng, channel_size, layers, kernel_shape):
        kernels = []
        for i, o in zip(layers[:-1], layers[1:]):
            rng, key  = random.split(rng)
            kernel = self.init_kernel(key, i, o, kernel_shape)
            bias = jnp.zeros(kernel.shape)
            kernels.append((kernel, bias))
        return kernels

        return kernels

    def conv2D(self, input, kernel, strides=(1,1), padding='SAME'):
        return lax.conv(input, kernel, strides, padding)

    # def inverse_conv2D(self, input, kernel, strides=1, padding='SAME')

    def encoder(self, x_data, parameters, kernel_size):
        for kernel, bias in parameters:
            z = self.conv2D(x_data, kernel) + bias
            x_data = nn.relu(z)
        return x_data

rng = random.PRNGKey(1331)
encoder = Autoencoder(rng)
layers = [1, 16, 32, 64, 128]

obs_reshape = resize(obs, (128,128), method="linear")

kernel_list = encoder.init_encoder(rng, 1, layers, kernel_shape=3)
obs_transformed = jnp.reshape(obs_reshape, (1,1,128,128))

# encoder.conv2D(obs_transformed, kernel_list[0])

result_image = encoder.encoder(obs_transformed, kernel_list, 3)

plt.imshow(result_image[0,5])

# %% Step 3: create a deconvolutional model that maps the latent space back to an image
#
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
