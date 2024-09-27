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

# %% Step 2: crate a convolutional model that maps the image to a latent space (like our MNIST classifier, except we won't classify anything)
class Autoencoder():
    def __init__(self, rng, layers, img_size, latent_size=100, kernel_shape=3) -> None:
        self.rng = rng
        self.layers = layers
        self.img_size = img_size
        self.latent_size = latent_size
        self.kernel_shape = kernel_shape

    def init_kernel(self, key, in_size, out_size):
        init_fn = nn.initializers.glorot_uniform()
        kernel = init_fn(key, (out_size, in_size, self.kernel_shape, self.kernel_shape))
        return kernel

    def init_fc(self, key, in_size, out_size):

        init_fn = nn.initializers.orthogonal()
        weights = init_fn(key, (in_size, out_size))

        return weights


    def init_encoder(self, rng, channel_size):
        params = {
            "kernels": [],
            "fc": []
        }
        for i, o in zip(self.layers[:-1], self.layers[1:]):
            rng, key  = random.split(rng)
            kernel = self.init_kernel(key, i, o)
            bias = jnp.zeros((1,1, self.img_size, self.img_size))
            params["kernels"].append((kernel, bias))
        rng, key = random.split(rng)
        linear_weights = self.init_fc(key, self.layers[-1]*self.img_size**2, self.latent_size)
        weight_bias = jnp.zeros(self.latent_size)

        params["fc"].append((linear_weights, weight_bias))

        return params


    def conv2D(self, input, kernel, strides=(1,1), padding='SAME'):
        return lax.conv(input, kernel, strides, padding)


    # def inverse_conv2D(self, input, kernel, strides=1, padding='SAME')

    def encoder(self, x_data, parameters):
        for kernel, bias in parameters["kernels"]:
            z = self.conv2D(x_data, kernel) + bias
            x_data = nn.relu(z)

        z = x_data.reshape(1,-1)
        for weights, bias in parameters["fc"]:
           x_data = z @ weights + bias

        return x_data



    # %% Step 3: create a deconvolutional model that maps the latent space back to an image
    def init_decoder(self, rng, channel_size):
        params = {
            "kernels": [],
            "fc": []
        }
        r_layers = list(reversed(self.layers))
        rng, key = random.split(rng)
        linear_weights = self.init_fc(key, self.latent_size, r_layers[0]*self.img_size**2)
        weight_bias = jnp.zeros(r_layers[0]*self.img_size**2)
        params["fc"].append((linear_weights, weight_bias))
        for i, o in zip(r_layers[:-1], r_layers[1:]):
            rng, key  = random.split(rng)
            kernel = self.init_kernel(key, i, o)
            bias = jnp.zeros((1,1,self.img_size, self.img_size))
            params["kernels"].append((kernel, bias))
        return params


    def deconv2D(self, input, kernel, strides=(1,1), padding='SAME'):
        return lax.conv_transpose(input, kernel, strides, padding, dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

    def decoder(self, x_data, parameters):
        r_layers = list(reversed(self.layers))
        for weights, bias in parameters["fc"]:
            z = x_data @ weights + bias
            x_data = nn.relu(z)

        z = x_data.reshape(1, r_layers[0], self.img_size, -1,)

        for kernel, bias in parameters["kernels"]:
            x_data = self.deconv2D(z, kernel) + bias
            z = nn.relu(x_data)

        return z

rng = random.PRNGKey(1331)
latent_space = 100
layers = [1, 8, 16, 32, 64]
kernel_shape = 3
obs_reshape = resize(obs, (128,128), method="linear")
obs_transformed = jnp.reshape(obs_reshape, (1,1,128,128))/255

autoencoder = Autoencoder(rng, layers, 128)


kernel_list_enc = autoencoder.init_encoder(rng, 1)
kernel_list_dec = autoencoder.init_decoder(rng, 1)

print()

latent_space = autoencoder.encoder(obs_transformed, kernel_list_enc)

resulting_image = autoencoder.decoder(latent_space, kernel_list_dec)
print(resulting_image.shape)

plt.imshow(resulting_image.reshape(128, 128))

# test_kernel = kernel_list_dec["kernels"][0][0]
# test_kernel.shape

# autoencoder.deconv2D(resulting_image, test_kernel)


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
