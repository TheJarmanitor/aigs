# %% lab_3.py
#    deep learning with JAX
# by: Noah Syrkis

# %% Imports ############################################################
from jax import grad, lax, random
from jax import grad, lax, random, nn, tree, jit
import jax.numpy as jnp
from tqdm import tqdm
from jaxtyping import Array
import chex
from typing import Callable, Dict, List, Tuple
import equinox as eqx
from typing import Callable, Dict, List
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

import os
import tensorflow_datasets as tfds
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


# %% Constants
batch_size = 32

# %% Load celebA dataset
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


# 1. MAKE AND INIT PARAMS FUNCTION THAT RETURNS A DICTIONARY WITH KERNELS AND MLP PARAMS
# 2. MAKE AN APPLY FUNCTION THAT TAKES IN THE PARAMS AND INPUTS AND RETURNS THE OUTPUT
# NOTE: DO BOTH THE CONV STUFF, AND THE FULLY CONNECTED STUFF. REMEMBER ACTIVATION FUNCTIONS.
# 3. AT THIS POINT YOU HAVE AN OUTPUT "Y_HAT" FROM THE MODEL. USE GRADS TO OPTIMIZE WEIGHTS.

# %% Load data
mnist = tfds.load('mnist', split='train')
x_data = jnp.array([x['image'] for x in tfds.as_numpy(mnist)]).astype(jnp.float32)
y_data = jnp.array([x['label'] for x in tfds.as_numpy(mnist)]).astype(jnp.int32)

# %%  Reshape data to baches
x_data = x_data.reshape(-1, batch_size, 1, 28, 28) / 255.0  # <- normalize
y_data = y_data.reshape(-1, batch_size)

# %% Define convolutional functions
def conv(input: Array, kernel: Array, stride: int = 1) -> Array:
    return lax.conv(input, kernel, (stride, stride), padding='SAME')


# %% Define the model init
def init_fn(rng, cfg={"fc": [1568, 128, 10]}):
    """
    Initialize the parameters of the model in any pytree (list, tuple, dict, etc.) structure you like.
    We are using dictionaries here. cfg is a dictionary that contains (some) configuration of the model.
    Better to use cfg than hardcoding the shapes.
    """
    rng, *keys = random.split(rng, 10)
    kernel_1 = random.normal(keys[0], (3, 1, 3, 3))
    kernel_2 = random.normal(keys[1], (2, 3, 3, 3))
    w1 = random.normal(keys[2], (cfg["fc"][0], cfg["fc"][1]))
    b1 = jnp.zeros(cfg["fc"][1])
    w2 = random.normal(keys[3], (cfg["fc"][1], cfg["fc"][2]))
    b2 = jnp.zeros(cfg["fc"][2])
    params = dict(kernel1=kernel_1, kernel2=kernel_2, mlp=[(w1, b1), (w2, b2)])
    return params


# %% Define the model apply
def apply_fn(params, x):
    """
    Apply the model to the input x using the parameters params.
    We thus go through the forward pass of the model, step by step.
    We can do so however we like, as long as we are consistent with the shapes.
    Here I do a few things explicitly, like reshaping the output of the second convolution.
    I loop through the MLP layers, but you could do this however you like.
    REMEMBER: this function will be used in the loss function, which we will differentiate with grad.
    This means JAX can take derivatives of functions with loops!
    """
    z = conv(x, params['kernel1'])
    z = jnp.tanh(z)
    z = conv(z, params['kernel2'])
    z = z.reshape(z.shape[0], -1)
    for w, b in params['mlp']:
        z = jnp.dot(z, w) + b
        z = jnp.tanh(z)
    return nn.softmax(z)

# %% Define the loss function
def loss_fn(params, x_data, y_data):  # y: n
    """
    Compute the loss of the model on the input data x_data with labels y_data.
    We use the mean squared error loss here, but you can use any loss you like.
    We also one-hot encode the labels y_data, as we are doing multiclass classification.
    """
    y_hat = apply_fn(params, x_data)
    y_data = jnp.eye(10)[y_data]  # <- one-hot encode  # y: n x 10
    loss = jnp.mean((y_hat - y_data) ** 2)
    return loss

# %% Call it all
rng = random.PRNGKey(0)
params = init_fn(rng)

grad_fn = jit(grad(loss_fn))
for epoch in range(5): # <- epochs
    for x, y in tqdm(zip(x_data, y_data), total=len(x_data)):  # <- batches
        grads = grad_fn(params, x, y)  # <- compute gradients
        params = tree.map(lambda p, g: p - 0.01 * g, params, grads)  # <- update parameters



# %% Autoencoders #######################################################
# Implement an autoencoder using JAX or Equinox.
# Look at the reconstruction error and the latent space.

def deconv(input: Array, kernel: Array, stride: int = 1) -> Array:  # <- transpose convolution (deconvolution) won't use for now
    return lax.conv_transpose(input, kernel, (stride, stride), padding='SAME')



# %% Variational Autoencoders ###########################################
# Implement a variational autoencoder using JAX or Equinox.
# Compare the performance of your model to a simple autoencoder.


# %% Bonus ###############################################################
# Take a selfie as your target image.
# Decode a random latent space vector to generate a new face.
# Compute the loss.
# Optimize the latent space vector to minimize the loss.
# Display the optimized latent space vector next to the target image of yourself.
