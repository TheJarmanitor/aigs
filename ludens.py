# %% Imports
import jax.numpy as jnp  # <- the core of jax are jnp.ndarrays
from jax import grad, random, nn, tree, jit
from tqdm import tqdm  # <- for progress bars
import tensorflow_datasets as tfds  # <- for data
import seaborn as sns
import matplotlib.pyplot as plt

# %% Load the data
mnist = tfds.load("mnist", split="train")  # <- load the mnist dataset
xs, ys = [], []  # <- create empty lists to store the data
for sample in tfds.as_numpy(mnist):  # <- iterate over the dataset
    x = sample["image"]  # <- extract the image
    y = sample["label"]  # <- extract the label
    xs.append(x)  # <- append the image to the list
    ys.append(y)  # <- append the label to the list
x = jnp.array(xs).reshape(60_000, 28 * 28) / 255.0  # <- convert the list to a jnp.ndarray
y = jnp.eye(10)[jnp.array(ys)]
batch_size = 32
x, y = x.reshape(-1, batch_size, 28 * 28), y.reshape(-1, batch_size, 10)

# %%
def model(x, params):
    w1, w2, b1, b2 = params        # <- assign the parameters to convenient names
    z = x @ w1 + b1                # <- linear transformation layer one
    z = nn.relu(z)                 # <- activation layer one
    z = z @ w2 + b2                # <- linear transformation layer two
    y_hat = nn.softmax(z, axis=1)  # <- turn rows into probabilities
    return y_hat                   # <- return the prediction

def loss_fn(param, x, y):
    y_hat = model(x, param)             # <- get the prediction from the model with the given parameters
    loss =  ((y - y_hat) ** 2).mean()   # <- calculate the mean squared error
    return loss

grad_fn = grad(loss_fn)   # <- get the gradient function of the loss function



keys = random.split(random.PRNGKey(0), 4)              # <- split the random key into 4 keys
w1 = random.normal(keys[0], (28 * 28, 32)) * 0.01      # <- initialize the weights of the first layer
w2 = random.normal(keys[1], (32, 10)) * 0.01           # <- initialize the weights of the second layer
b1 = random.normal(keys[2], (32,)) * 0.01              # <- initialize the bias of the first layer
b2 = random.normal(keys[3], (10,)) * 0.01              # <- initialize the bias of the second layer
params = (w1, w2, b1, b2)                              # <- pack the parameters into a tuple

# %%

def update(parameter, gradient):
    return parameter - 0.01 * gradient  # <- update the parameter with a learning rate of 0.01

pbar = tqdm(range(20))                              # <- create a progress bar (20 epochs)
for epoch in pbar:                                  # <- iterate through the dataset multiple times
    for x_batch, y_batch in zip(x, y):              # <- iterate over the batches
        grads = grad_fn(params, x_batch, y_batch)   # <- calculate the gradients
        params = tree.map(update, params, grads)    # <- update the parameters to lower loss
        loss = loss_fn(params, x_batch, y_batch)    # <- calculate the loss
        pbar.set_description(f"Loss: {loss:.2f}")

# %%
y_hat = model(x, params)
(y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
