# %% libraries
import jax
from jax import random, grad, jit, vmap, lax, tree, nn
import jax.numpy as jnp
from jax.image import resize
import optax
import jumanji
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm


# %% Create environment
rng = jax.random.PRNGKey(0)

env = jumanji.make("Sokoban-v0")

# %% data gathering


def data_gathering(key, env, n_steps=10000):
    random_keys = random.split(key, n_steps)
    data_list = []
    for r in random_keys:
        state, timestep = env.reset(key)
        data_list.append(jnp.array((state["fixed_grid"], state["variable_grid"])))
    return jnp.array(data_list, dtype="float32")


data = data_gathering(rng, env)

print(data.shape)
# %%


class Autoencoder:
    def __init__(
        self, rng, layers, img_size, latent_size=100, kernel_shape=3, batch_size=1
    ) -> None:
        self.rng = rng
        self.layers = layers
        self.img_size = img_size
        self.latent_size = latent_size
        self.kernel_shape = kernel_shape
        self.batch_size = batch_size

    def init_kernel(self, key, in_size, out_size):
        init_fn = nn.initializers.glorot_uniform()
        kernel = init_fn(key, (out_size, in_size, self.kernel_shape, self.kernel_shape))
        return kernel

    def init_fc(self, key, in_size, out_size):

        init_fn = nn.initializers.orthogonal()
        weights = init_fn(key, (in_size, out_size))

        return weights

    def init_encoder(self, rng, channel_size):
        params = {"kernels": [], "fc": []}
        for i, o in zip(self.layers[:-1], self.layers[1:]):
            rng, key = random.split(rng)
            kernel = self.init_kernel(key, i, o)
            bias = jnp.zeros((1, 1, self.img_size, self.img_size))
            params["kernels"].append((kernel, bias))
        rng, key = random.split(rng)
        linear_weights = self.init_fc(
            key, self.layers[-1] * self.img_size**2, self.latent_size
        )
        weight_bias = jnp.zeros(self.latent_size)

        params["fc"].append((linear_weights, weight_bias))

        return params

    def conv2D(self, input, kernel, strides=(1, 1), padding="SAME"):
        return lax.conv(input, kernel, strides, padding)

    # def inverse_conv2D(self, input, kernel, strides=1, padding='SAME')

    def encoder(self, x_data, parameters):
        for kernel, bias in parameters["kernels"]:
            z = self.conv2D(x_data, kernel) + bias
            x_data = nn.relu(z)

        z = x_data.reshape(x_data.shape[0], -1)
        for weights, bias in parameters["fc"]:
            x_data = z @ weights + bias

        return x_data

    # %% Step 3: create a deconvolutional model that maps the latent space back to an image
    def init_decoder(self, rng, channel_size):
        params = {"kernels": [], "fc": []}
        r_layers = list(reversed(self.layers))
        rng, key = random.split(rng)
        linear_weights = self.init_fc(
            key, self.latent_size, r_layers[0] * self.img_size**2
        )
        weight_bias = jnp.zeros(r_layers[0] * self.img_size**2)
        params["fc"].append((linear_weights, weight_bias))
        for i, o in zip(r_layers[:-1], r_layers[1:]):
            rng, key = random.split(rng)
            kernel = self.init_kernel(key, i, o)
            bias = jnp.zeros((1, 1, self.img_size, self.img_size))
            params["kernels"].append((kernel, bias))
        return params

    def deconv2D(self, input, kernel, strides=(1, 1), padding="SAME"):
        return lax.conv_transpose(
            input, kernel, strides, padding, dimension_numbers=("NCHW", "OIHW", "NCHW")
        )

    def decoder(self, x_data, parameters):
        r_layers = list(reversed(self.layers))
        for weights, bias in parameters["fc"]:
            z = x_data @ weights + bias
            x_data = nn.relu(z)

        z = x_data.reshape(
            -1,
            r_layers[0],
            self.img_size,
            self.img_size,
        )

        for kernel, bias in parameters["kernels"]:
            x_data = self.deconv2D(z, kernel) + bias
            z = nn.relu(x_data)

        return z

    def forward(self, x_data, enc_params, dec_params):
        latent_space = self.encoder(x_data, enc_params)
        output = self.decoder(latent_space, dec_params)
        return output


# %%


def mse_loss(params, model, batch):
    imgs = batch
    recon_imgs = model(batch, params["encoder"], params["decoder"])
    loss = (
        ((recon_imgs - batch) ** 2).mean(axis=0).sum()
    )  # Mean over batch, sum over pixels
    return loss


def get_batch(data, batch_size):
    indices = jnp.arange(data.shape[0])
    sample_idxs = random.choice(rng, indices, shape=(batch_size,), replace=False)
    return data[sample_idxs]


# %%
latent_space = 100
layers = [2, 8, 16, 32, 64]
kernel_shape = 3

autoencoder = Autoencoder(rng, layers, 10, batch_size=128)

params = {
    "encoder": autoencoder.init_encoder(rng, 2),
    "decoder": autoencoder.init_decoder(rng, 2),
}
optim = optax.adam(learning_rate=0.000005)
opt_state = optim.init(params)
epochs = 1000

for i in (pbar := tqdm(range(epochs))):
    batch = get_batch(data, 128)
    loss, grad = jax.value_and_grad(mse_loss)(params, autoencoder.forward, batch)
    pbar.set_description("Processing %.2f" % loss)
    updates, opt_state = optim.update(grad, opt_state, params)

    params = optax.apply_updates(params, updates)
# %%
test = autoencoder.encoder(get_batch(data, 128), params["encoder"])
test.shape

# %%
test[15]
# %%

test_level = autoencoder.decoder(test[15], params["decoder"]).astype(int)

test_level.shape

# %%
state, _ = env.reset(rng)

state["fixed_grid"] = test_level[:, 0]
