# %% libraries
import jax
from jax import random, grad, jit, vmap, lax, tree, nn
import jax.numpy as jnp
from jax.image import resize
import optax
import jumanji
from jumanji.wrappers import AutoResetWrapper
from functools import partial
import matplotlib.pyplot as plt


# %% Create environment
rng = jax.random.PRNGKey(0)

env = jumanji.make("Sokoban-v0")
env = AutoResetWrapper(env)

# %% data gathering


def step_fn(state, key, env):
    num_actions = env.action_spec.num_values
    action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=())
    new_state, timestep = env.step(state, action)
    return new_state, timestep


def data_gathering(key, step_fn, env, n_steps=1000):
    state, timestep = env.reset(key)
    random_keys = random.split(key, n_steps)
    data_list = []
    for r in random_keys:
        new_state, _ = step_fn(state, r, env)
        data_list.append(
            jnp.array((new_state["fixed_grid"], new_state["variable_grid"]))
        )
        # print("data added")
        state = new_state
    return jnp.array(data_list, dtype="float32")


state, _ = env.reset(rng)

data = data_gathering(rng, step_fn, env)

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

        z = x_data.reshape(self.batch_size, -1)
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
            self.batch_size,
            r_layers[0],
            self.img_size,
            -1,
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


@jax.jit
def mse_loss(model, params, batch):
    imgs, _ = batch
    recon_imgs = model(batch, *params)
    loss = (
        ((recon_imgs - imgs) ** 2).mean(axis=0).sum()
    )  # Mean over batch, sum over pixels
    return loss


optim = optax.adam(learning_rate=0.0005)

# %%
latent_space = 100
layers = [2, 8, 16, 32, 64]
kernel_shape = 3

autoencoder = Autoencoder(rng, layers, 10, batch_size=32)

params = {
    "encoder": autoencoder.init_encoder(rng, 2),
    "decoder": autoencoder.init_decoder(rng, 2),
}

indices = jnp.arange(data.shape[0])
sample_idxs = random.choice(
    rng, indices, shape=(autoencoder.batch_size,), replace=False
)
state_example = data[sample_idxs]

result_level = autoencoder.forward(state_example, params["encoder"], params["decoder"])

print(result_level.shape)
