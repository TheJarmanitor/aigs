# %% libraries
import jax
from jax import random, grad, jit, vmap, lax, tree, nn
from jumanji.environments.routing.sokoban.env import State
from jumanji.environments.routing.sokoban.generator import DeepMindGenerator
import jax.numpy as jnp
from jax.image import resize
import matplotlib
import optax
import jumanji
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

%matplotlib inline


# %% Create environment
rng = jax.random.PRNGKey(0)

env = jumanji.make("Sokoban-v0")

# %% data gathering


def data_gathering(key, env, n_steps=3000, scale=False):
    random_keys = random.split(key, n_steps)
    data_list = []
    for r in random_keys:
        state, timestep = env.reset(r)
        data_list.append(jnp.array((state["fixed_grid"], state["variable_grid"])))
    if scale:
        return jnp.array(data_list, dtype="float32")/4.0
    return jnp.array(data_list, dtype="float32")


data = data_gathering(rng, env)

print(data.shape)
# %%


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
            x_data = nn.selu(z)

        z = x_data.reshape(x_data.shape[0], -1)
        for weights, bias in parameters["fc"]:
            x_data = z @ weights + bias

        return nn.tanh(x_data)

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
            z = nn.selu(z)

        x_data = z.reshape(
            -1,
            r_layers[0],
            self.img_size,
            self.img_size,
        )

        for kernel, bias in parameters["kernels"]:
            z = self.deconv2D(x_data, kernel) + bias
            x_data = nn.selu(z)

        return x_data

    def forward(self, x_data, enc_params, dec_params):
        latent_space = self.encoder(x_data, enc_params)
        output = self.decoder(latent_space, dec_params)
        return output


# %%


def mse_loss(params, model, batch):
    imgs = batch
    recon_imgs = model(batch, params["encoder"], params["decoder"])
    loss = (
        ((recon_imgs - imgs) ** 2).mean(axis=0).sum()
    )  # Mean over batch, sum over pixels
    return loss


def get_batch(data, batch_size):
    indices = jnp.arange(data.shape[0])
    sample_idxs = random.choice(rng, indices, shape=(batch_size,), replace=False)
    return data[sample_idxs]


# %%
latent_space = 100
layers = [2, 8, 16, 32, 64, 128]
kernel_shape = 3

autoencoder = Autoencoder(rng, layers, 10, batch_size=128)

params = {
    "encoder": autoencoder.init_encoder(rng, 2),
    "decoder": autoencoder.init_decoder(rng, 2),
}
optim = optax.adam(learning_rate=0.0005)
opt_state = optim.init(params)
epochs = 1000

for i in (pbar := tqdm(range(epochs))):
    batch = get_batch(data, 128)
    loss, grad = jax.value_and_grad(mse_loss)(params, autoencoder.forward, batch)
    pbar.set_description("Current loss   %.3f" % loss)
    updates, opt_state = optim.update(grad, opt_state, params)

    params = optax.apply_updates(params, updates)
# %%

rng2  = jax.random.PRNGKey(1)
test_data = data_gathering(rng2, env, 32)
test = autoencoder.encoder(test_data, params["encoder"])

print(test[0])
# %%
key, rng = random.split(rng)
random_space = random.choice(key, jnp.array(range(test.shape[0])))
test_level = jnp.round(autoencoder.decoder(test[random_space], params["decoder"]))
def custom_env_state(rng, fixed_grid, variable_grid, agent_location=(0,0)):
    state = State(key=rng
            ,fixed_grid=fixed_grid
            ,variable_grid=variable_grid
            ,agent_location=jnp.array(agent_location)
            ,step_count= jnp.array(0))
    return state

rng, key = random.split(rng)
test_level_state = custom_env_state(key, test_level[0,0], test_level[0,1])

# done_level, _ = env.reset(rng)
    # done_level["fixed_grid"]
env.render(test_level_state)
# %%
remade_level = test_data[random_space]
# remade_level[0]
remade_level_state = custom_env_state(key, remade_level[0], remade_level[1])


# remade_level_state
env.render(remade_level_state)
