# %% import libraries
from ast import increment_lineno
import kagglehub
from jax import lax, random, nn, tree, value_and_grad, grad, jit, vmap, tree_util
import jax.numpy as jnp
import optax
from optax.losses import sigmoid_binary_cross_entropy
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, List
from jax.typing import ArrayLike
%matplotlib inline

# %%
path = kagglehub.dataset_download("ebrahimelgazar/pixel-art")

print("Path to dataset files:", path)
# %%
data = jnp.load(path+"/sprites.npy")
labels = jnp.load(path+"/sprites_labels.npy")

plt.imshow(data[0])
# %%

class ConvLayer:
    def __init__(self, in_channel, out_channel, kernel_shape, stride=1, padding='SAME') -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding

    def init_params(self, key, init_fn=nn.initializers.glorot_uniform):
        rng, key = random.split(key)
        init_fn = init_fn()
        weights = init_fn(key, (self.out_channel, self.in_channel, self.kernel_shape, self.kernel_shape))
        bias = jnp.zeros((self.out_channel,))
        return (weights, bias)

    def forward(self, x, params):
        kernel, bias = params
        conv = lax.conv_general_dilated(
            x, kernel, (self.stride, self.stride), self.padding,
            dimension_numbers=("NHWC", "OIHW", "NHWC"))
        return conv + bias

class TransposeConvLayer(ConvLayer):
    def __init__(self, in_channel, out_channel, kernel_shape, stride=2, padding='SAME') -> None:
        ConvLayer.__init__(self, in_channel, out_channel, kernel_shape, stride, padding='SAME')

    def forward(self, x, params):
        kernel, bias = params
        deconv = lax.conv_transpose(
                    x, kernel, strides=(self.stride, self.stride), padding=self.padding,
                    dimension_numbers=("NHWC", "OIHW", "NHWC")
                )
        return deconv + bias

class DenseLayer:
    def __init__(self, in_features, out_features) -> None:
        self.in_features = in_features
        self.out_features = out_features

    def init_params(self, key, init_fn=nn.initializers.glorot_uniform):
        rng, key = random.split(key)
        init_fn = init_fn()
        weights = init_fn(key, (self.in_features, self.out_features))
        bias = jnp.zeros((self.out_features,))
        return (weights, bias)

    def forward(self, x, params):
        weights, bias = params
        features = x @ weights
        return features + bias


class BatchNormLayer:
    def __init__(self, features, epsilon=1e-5, momentum=0.9) -> None:
        self.features = features
        self.epsilon = epsilon
        self.momentum = momentum

    def init_params(self, key): #key is not used, but i'm lazy
        gamma = jnp.ones((1,1,1,self.features))
        beta = jnp.zeros((1,1,1,self.features))


        return (gamma, beta)

    def init_variables(self):
        running_mean = jnp.zeros((1,1,1,self.features))
        running_var = jnp.ones((1,1,1,self.features))

        return (running_mean, running_var)

    def forward(self, x, params, variables, training=True):
        gamma, beta = params
        running_mean, running_var = variables

        if training:
            batch_mean = jnp.mean(x, axis=(0, 1, 2), keepdims=False)
            batch_var = jnp.var(x, axis=(0, 1, 2), keepdims=False)

            x_hat = (x - batch_mean) / jnp.sqrt(batch_var + self.epsilon)

            new_running_mean = self.momentum * running_mean + (1 - self.momentum) * batch_mean
            new_running_var = self.momentum * running_var + (1 - self.momentum) * batch_var

            out = gamma * x_hat + beta
            return out, (new_running_mean, new_running_var)
        x_hat = (x - running_mean) / jnp.sqrt(running_var + self.epsilon)

        out = gamma * x_hat + beta
        return out, (running_mean, running_var)


class Generator:
    def __init__(self, rng, latent_dim, output_channel, hidden_features) -> None:
        self.layers = []
        self.batch_norm_layers = []
        self.latent_dim = latent_dim
        self.output_channel = output_channel
        self.hidden_features = hidden_features
        self.rng = rng

        self.add_transpose_conv_layer(self.latent_dim, self.hidden_features*4, 4)
        self.add_batch_norm_layer(self.hidden_features*4)
        self.add_transpose_conv_layer(self.hidden_features*4, self.hidden_features*2, 4)
        self.add_batch_norm_layer(self.hidden_features*2)
        self.add_transpose_conv_layer(self.hidden_features*2, self.hidden_features, 4)
        self.add_batch_norm_layer(self.hidden_features)
        self.add_transpose_conv_layer(self.hidden_features, self.output_channel, 4)



    def add_transpose_conv_layer(self,  input_channels, output_channels, kernel_shape, stride=2, padding="SAME"):
        self.layers.append(TransposeConvLayer(input_channels, output_channels, kernel_shape, stride, padding))

    def add_batch_norm_layer(self, num_features):
        self.batch_norm_layers.append(BatchNormLayer(num_features))


    def init_params(self):
        params = []
        for layer in self.layers + self.batch_norm_layers:
            rng, key = random.split(self.rng)
            params.append(layer.init_params(key))
        return params

    def init_variables(self):
        variables = []
        for layer in self.batch_norm_layers:
            variables.append(layer.init_variables())
        return variables

    def dropout_fn(self, key, x, dropout):
        keep_prob = 1 - dropout  # The probability of keeping a unit
        mask = random.bernoulli(key, keep_prob, shape=x.shape)  # Generate the dropout mask
        return jnp.where(mask, x / keep_prob, 0)

    def forward(self, x, params, variables, training=True):
        transposed_conv_params = params[:len(self.layers)]
        batch_norm_params = params[len(self.layers):]

        new_variables = []
        for i in range(len(self.layers[:-1])):
            x = self.layers[i].forward(x, transposed_conv_params[i])
            x, new_variable = self.batch_norm_layers[i].forward(x, batch_norm_params[i], variables[i], training)
            new_variables.append(new_variable)
            x = nn.relu(x)
        x = self.layers[-1].forward(x, transposed_conv_params[-1])
        if training:
            return nn.tanh(x), new_variables
        return nn.tanh(x)

class Discriminator:
    def __init__(self, rng, input_channels, hidden_features) -> None:
        self.conv_layers = []
        self.dense_layers = []
        self.input_channels = input_channels
        self.hidden_features = hidden_features
        self.rng = rng

        self.add_conv_layer(self.input_channels, self.hidden_features, 4)
        self.add_conv_layer(self.hidden_features, self.hidden_features*2, 4)
        self.add_conv_layer(self.hidden_features*2, self.hidden_features*4, 4)

        self.add_dense_layer(4*4*self.hidden_features, self.hidden_features)
        self.add_dense_layer(self.hidden_features, 1)


    def add_conv_layer(self, input_channels, output_channels, kernel_shape, stride=2, padding="SAME"):
        self.conv_layers.append(ConvLayer(input_channels, output_channels, kernel_shape, stride, padding))

    def add_dense_layer(self, input_features, output_features):
        self.dense_layers.append(DenseLayer(input_features, output_features))

    def init_params(self):
        params = []
        for layer in self.conv_layers+self.dense_layers:
            rng, key = random.split(self.rng)
            params.append(layer.init_params(key))
        return params



    def forward(self, x, params):
        conv_params = params[:len(self.conv_layers)]
        dense_params = params[len(self.conv_layers):]
        # for layer, param in zip(self.conv_layers, conv_params):
        for i in range(len(self.conv_layers[:-1])):
            x = self.conv_layers[i].forward(x, conv_params[i])
            x = nn.leaky_relu(x, 0.2)
        x = x.reshape((x.shape[0], -1))
        # x = x.reshape(-1, 4*4*self.hidden_features)
        # # for layer, param in zip(self.dense_layers[:-1], dense_params[:-1]):
        # for i in range(len(self.dense_layers[:-1])):
        #     x = self.dense_layers[i].forward(x, dense_params[i])
        #     x = nn.leaky_relu(x, 0.2)
        # x = self.dense_layers[-1].forward(x, dense_params[-1])
        return x
# %%



def generator_grad(g_params, d_params, g_variables, G_func, D_func, z):
    fake_images, new_g_variables = G_func.forward(z, g_params, g_variables)
    fake_logits = D_func.forward(fake_images, d_params)
    g_loss = -jnp.mean(fake_logits)
    return (g_loss, new_g_variables)


def discriminator_grad(d_params, g_params, g_variables, G_func, D_func, z, x):
    fake_images = G_func.forward(z, g_params, g_variables, training=False)

    fake_logits = D_func.forward(fake_images, d_params)
    real_logits = D_func.forward(x, d_params)

    # fake_loss = jnp.mean(sigmoid_binary_cross_entropy(logits=fake_logits, labels=jnp.zeros_like(fake_logits)))
    # real_loss = jnp.mean(sigmoid_binary_cross_entropy(logits=real_logits, labels=jnp.ones_like(real_logits)))
    d_loss = -(jnp.mean(real_logits) - jnp.mean(fake_logits))
    return d_loss


def update_step(params, grads, optimizer, opt_state):
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

def clip_weights(params, clip_value=0.01):
    return tree_util.tree_map(lambda p: jnp.clip(p, -clip_value, clip_value), params)


def get_batch(data, rng, batch_size):
    rng, key = random.split(rng)
    indices = jnp.arange(data.shape[0])
    sample_idxs = random.choice(rng, indices, shape=(batch_size,), replace=False)
    batch = data[sample_idxs]
    batch = (batch-jnp.min(data))/(jnp.max(data)-jnp.min(data))
    return batch
# %%

latent_dim = 256
img_channel = 3
rng_g = random.PRNGKey(1331)
rng_d = random.PRNGKey(3113)
rng_batch = random.PRNGKey(2710)


G = Generator(rng_g, latent_dim, img_channel, 64)

D = Discriminator(rng_d, img_channel, 64)

g_params = G.init_params()
d_params = D.init_params()

g_variables = G.init_variables()

g_optimizer = optax.adam(2e-4, 0.9, 0.99)
# d_optimizer = optax.adam(2e-4, 0.5, 0.999)
d_optimizer = optax.rmsprop(learning_rate=5e-5)


g_opt_state = g_optimizer.init(g_params)
d_opt_state = d_optimizer.init(d_params)
epochs = 500
batch_size = 128
clip_value=0.25


# %%
for _ in (pbar := tqdm(range(epochs))):


    rng_g, key = random.split(rng_g)
    batch_real_images = get_batch(data, rng_batch, batch_size)
    batch_real_images = (batch_real_images - 0.5) / 0.5
    z = random.normal(key, (latent_dim,batch_size)).reshape(-1,1,1,latent_dim)

    d_loss, d_grad = value_and_grad(discriminator_grad)(d_params, g_params, g_variables, G, D, z, batch_real_images)
    (g_loss, new_g_variables), g_grad = value_and_grad(generator_grad, has_aux=True)(g_params, d_params, g_variables, G, D, z)

    pbar.set_description(f"G Loss: {g_loss:.3f} | D Loss: {d_loss:.3f}")

    d_params, d_opt_state = update_step(d_params, d_grad, d_optimizer, d_opt_state)
    g_params, g_opt_state = update_step(g_params, g_grad, g_optimizer, g_opt_state)

    g_variables = new_g_variables
    d_params = clip_weights(d_params, clip_value=clip_value)




# %%
rng_g, key = random.split(rng_g)
test_z = random.normal(key, (latent_dim,)).reshape(-1,1,1,latent_dim)

test_image = G.forward(test_z, g_params, g_variables, training=False)

plt.imshow(0.5*test_image[0]+0.5)
 # %%
