# %% import libraries
from ast import increment_lineno
import kagglehub
from jax import lax, random, nn, tree, value_and_grad, grad, jit, vmap
import jax.numpy as jnp
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
plt.imshow(data[1])
# %%

class ConvLayer:
    def __init__(self, in_channel, out_channel, kernel_shape, stride=1, padding='SAME') -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self.params = None

    def init_params(self, key, init_fn=nn.initializers.glorot_uniform):
        rng, key = random.split(key)
        weights = init_fn(key, (self.in_channel, self.out_channel, self.kernel_shape, self.kernel_shape))
        bias = jnp.zeros((self.out_channel,))
        self.params = (weights, bias)

    def forward(self, x):
        assert self.params is not None
        weights, bias = self.params
        conv = lax.conv(x, weights, (self.stride, self.stride), self.padding)
        return conv + bias

class TransposeConvLayer(ConvLayer):
    def __init__(self, in_channel, out_channel, kernel_shape, stride=2, padding='SAME') -> None:
        ConvLayer.__init__(self, in_channel, out_channel, kernel_shape, stride, padding='SAME')

    def forward(self, x):
        assert self.params is not None
        weights, bias = self.params
        deconv = lax.conv_transpose(
            x, weights, (self.stride, self.stride), self.padding, dimension_numbers=("NCHW", "OIHW", "NCHW")
        )
        return deconv + bias

class DenseLayer:
    def __init__(self, in_features, out_features) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.params = None

    def init_params(self, key, init_fn=nn.initializers.glorot_uniform):
        weights = init_fn(key, (self.in_features, self.out_features))
        bias = jnp.zeros((self.out_features,))
        self.params = (weights, bias)


class Generator:
    def __init__(self) -> None:
        pass

class Discriminator:
    def __init__(self) -> None:
        pass
