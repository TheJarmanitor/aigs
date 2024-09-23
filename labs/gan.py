# %% Imports
import tensorflow_datasets as tfds
from jax import lax, random, nn, tree, value_and_grad, grad, jit
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Hyperparaks
latent_space = 10
lr = 0.001
gen_layers = [latent_space, 128, 128, 784]
dis_layers = [784, 128, 128, 1]
steps = 1000
batch_size = 32

# %% load mnist
mnist = tfds.load("mnist", split="train")
x_data = jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).reshape(-1, 784) / 255.0
y_data = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])


# %%
def init_fn(rng, layers):
    params = []
    for i, o in zip(layers[:-1], layers[1:]):
        w = random.normal(rng, (i, o)) * 0.01
        b = jnp.zeros(o)
        params.append((w, b))
    return params


def apply_fn(params, inputs):
    for w, b in params:
        outputs = inputs @ w + b
        inputs = nn.relu(outputs)
    return nn.sigmoid(outputs)  # type: ignore


@jit
@value_and_grad
def gen_grad_fn(params, s, r):
    f = apply_fn(params["gen"], s)
    y_hat_dis = apply_fn(params["dis"], r)
    y_hat_gen = apply_fn(params["dis"], f)
    gen_loss = jnp.mean(jnp.log(y_hat_dis)) + jnp.mean(jnp.log(1 - y_hat_gen))
    return gen_loss


@jit
@value_and_grad
def dis_grad_fn(params, s, r):
    f = apply_fn(params["gen"], s)
    x = jnp.concatenate((s, r))
    y = jnp.array([0, 1]).repeat(batch_size)
    y_hat = apply_fn(params["dis"], x)
    dis_loss = jnp.square((y - y_hat)).mean()
    return dis_loss


def update_fn(params, s, r, grad_fn, model):
    loss, grads = grad_fn(params, s, r)
    params_model = tree.map(lambda p, g: p - lr * g, params[model], grads[model])
    return params_model, loss


# %% Init
rng, *keys = random.split(random.PRNGKey(0), 4)
dis_p = init_fn(keys[0], dis_layers)
gen_p = init_fn(keys[1], gen_layers)
params = {"dis": dis_p, "gen": gen_p}

fake_1 = apply_fn(gen_p, random.normal(keys[2], (batch_size, latent_space)))

for i in (pbar := tqdm(range(steps))):
    rng, *keys = random.split(rng, 3)
    s = random.normal(keys[0], (batch_size, latent_space))
    r = x_data[random.choice(keys[1], len(x_data), (batch_size,), replace=False)]
    gen_p, gen_loss = update_fn(params, s, r, gen_grad_fn, "gen")
    dis_p, dis_loss = update_fn(params, s, r, dis_grad_fn, "dis")
    params = {"dis": dis_p, "gen": gen_p}

    pbar.set_description(f"Dis Loss: {dis_loss:.2f}, Gen Loss: {gen_loss:.2f}")

rng, key = random.split(rng)
fake_2 = apply_fn(gen_p, random.normal(key, (batch_size, latent_space)))

fig, axes = plt.subplots(2)
axes[0].imshow(fake_1[0].reshape(28, 28))
axes[1].imshow(fake_2[0].reshape(28, 28))
plt.show()
