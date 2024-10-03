# %% Imports
import gymnasium as gym
from tqdm import tqdm
from jax import random, nn, jit, tree, value_and_grad, grad
import jax.numpy as jnp
from collections import deque
import optax

# %% Constants
batch_size = 128
episodes = 600
gamma = 0.9
lr = 3e-4
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999
memory_size = 10000
layers = [4, 128, 128, 2]


# %% Model
def init_fn(rng, layers):
    params = []
    for i, o in zip(layers[:-1], layers[1:]):
        w = random.normal(rng, (i, o)) * 0.01
        b = jnp.zeros(o)
        params.append((w, b))
    return params


@jit
def apply_fn(params, inputs):
    for w, b in params:
        outputs = inputs @ w + b
        inputs = nn.relu(outputs)
    return outputs  # type: ignore


def action_fn(key, params, state, epsilon):
    if random.uniform(key) < epsilon:
        return random.randint(key, (), 0, 2).item()
    return apply_fn(params, state).argmax().item()


# grad_fn

# %% Init
env = gym.make("CartPole-v1")
memory = deque(maxlen=memory_size)
rng = random.PRNGKey(0)
params = init_fn(rng, layers)
opt = optax.adam(lr)
opt_state = opt.init(params)


# sample_fn
def sample_fn(key, memory):
    idxs = random.choice(key, len(memory), (batch_size,), replace=False)
    batch = zip(*(memory[i] for i in idxs))
    return tuple(map(jnp.array, batch))


@value_and_grad
def grad_fn(params, s, a, r, ns, t):
    pred = apply_fn(params, s)[jnp.arange(batch_size), a]
    targ = r + gamma * apply_fn(params, ns).max(axis=-1) * (1 - t)
    loss = ((pred - targ) ** 2).mean()
    return loss


@jit
def update_fn(params, batch, opt_state):
    loss, grads = grad_fn(params, *batch)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    # params = tree.map(lambda p, g: p - lr * g, params, grads)
    return params, opt_state, loss


# %% Training

# for loop through episodes
for i in (pbar := tqdm(range(episodes))):
    (state, _), done, truncated = env.reset(seed=i), False, False
    while not done and not truncated:
        rng, key = random.split(rng)
        action = action_fn(key, params, state, epsilon)
        new_state, reward, done, truncated, _ = env.step(action)
        memory.append((state, action, reward + truncated, state := new_state, done))  # type: ignore
        if len(memory) == memory_size:
            rng, key = random.split(rng)
            batch = sample_fn(key, memory)
            params, opt_state, loss = update_fn(params, batch, opt_state)
            epsilon = max(epsilon_min, epsilon_decay * epsilon
            pbar.set_description(f"Loss: {loss:.2f}, Epsilon: {epsilon:.2f}")
env.close()


# %%  # run 10 runs with env
env = gym.make("CartPole-v1", render_mode="human")
# run 10 episodes with the apply_fn
for i in range(10):
    (state, _), done, truncated = env.reset(seed=i), False, False
    while not done and not truncated:
        action = apply_fn(params, state).argmax().item()
        state, _, done, truncated, _ = env.step(action)
env.close()
