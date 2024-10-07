# %% Imports (add as needed) #############################################
import gymnasium as gym  # not jax based
from jax import random, vmap, nn, jit
from jax import numpy as jnp
from tqdm import tqdm
from collections import deque

# %% Constants ###########################################################
env = gym.make("CartPole-v1")
memory = deque(maxlen=1000)  # <- replay buffer
gamma = 0.99  # <- discount factor

# %% Model ###############################################################
def params_fn(key):
    keys = random.split(key, 4)
    w1 = random.normal(key, (4, 32))
    b1 = random.normal(key, (32,))
    w2 = random.normal(key, (32, 2))
    b2 = random.normal(key, (2,))  #
    return (w1, w2, b1, b2)

def model_fn(params, state):  # f(x) = wx + b  # <-  it just this
    w1, w2, b1, b2 = params
    z = state @ w1 + b1  # <- z is my intermediate variable
    z = z @ w2 + b2
    return z  # <- return the prediction value of each action

# %% Model ###############################################################
def sample_batch(rng, memory, batch_size):
    idxs = random.randint(rng, (batch_size,), 0, len(memory))
    batch = [memory[i] for i in idxs]
    return list(map(jnp.array, zip(*batch)))

# %% Environment #########################################################
rng = random.PRNGKey(0)
params = params_fn(rng)  # init params
state, info = env.reset()  # init env

for i in tqdm(range(10 ** 2)):  # run env to completion 100 times
    terminated, truncated = False, False
    while not (terminated | truncated):                                     # while not fallen over or run for too long
        action = model_fn(params, state).argmax().item()                    # get action with higest q-value from model
        next_state, reward, terminated, truncated, info = env.step(action)  # take action in env and get next state
        memory.append((state, action, reward, next_state, terminated))      # store experience in memory
    if len(memory) > 32:
        batch = sample_batch(rng, memory, 32)  # sample a batch of experiences
        params = update_fn(params, batch)  # update the model with the batch

env.close()
# %%


# %%
state, action, reward, next_state, terminated = batch
td_error = reward + gamma *  model_fn(params, next_state).max(axis=1) - model_fn(params, state)[jnp.arange(32), action]
