# %% Imports (add as needed) #############################################
from functools import partial
import gymnasium as gym  # not jax based
from jax import random, nn, jit, vmap
from jax._src.api import jit
import jax.numpy as jnp
from numpy.random.mtrand import gamma
from tqdm import tqdm
from collections import deque, namedtuple
import chex
from copy import deepcopy

# %% Constants ###########################################################
env = gym.make("MountainCar-v0", render_mode="human")  #  render_mode="human")
rng = random.PRNGKey(1331)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=1000)  # <- replay buffer
gamma = 0.99
epsilon = 0.2
sample_size = 32
# define more as needed
# %%
print(env.observation_space.__dict__)


# %% Model ###############################################################
rng, *keys = random.split(rng, 10)


@chex.dataclass
class Params:
    w1 = random.normal(keys[0], (env.observation_space._shape[0], 16))
    b1 = random.normal(keys[1], 16)
    w2 = random.normal(keys[2], (16, 16))
    b2 = random.normal(keys[3], 16)
    w3 = random.normal(keys[4], (16, env.action_space.n))
    b3 = random.normal(keys[5], int(env.action_space.n))

test_params = Params()

print(test_params.w3.shape)
def model(params: Params, x_data):
    z = x_data @ params.w1 + params.b1
    z = nn.relu(z)
    z = z @ params.w2 + params.b2
    z = nn.relu(z)
    z = z @ params.w3 + params.b3
    z = nn.softmax(z)

    # return jnp.argmax(z)
    return z


# @partial(vmap, in_axes=(None, None, None, 0, 0, 0, None, 0))
def td_loss(params, target_params, model, obs, next_obs, action, gamma, reward):
    target = gamma * jnp.max(model(target_params, next_obs))
    prediction = model(params, obs)[action]
    return (reward + target - prediction)**2


def random_policy_fn(rng): # action (shape: ())
    n = env.action_space.__dict__['n']
    return random.randint(rng, (1,), 0, n).item()

def e_greedy_policy_fn(rng, q_a, epsilon, decay=0.99):  # obs (shape: (2,)) to action (shape: ())
    rng, key = random.split(rng)
    random_action = random_policy_fn(rng)
    action = jnp.argmax(q_a)
    final_action = random.choice(key, jnp.array([action, random_action]), p=jnp.array([1-epsilon*decay, epsilon*decay]))
    return int(final_action)

# def update():
#     raise NotImplementedError

# %% test
rng, key = random.split(rng)
obs, info = env.reset()
q_a = model(Params(), obs)
action = e_greedy_policy_fn(key, q_a, epsilon)
next_obs, reward, terminated, truncated, info = env.step(action)
print(td_loss(Params(), Params(), model, obs, next_obs, action, gamma, reward))
# env.close()


# %% Environment #########################################################
def random_play(rng, steps):
    obs, info = env.reset()
    for i in tqdm(range(steps)):
        rng, key = random.split(rng)
        action = random_policy_fn(key)
        next_obs, reward, terminated, truncated, info = env.step(action)
        memory.append(entry(obs, action, reward, next_obs, terminated | truncated))
        obs, info = next_obs, info if not (terminated | truncated) else env.reset()
    env.close()


random_play(rng, 100)


# %% trainig
def train(rng, model, params, steps, gamma, epsilon, t, sample_size):
    obs, info = env.reset()
    target_params = params.deepcopy()
    for i in tqdm(range(steps)):
        rng, key = random.split(rng)

        q_a = model(params, obs)
        action = e_greedy_policy_fn(rng, q_a, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        memory.append(entry(obs, action, reward, next_obs, terminated | truncated))
        if len(memory) < sample_size:


        obs, info = next_obs, info if not (terminated | truncated) else env.reset()
