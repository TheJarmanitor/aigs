# %% Imports (add as needed) #############################################
from functools import partial
import gymnasium as gym  # not jax based
from jax import random, nn, jit, vmap, value_and_grad, tree
from jax._src.api import jit
import jax.numpy as jnp
from numpy.random.mtrand import gamma
from tqdm import tqdm
from collections import deque, namedtuple
import chex
from copy import deepcopy
from random import sample

# %% Constants ###########################################################
env = gym.make("CartPole-v1", render_mode="human")  #  render_mode="human")
rng = random.PRNGKey(1331)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=1000)  # <- replay buffer
gamma = 0.95
epsilon = 0.3
epsilon_decay = 0.01
sample_size = 32
learning_rate = 0.005
# define more as needed



# %% Model ###############################################################



# @chex.dataclass
# class Params:
    # w1 = random.normal(keys[0], (env.observation_space._shape[0], 16))
    # b1 = random.normal(keys[1], 16)
    # w2 = random.normal(keys[2], (16, 16))
    # b2 = random.normal(keys[3], 16)
    # w3 = random.normal(keys[4], (16, env.action_space.n))
    # b3 = random.normal(keys[5], int(env.action_space.n))
#     def __iter__(self):
#             yield self.w1
#             yield self.b1
#             yield self.w2
#             yield self.b2
#             yield self.w3
#             yield self.b3
def init_params(rng, env):
    rng, *keys = random.split(rng, 10)
    w1 = random.normal(keys[0], (env.observation_space._shape[0], 16))
    b1 = random.normal(keys[1], 16)
    w2 = random.normal(keys[2], (16, 16))
    b2 = random.normal(keys[3], 16)
    w3 = random.normal(keys[4], (16, env.action_space.n))
    b3 = random.normal(keys[5], int(env.action_space.n))
    return [w1, b1, w2, b2, w3, b3]

def model(params, x_data):
    z = x_data @ params[0] + params[1]
    z = nn.relu(z)
    z = z @ params[2] + params[3]
    z = nn.relu(z)
    z = z @ params[4] + params[5]
    z = nn.softmax(z)

    # return jnp.argmax(z)
    return z


def random_policy_fn(rng): # action (shape: ())
    n = env.action_space.__dict__['n']
    return random.randint(rng, (1,), 0, n).item()

def e_greedy_policy_fn(rng, q_a, epsilon, decay=0.99):  # obs (shape: (2,)) to action (shape: ())
    rng, key = random.split(rng)
    random_action = random_policy_fn(rng)
    action = jnp.argmax(q_a)
    final_action = random.choice(key, jnp.array([action, random_action]), p=jnp.array([1-epsilon, epsilon]))
    return int(final_action)

def batch_td_loss(params, target_params, model, gamma, obs, action, reward, next_obs, done):

    @partial(vmap, in_axes=(None, None, None, None, 0, 0, 0, 0, 0))
    def td_loss(params, target_params, model, gamma, obs, action, reward, next_obs, done):
        target = gamma * jnp.max(model(target_params, next_obs))
        prediction = model(params, obs)[action]
        # return target
        return jnp.square(reward + (1 - done) * target - prediction)

    return jnp.mean(td_loss(params, target_params, model, gamma, obs, action, reward, next_obs, done))

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


# random_play(rng, 100)

# %% trainig
def train(rng, model, params, steps, gamma, epsilon, t, sample_size, learning_rate):
    obs, info = env.reset()
    avg_reward = []
    target_params = params
    done = False
    for i in range(1,steps+1):
        while not done:
            rng, key = random.split(rng)

            q_a = model(params, obs)
            action = e_greedy_policy_fn(rng, q_a, epsilon, )
            next_obs, reward, terminated, truncated, info = env.step(action)
            avg_reward.append(reward)
            done = terminated | truncated
            memory.append(entry(obs, action, reward, next_obs, done))


        if len(memory) < sample_size:
            batch = sample(memory, len(memory))
        else:
            batch = sample(memory, sample_size)

        obs_batch = jnp.array([x.obs for x in batch])
        action_batch = jnp.array([x.action for x in batch])
        reward_batch = jnp.array([x.reward for x in batch])
        next_obs_batch = jnp.array([x.next_obs for x in batch])
        done_batch = jnp.array([x.done for x in batch])


        loss, grads = value_and_grad(batch_td_loss)(params, target_params, model, gamma, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)
        # params = tree.map(lambda p, g: p -  g * learning_rate, params, grads)
        params = [params[i] - learning_rate * grads[i] for i in range(len(params))]
        # print(params[0])
        if i % t == 0:
            target_params = params
            # print(f"Step: {i}, Average reward: {jnp.mean(jnp.array(avg_reward))}")
            print(params[0])

        obs, info = env.reset()

    env.close()

env = gym.make("CartPole-v1", render_mode="human")  #  render_mode="human")

params = init_params(rng, env)

print(params[0].shape)

train(rng, model, params, 10000, gamma, epsilon,100, 32, learning_rate)
