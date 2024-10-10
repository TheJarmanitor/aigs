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
import pickle


# %% Model ###############################################################

def init_params(rng, layers):
    params = []
    init_fn = nn.initializers.orthogonal()
    rng, *keys = random.split(rng, len(layers))
    for i, o in zip(layers[:-1], layers[1:]):
        w = init_fn(keys[0], (i, o))
        b = jnp.zeros(o)
        params.append((w,b))
    return params



@jit
def model(params, x_data):
    for w, b in params:
        z = x_data @ w + b
        x_data = nn.relu(z)

    # return jnp.argmax(z)
    return z


def random_policy_fn(rng):  # action (shape: ())
    rng, key = random.split(rng)
    n = env.action_space.__dict__["n"]
    return random.randint(key, (1,), 0, n).item(), rng


def e_greedy_policy_fn(rng, q_a, epsilon):  # obs (shape: (2,)) to action (shape: ())
    rng, key = random.split(rng)
    action = jnp.argmax(q_a)
    if random.uniform(key, ()) < epsilon:
        action, rng = random_policy_fn(rng)
    return int(action), rng

@value_and_grad
def td_loss(params, target_params, model, gamma, obs, action, reward, next_obs, done):
    next_action = jnp.argmax(model(params, next_obs), axis=1)
    q_next = jnp.take_along_axis(
        model(target_params, next_obs), next_action[:, None], axis=1
    ).squeeze()
    target = reward + gamma * q_next * (1.0 - done)
    q_pred = model(params, obs)
    q_pred_selected = jnp.take_along_axis(q_pred, action[:, None], axis=1).squeeze()
    loss = jnp.mean(jnp.square(target - q_pred_selected))
    return loss

def update_params(params, target_params, model, gamma, lr, env_info):
    loss, grads = td_loss(params, target_params, model, gamma, *env_info)
    params = tree.map(lambda p, g: p - lr * g, params, grads)

    return params, loss


def soft_update(params, target_params, tau):
    return tree.map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)

def sample_batch(key, memory, batch_size):
    idxs = random.choice(key, len(memory), (batch_size,), replace=False)

    batch = map(jnp.array, zip(*(memory[i] for i in idxs)))

    return tuple(batch)

# %% Environment #########################################################
def random_play(rng, max_episodes):
    obs, info = env.reset()
    avg_cum_reward = []
    steps = 0
    done = False
    for i in range(max_episodes):
        cum_reward = 0.0
        while not done:
            rng, key = random.split(rng)
            action, rng = random_policy_fn(key)
            next_obs, reward, terminated, truncated, info = env.step(action)
            cum_reward += reward
            done = terminated | truncated
            memory.append(entry(obs, action, reward, next_obs, done))
            obs, info = next_obs, info
        avg_cum_reward.append(cum_reward)
        if max_episodes % 10 == 0:
            print("Cumulative reward: ", jnp.mean(jnp.array(avg_cum_reward)))
        done = False
        obs, info = env.reset()
    env.close()




# %% trainig
def train(
    rng,
    model,
    params,
    max_episodes,
    gamma,
    epsilon,
    t,
    sample_size,
    learning_rate,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    tau=1,
    warmup_steps=1000,
):
    obs, info = env.reset()
    avg_cum_reward = []
    avg_loss = []
    target_params = deepcopy(params)
    steps = 0
    done = False
    warmup_phase = True
    best_reward = 0
    # for i in range(1, max_episodes):
    for i in (pbar := tqdm(range(max_episodes))):
        cum_reward = 0.0
        while not done:
            steps += 1
            if warmup_phase:
                action, rng = random_policy_fn(rng)
            else:
                q_a = model(params, obs)
                action, rng = e_greedy_policy_fn(rng, q_a, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            cum_reward += reward
            done = terminated | truncated
            memory.append(entry(obs, action, reward, next_obs, done))
            obs, info = next_obs, info
            if warmup_phase and steps >= warmup_steps:
                steps = 0
                warmup_phase = False
            if steps % t == 0 and not warmup_phase:
                current_reward_avg = jnp.mean(jnp.array(avg_cum_reward))
                if current_reward_avg >= best_reward:
                    with open("weights.pkl", "wb") as file:
                        pickle.dump(params, file)
                    best_reward = current_reward_avg
                target_params = soft_update(params, target_params, tau)
                # print(
                #     f"Episode: {i}, Step: {steps}, average cumulative_reward: {current_reward_avg} average loss: {jnp.mean(jnp.array(avg_loss))}"
                # )
        if len(memory) >= sample_size and not warmup_phase:
            batch = sample_batch(rng, memory, sample_size)
            # print(loss)
            params, loss = update_params(params, target_params, model, gamma, learning_rate, batch)
            avg_loss.append(loss)
        if not warmup_phase:
            avg_cum_reward.append(cum_reward)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        done = False
        obs, info = env.reset()
        pbar.set_description(f"epsilon: {epsilon}, average cumulative_reward: {jnp.mean(jnp.array(avg_cum_reward))} average loss: {jnp.mean(jnp.array(avg_loss))}")

    env.close()


# %% Constants ###########################################################
rng = random.PRNGKey(1331)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=10000)  # <- replay buffer
max_episodes = 2000
gamma = 0.95
epsilon = 1
t = 1000
epsilon_decay = 0.995
sample_size = 128
learning_rate = 5e-4
# define more as needed
env = gym.make("CartPole-v1", render_mode="human")  #  render_mode="human")
layers = [env.observation_space.shape[0], 64, 64, env.action_space.n]

params = init_params(rng, layers)


train(
    rng,
    model,
    params,
    max_episodes,
    gamma,
    epsilon,
    t,
    sample_size,
    learning_rate,
    epsilon_decay,
)
# random_play(rng, 100)


# %% Environment #########################################################
def play(rng, model, max_episodes, weights_file):
    with open(weights_file, "rb") as file:
        params = pickle.load(file)
    obs, info = env.reset()
    done = False
    for i in range(max_episodes):
        while not done:
            rng, key = random.split(rng)
            q_a = model(params, obs)
            action, rng = e_greedy_policy_fn(key, q_a, epsilon=0.01)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            memory.append(entry(obs, action, reward, next_obs, done))
            obs, info = next_obs, info
        done = False
        obs, info = env.reset()
    env.close()

play(rng, model, 100, weights_file="weights.pkl")
=======
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
>>>>>>> upstream/main
