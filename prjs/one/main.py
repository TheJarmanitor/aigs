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


def init_params(rng, env):
    rng, *keys = random.split(rng, 10)
    init_fn = nn.initializers.orthogonal()
    w1 = init_fn(keys[0], (env.observation_space.shape[0], 16))
    b1 = jnp.zeros(16)
    w2 = init_fn(keys[1], (16, 16))
    b2 = jnp.zeros(16)
    w3 = init_fn(keys[2], (16, env.action_space.n))
    b3 = jnp.zeros(env.action_space.n)
    return [w1, b1, w2, b2, w3, b3]


@jit
def model(params, x_data):
    z = x_data @ params[0] + params[1]
    z = nn.relu(z)
    z = z @ params[2] + params[3]
    z = nn.relu(z)
    z = z @ params[4] + params[5]
    # z = nn.softmax(z)

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


def td_loss(params, target_params, model, gamma, obs, action, reward, next_obs, done):
    next_action = jnp.argmax(model(params, next_obs), axis=1)
    q_next = jnp.take_along_axis(
        model(target_params, next_obs), next_action[:, None], axis=1
    ).squeeze()
    target = reward + gamma * q_next * (1.0 - done)
    q_pred = model(params, obs)
    action = action[:, None]
    q_pred_selected = jnp.take_along_axis(q_pred, action, axis=1).squeeze()
    loss = jnp.mean(jnp.square(target - q_pred_selected))
    return loss


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


def soft_update(params, target_params, tau):
    return tree.map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)


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
    for i in range(1, max_episodes):
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
                print("Warmup has ended")
                steps = 0
                warmup_phase = False
            if steps % t == 0 and not warmup_phase:
                current_reward_avg = jnp.mean(jnp.array(avg_cum_reward))
                if current_reward_avg >= best_reward:
                    print("New best policy found! Saving")
                    with open("weights.pkl", "wb") as file:
                        pickle.dump(params, file)
                    best_reward = current_reward_avg
                target_params = soft_update(params, target_params, tau)
                print(
                    f"Episode: {i}, Step: {steps}, average cumulative_reward: {current_reward_avg} average loss: {jnp.mean(jnp.array(avg_loss))}"
                )
        if len(memory) >= sample_size and not warmup_phase:
            batch = sample(memory, sample_size)

            obs_batch = jnp.array([x.obs for x in batch])
            action_batch = jnp.array([x.action for x in batch])
            reward_batch = jnp.array([x.reward for x in batch])
            next_obs_batch = jnp.array([x.next_obs for x in batch])
            done_batch = jnp.array([x.done for x in batch])

            loss, grads = value_and_grad(td_loss)(
                params,
                target_params,
                model,
                gamma,
                obs_batch,
                action_batch,
                reward_batch,
                next_obs_batch,
                done_batch,
            )
            # print(loss)
            avg_loss.append(loss)
            params = tree.map(lambda p, g: p - learning_rate * g, params, grads)
            # params = [params[i] - learning_rate * grads[i] for i in range(len(params))]
        if not warmup_phase:
            avg_cum_reward.append(cum_reward)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        done = False
        obs, info = env.reset()

    env.close()


# %% Constants ###########################################################
rng = random.PRNGKey(1331)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=10000)  # <- replay buffer
max_episodes = 2000
gamma = 0.95
epsilon = 0.5
t = 1000
epsilon_decay = 0.995
sample_size = 128
learning_rate = 0.1
# define more as needed
env = gym.make("CartPole-v1", render_mode="human")  #  render_mode="human")

params = init_params(rng, env)


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
