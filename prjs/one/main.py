# %% Imports (add as needed) #############################################
import gymnasium as gym  # not jax based
from jax import random, nn
import jax.numpy as jnp
from numpy.random.mtrand import gamma
from tqdm import tqdm
from collections import deque, namedtuple
import chex

# %% Constants ###########################################################
env = gym.make("MountainCar-v0", render_mode="human")  #  render_mode="human")
rng = random.PRNGKey(1331)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=1000)  # <- replay buffer
gamma = 0.99
epsilon = 0.2
# define more as needed
# %%
print(env.observation_space.__dict__)


# %% Model ###############################################################
rng, *keys = random.split(rng,10)
@chex.dataclass
class Params:
    w1 = random.normal(keys[0], (env.observation_space._shape[0], 16))
    b1 = random.normal(keys[1], 16)
    w2 = random.normal(keys[2], (16, 16))
    b2 = random.normal(keys[3], 16)
    w3 = random.normal(keys[4], (16, env.action_space.n))
    b3 = random.normal(keys[5], int(env.action_space.n))


def model(params: Params, x_data):
    z = x_data @ params.w1 + params.b1
    z = nn.relu(z)
    z = z @ params.w2 + params.b2
    z = nn.relu(z)
    z = z @ params.w3 + params.b3
    z = nn.softmax(z)

    return jnp.argmax(z, axis=1)

def td_loss()

# def random_policy_fn(rng, obs): # action (shape: ())
#     n = env.action_space.__dict__['n']
#     return random.randint(rng, (1,), 0, n).item()



# def your_policy_fn(rng, obs):  # obs (shape: (2,)) to action (shape: ())
#     raise NotImplementedError

# def update():
#     raise NotImplementedError

# %% Environment #########################################################
obs, info = env.reset()
for i in tqdm(range(1000)):
    rng, key = random.split(rng)
    action = random_policy_fn(key, obs)

    next_obs, reward, terminated, truncated, info = env.step(action)
    memory.append(entry(obs, action, reward, next_obs, terminated | truncated))
    obs, info = next_obs, info if not (terminated | truncated) else env.reset()

env.close()
