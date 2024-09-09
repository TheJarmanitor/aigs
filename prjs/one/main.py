# %% Imports (add as needed) #############################################
import gymnasium as gym  # not jax based
from jax import random
from tqdm import tqdm
from collections import deque, namedtuple

# %% Constants ###########################################################
env = gym.make("MountainCar-v0")  #  render_mode="human")
rng = random.PRNGKey(0)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=1000)  # <- replay buffer
# define more as needed

# %% Model ###############################################################
def random_policy_fn(rng, obs): # action (shape: ())
    n = env.action_space.__dict__['n']
    return random.randint(rng, (1,), 0, n).item()

def your_policy_fn(rng, obs):  # obs (shape: (2,)) to action (shape: ())
    raise NotImplementedError

# %% Environment #########################################################
obs, info = env.reset()
for i in tqdm(range(100)):

    rng, key = random.split(rng)
    action = random_policy_fn(key, obs)

    next_obs, reward, terminated, truncated, info = env.step(action)
    memory.append(entry(obs, action, reward, next_obs, terminated | truncated))
    obs, info = next_obs, info if not (terminated | truncated) else env.reset()

env.close()
