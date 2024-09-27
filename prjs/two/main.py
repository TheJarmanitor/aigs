# %% libraries
import jax
import jumanji
from jumanji.wrappers import AutoResetWrapper



# %% Create environment
rng = jax.random.PRNGKey(0)

env = jumanji.make("Sokoban-v0")
env = AutoResetWrapper(env)

# %% data gathering

def step_fn(key, env, state):
    num_actions = env.action_spec.num_values
    action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=())
    new_state, timestep = env.step(state, action)
    return new_state, timestep



def data_gathering(key, step_fn, env, n_steps=1000):
    state = env.reset(key)
    random_keys = jax.random.split(key, n_steps)
    state, rollout = jax.lax.scan(step_fn, )
    return rollout

data_gathering(rng, step_fn, env)
