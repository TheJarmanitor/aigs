# %% lab_6.py
#    monte carlo tree search with JAX
# by: Noah Syrkis

# %% Imports ############################################################
from jax import grad, random, jit, vmap
import jax.numpy as jnp
from jaxtyping import Array
import pgx
from functools import partial

# %% Setup ##############################################################
batch_size = 16
env = pgx.make("tic_tac_toe")
init = jit(vmap(env.init))
step = jit(vmap(env.step))


# %% Define the model ###################################################
@vmap
def rand(key: Array, current_player: Array, observation: Array, legal_action_mask: Array) -> Array:
    action = random.choice(key, jnp.arange(legal_action_mask.size), p=legal_action_mask)
    return action


# %% Initialize the model ###############################################
rng, key = random.split(random.PRNGKey(0))
keys = random.split(key, batch_size)
state = init(keys)


# %% Run the model ######################################################
for model in [rand]:
    state_seq = [state]
    while not (state.terminated | state.truncated).all():
        rng, key = random.split(rng)
        keys = random.split(key, batch_size)

        action = rand(keys, state.current_player,  state.observation, state.legal_action_mask)
        state_seq.append(state := step(state, action))

    # %% Print the results ##################################################
    pgx.save_svg_animation(state_seq, "tic_tac_toe.svg")
