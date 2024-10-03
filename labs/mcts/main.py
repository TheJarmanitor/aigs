# %% main.py
#     Lab 6: MiniMax, Alpha-Beta Pruning, Monte Carlo Tree Search (MCTS), and some thoughts on the limitations of JAX
# by: Noah Syrkis

# %% Imports
import pgx
from jax import jit, random, vmap, lax
import jax.numpy as jnp
from functools import partial, lru_cache

# %% Setup
rng = random.PRNGKey(0)
env = pgx.make("tic_tac_toe")
init = jit(env.init)
step = jit(env.step)


# %% Functions
def animate_state_seq(state_seq, filename):  # visualize a sequence of states
    pgx.save_svg_animation(state_seq, filename, frame_duration_seconds=0.5)


def step_fn(search_fn, state, maxim):  # recursive function for tree search (you will write search_fn)
    actions = jnp.where(state.legal_action_mask)[0]  # valid actions
    values = jnp.array([search_fn(step(state, action), not maxim) for action in actions])  # computer (or estimate)
    action = actions[jnp.argmax(values) if maxim else jnp.argmin(values)]  # select best action
    return action


# %% 1. Minimax
# Implement the minimax algorithm for tic-tac-toe.
# JAX is not well-suited for tree search algorithms, but we can still implement them.
# However, do not be "smart" about it. Use a naive if-else structure.
# Important: once you have done this, think/talk about what you have used the state observations for.
def minimax(state, maxim):
    if state.terminated or state.truncated:
        return state.rewards[0]  # return the reward for player 0
    else:
        raise NotImplementedError("You need to implement the rest of the minimax algorithm.")
        return value


# %% 2. Alpha-Beta Pruning
# Modify the minimax algorithm to use alpha-beta pruning.


# %% 3. Monte Carlo Tree Search (MCTS) (optional and advanced)
# Try to implement MCTS for any game you like in pgx.


# %% 4. mctx (optional and advanced)
# Explore https://github.com/google-deepmind/mctx
# Skim through this notebook, and run it if you want:
# https://github.com/Carbon225/mctx-classic/blob/master/connect4.ipynb


####################################################################################################
# %% Test code that Noah used while playin around with the code
# state = init(rng)
# state_seq = [state]
# while not (state.terminated or state.truncated):
#     action = step_fn(minimax, state, state.current_player == 0)
#     state = step(state, action)
#     state_seq.append(state)
# pgx.save_svg_animation(state_seq, "minimax.svg", frame_duration_seconds=1)
