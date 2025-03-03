import logging

import numpy as np

from .zg_gym import ZoningGameEnv, Tile, eval_tile_indiv_score, blank_of_size, pad_grid_inplace

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def play_one_game(policy, env = None, seed = None, on_invalid = None):
    if env is None:
        env = ZoningGameEnv()
    obs, info = env.reset(seed = seed)
    
    while True:
        next_move = policy(obs)
        obs, reward, terminated, truncated, info = env.step(next_move, on_invalid=on_invalid)
        if terminated or truncated:
            return env, obs, reward, terminated, truncated, info

def get_legal_moves(obs):
    tile_grid, _ = obs
    return np.flatnonzero(tile_grid == Tile.EMPTY.value)

def create_policy_random(rng = None, seed=None, legal_moves_decider=get_legal_moves):
    "Creates a random policy function given an RNG seed `seed`."
    myrand = np.random.default_rng(seed=seed) if rng is None else rng
    def policy_random(obs):
        return myrand.choice(legal_moves_decider(obs))
    return policy_random

def _evaluate_move(evaluator, tile_grid, next_tile, move):
    tile_grid = tile_grid.copy()
    row, col = move // tile_grid.shape[1], move % tile_grid.shape[1]
    tile_grid[row, col] = next_tile
    return evaluator(tile_grid, row, col)

def _create_policy_greedy(evaluator, rng = None, seed=None, legal_moves_decider=get_legal_moves):
    """
    Given an `evaluator` function that takes a `tile_grid` and a `row` and `col` and returns
    a value, creates a policy that plays the move that always maximizes that value, breaking
    ties randomly.
    """
    myrand = np.random.default_rng(seed=seed) if rng is None else rng
    def policy_greedy(obs):
        tile_grid, tile_queue = obs
        next_tile = tile_queue[0]
        move_options = legal_moves_decider(obs)
        move_scores = [_evaluate_move(evaluator, tile_grid, next_tile, move) for move in move_options]
        max_score = max(move_scores)
        move_weights = [(1+myrand.random() if score == max_score else 0) for score in move_scores]
        return move_options[np.argmax(move_weights)]
    return policy_greedy

def _create_indiv_greedy_evaluator():
    padded_grid = blank_of_size(0)
    def _indiv_greedy_evaluator(tile_grid, row, col):
        pad_grid_inplace(padded_grid, tile_grid)
        return eval_tile_indiv_score(padded_grid, row, col)
    return _indiv_greedy_evaluator

def create_policy_indiv_greedy(rng = None, seed=None, legal_moves_decider=get_legal_moves):
    """
    Creates a policy that puts the next tile wherever would maximize its individual score at
    the current point in time, breaking ties randomly.
    """
    return _create_policy_greedy(evaluator=_create_indiv_greedy_evaluator(), rng=rng, seed=seed, legal_moves_decider=legal_moves_decider)

def _create_total_greedy_evaluator():
    padded_grid = blank_of_size(0)
    def _total_greedy_evaluator(tile_grid, row, col):
        pad_grid_inplace(padded_grid, tile_grid)
        total_score = 0
        for row, col in np.ndindex(tile_grid.shape):
            total_score += eval_tile_indiv_score(padded_grid, row, col)
        return total_score
    return _total_greedy_evaluator

def create_policy_total_greedy(rng = None, seed=None, legal_moves_decider=get_legal_moves):
    """
    Creates a policy that puts the next tile wherever would maximize the total grid score at
    the current point in time, breaking ties randomly.
    """
    return _create_policy_greedy(evaluator=_create_total_greedy_evaluator(), rng=rng, seed=seed, legal_moves_decider=legal_moves_decider)
