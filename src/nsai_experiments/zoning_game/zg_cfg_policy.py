import numpy as np

from .zg_gym import ZoningGameEnv
from .zg_cfg import interpret_valid_moves, _parse_if_necessary
from .zg_policy import get_legal_moves, create_policy_indiv_greedy, play_one_game

def create_policy_cfg_with_fallback(ruleset, fallback_policy_creator, rng = None, seed=None, legal_moves_decider=get_legal_moves):
    """
    Create a policy that uses `ruleset` to constrain the valid moves and breaks ties between
    multiple valid moves with `fallback_policy_creator`
    """
    # TODO test

    myrand = np.random.default_rng(seed=seed) if rng is None else rng
    ruleset = _parse_if_necessary(ruleset)

    # Compute which moves are both legal and compliant with the ruleset
    def legal_and_valid_moves(obs):
        tile_grid, tile_queue = obs
        legal_moves = legal_moves_decider(obs)
        ruleset_valid_moves = np.flatnonzero(interpret_valid_moves(ruleset, tile_grid, tile_queue[0]))
        result = np.intersect1d(legal_moves, ruleset_valid_moves)
        # If no moves are both legal and compliant, we will have to go with non-compliant
        # moves; return all legal moves even though they don't comply with the ruleset
        if len(result) == 0: return legal_moves
        return result
    
    # If there are multiple legal and ruleset-valid moves, decide between them using the fallback policy
    fallback_policy = fallback_policy_creator(rng=myrand, legal_moves_decider=legal_and_valid_moves)

    def policy_cfg(obs):
        # NOTE could add logging, etc. here
        return fallback_policy(obs)
    
    return policy_cfg

def create_policy_cfg_indiv_greedy(ruleset, rng = None, seed=None, legal_moves_decider=get_legal_moves):
    """
    `create_policy_cfg_with_fallback` with `fallback_policy_creator=create_policy_indiv_greedy`
    """
    return create_policy_cfg_with_fallback(ruleset, create_policy_indiv_greedy, rng=rng, seed=seed, legal_moves_decider=legal_moves_decider)

def evaluate_ruleset(ruleset, fallback_policy_creator=create_policy_indiv_greedy, control_policy_creator=None, policy_seeds = range(0, 10), env_seeds = range(10, 20), on_invalid = None):
    """
    Run a bunch of games with the given `ruleset` and `fallback_policy_creator`, and also
    run those same games with just the `control_policy_creator` (same as
    `fallback_policy_creator` if `None`), and report total scores for each.
    """
    # TODO test
    
    if control_policy_creator is None: control_policy_creator = fallback_policy_creator
    env = ZoningGameEnv()
    ruleset_score = 0
    control_score = 0

    for policy_seed in policy_seeds:
        ruleset_policy = create_policy_cfg_with_fallback(ruleset, fallback_policy_creator, seed=policy_seed)
        control_policy = control_policy_creator(seed=policy_seed)

        for env_seed in env_seeds:
            _, _, ruleset_reward, _, _, _ = play_one_game(ruleset_policy, env=env, seed=env_seed, on_invalid=on_invalid)
            _, _, control_reward, _, _, _ = play_one_game(control_policy, env=env, seed=env_seed, on_invalid=on_invalid)
            ruleset_score += ruleset_reward
            control_score += control_reward
    return ruleset_score.item(), control_score.item()
