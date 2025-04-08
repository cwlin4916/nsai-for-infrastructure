import logging
from nsai_experiments.zoning_game.zg_gym import ZoningGameEnv, flatten_zg_obs

def create_dataset_1_one_game(policy_creator, policy_seed, env_seed):
    env = ZoningGameEnv()
    policy = policy_creator(seed=policy_seed)
    states = []
    values = []
    obs, info = env.reset(seed=env_seed)
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        while True:
            next_move = policy(obs)
            states.append(flatten_zg_obs(obs))
            values.append(env._eval_tile_grid_score())
            obs, reward, terminated, truncated, info = env.step(next_move, on_invalid=None)
            if terminated or truncated:
                return states, values
    finally:
        logger.setLevel(previous_level)
