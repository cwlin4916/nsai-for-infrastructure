import logging
from nsai_experiments.zoning_game.zg_gym import ZoningGameEnv, flatten_zg_obs
from multiprocessing import Pool
import numpy as np
import torch
from pathlib import Path

def create_dataset_1_one_game(policy_creator, policy_seed, env_seed):
    env = ZoningGameEnv()
    policy = policy_creator(seed=policy_seed)
    states = []
    moves = []
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
            moves.append(next_move)
            obs, reward, terminated, truncated, info = env.step(next_move, on_invalid=None)
            if terminated or truncated:
                return states, values, moves
    finally:
        logger.setLevel(previous_level)

def _generate_zg_data(policy_creator, n_games = 10_000):
    states = []
    values = []
    moves = []
    with Pool() as pool:
        print(pool)
        results = pool.starmap(create_dataset_1_one_game,
                               [(policy_creator, game_i+n_games, game_i) for game_i in range(n_games)])
    for my_states, my_values, my_moves in results:
        states.extend(my_states)
        values.extend(my_values)
        moves.extend(my_moves)
    
    states = np.array(states)
    values = np.array(values)
    moves = np.array(moves)

    states = torch.tensor(states, dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)
    moves = torch.tensor(moves, dtype=torch.long)
    
    return states, values, moves

def get_zg_data(policy_creator, n_games = 10_000, savedir = "zg_data"):
    savedir = Path(savedir)
    subdir = savedir / f"{policy_creator.__name__}__{n_games}"
    subdir.mkdir(parents=True, exist_ok=True)
    states_path = subdir / "zg_states.pt"
    values_path = subdir / "zg_values.pt"
    moves_path = subdir / "zg_moves.pt"
    if states_path.exists() and values_path.exists() and moves_path.exists():
        print(f"Loading data from disk: {subdir}")
        states_tensor = torch.load(states_path)
        values_tensor = torch.load(values_path)
        moves_tensor = torch.load(moves_path)
    else:
        print(f"Generating data, saving to {subdir}")
        states_tensor, values_tensor, moves_tensor = _generate_zg_data(policy_creator, n_games)
        torch.save(states_tensor, states_path)
        torch.save(values_tensor, values_path)
        torch.save(moves_tensor, moves_path)
    return states_tensor, values_tensor, moves_tensor
