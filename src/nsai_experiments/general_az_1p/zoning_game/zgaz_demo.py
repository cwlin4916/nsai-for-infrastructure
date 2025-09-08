from nsai_experiments.general_az_1p.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import numpy as np

from nsai_experiments.general_az_1p.agent import Agent

from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGameGame
from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGamePolicyValueNet

def main():
    mygame = ZoningGameGame()
    mynet = ZoningGamePolicyValueNet(random_seed=47, training_params={"epochs": 10})  #, device = "cpu")
    myagent = Agent(mygame, mynet, random_seeds={"mcts": 48, "train": 49, "eval": 50},
                    n_past_iterations_to_train=10, n_games_per_train=3000, n_games_per_eval=300, threshold_to_keep = 0.5,
                    # n_procs=-1,
                    mcts_params={"n_simulations": 100, "c_exploration": 0.4})

    myagent.play_train_multiple(20)

if __name__ == "__main__":
    main()
