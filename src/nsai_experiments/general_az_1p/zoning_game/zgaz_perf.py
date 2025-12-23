"A quick script for Zoning Game AlphaZero performance profiling"

from nsai_experiments.general_az_1p.setup_utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import logging

from nsai_experiments.general_az_1p.game import Game
from nsai_experiments.general_az_1p.policy_value_net import PolicyValueNet
from nsai_experiments.general_az_1p.agent import Agent

from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGameGame
from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGamePolicyValueNet

def main():
    mygame = ZoningGameGame()
    mynet = ZoningGamePolicyValueNet(random_seed=47, training_params={"epochs": 10})
    myagent = Agent(mygame, mynet, n_procs=10, random_seeds={"mcts": 48, "train": 49, "eval": 50})

    logging.getLogger().setLevel(logging.WARN)
    myagent.play_train_multiple(100)

if __name__ == "__main__":
    main()
