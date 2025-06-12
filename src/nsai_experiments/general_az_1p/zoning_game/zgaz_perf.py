"A quick script for Zoning Game AlphaZero performance profiling"

# Limit NumPy multithreading because we're doing our own
import os
for lib in ["OMP", "OPENBLAS", "MKL"]:
    os.environ[f"{lib}_NUM_THREADS"] = "1"

import logging

from nsai_experiments.general_az_1p.game import Game
from nsai_experiments.general_az_1p.policy_value_net import PolicyValueNet
from nsai_experiments.general_az_1p.agent import Agent

from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGameGame
from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGamePolicyValueNet

def main():
    mygame = ZoningGameGame()
    mynet = ZoningGamePolicyValueNet(random_seed=47, training_params={"epochs": 10})
    myagent = Agent(mygame, mynet, random_seeds={"mcts": 48, "train": 49, "eval": 50})

    logging.getLogger().setLevel(logging.WARN)
    myagent.play_train_multiple(3)

if __name__ == "__main__":
    main()
