"A quick demo showing learning on Cartpole."

from nsai_experiments.general_az_1p.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

from nsai_experiments.general_az_1p.agent import Agent

from nsai_experiments.general_az_1p.cartpole.cartpole_az_impl import CartPoleGame
from nsai_experiments.general_az_1p.cartpole.cartpole_az_impl import CartPolePolicyValueNet

def main():
    mygame = CartPoleGame(max_steps=100)
    mynet = CartPolePolicyValueNet(random_seed=47,
                                   training_params={"epochs": 10, "learning_rate": 0.03, "policy_weight": 2.0},
                                   device = "cpu")
    myagent = Agent(mygame, mynet,
                    random_seeds={"mcts": 48, "train": 49, "eval": 50},
                    n_games_per_train = 100,
                    n_games_per_eval = 20,
                    n_past_iterations_to_train=2,
                    mcts_params={"c_exploration": 0.2})

    myagent.play_train_multiple(3)

if __name__ == "__main__":
    main()
