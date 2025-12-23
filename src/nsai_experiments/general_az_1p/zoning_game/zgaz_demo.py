# python -m nsai_experiments.general_az_1p.zoning_game.zgaz_demo > zgaz.log
# tail -fn+1 zgaz.log

from nsai_experiments.general_az_1p.setup_utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

from nsai_experiments.general_az_1p.agent import Agent

from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGameGame
from nsai_experiments.general_az_1p.zoning_game.zoning_game_az_impl import ZoningGamePolicyValueNet

from nsai_experiments.zoning_game.zg_policy import \
    create_policy_random, create_policy_indiv_greedy, create_policy_total_greedy

def main():
    mygame = ZoningGameGame()
    mynet = ZoningGamePolicyValueNet(random_seed=47, training_params={"epochs": 10, "learning_rate": 0.0003})
    myagent = Agent(mygame, mynet, random_seeds={"mcts": 48, "train": 49, "eval": 50, "external_policy": 51},
                    n_past_iterations_to_train=10, n_games_per_train=3000, n_games_per_eval=300, threshold_to_keep = 0.5,
                    mcts_params={"n_simulations": 100, "c_exploration": 0.5},
                    external_policy_creators_to_pit={"random": create_policy_random, "individual greedy": create_policy_indiv_greedy, "total greedy": create_policy_total_greedy})
    

    mynet.load_checkpoint("zgaz_checkpoints/1757654225_iter_200")
    myagent.load_checkpoint("zgaz_checkpoints/1757654225_iter_200")

    myagent.play_train_multiple(1000, checkpoint_every=20, checkpoint_dir="/scratch/gkonars/NSAI/zgaz/zgaz_checkpoints", start_at=200)

if __name__ == "__main__":
    main()
