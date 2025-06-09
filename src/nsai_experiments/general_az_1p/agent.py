from typing import Any
import warnings
import copy

import numpy as np

from .game import Game
from .policy_value_net import PolicyValueNet
from .mcts import MCTS

class Agent():
    game: Game
    net: PolicyValueNet
    all_training_examples: list[tuple[Any, tuple[Any, float]]] = []  # Accumulated training examples (state, (policy, reward))

    n_games_per_train: int  # Number of games to play for training examples per training call
    n_games_per_eval: int  # Number of games to play in the pitting step per training call
    threshold_to_keep: float  # Threshold for win rate to keep the new network
    reward_discount: float  # For the case where only reward is at the end, set to 1.0 to use end reward at all steps
    mcts_params: dict  # Passed through to MCTS constructor

    def __init__(self, game: Game, net: PolicyValueNet,
                 n_games_per_train: int = 100,
                 n_games_per_eval: int = 10,
                 threshold_to_keep: float = 0.55,
                 reward_discount: float = 1.0,
                 mcts_params: dict | None = None):
        self.game = game
        self.n_games_per_train = n_games_per_train
        self.n_games_per_eval = n_games_per_eval
        self.threshold_to_keep = threshold_to_keep
        self.net = net
        self.reward_discount = reward_discount
        self.mcts_params = mcts_params if mcts_params is not None else {}

    def play_single_game(self, max_moves: int = 10_000):
        train_examples = []
        rewards = []
        mcts = MCTS(self.game, self.net, **self.mcts_params)
        for i in range(max_moves):
            move_probs = mcts.perform_simulations()
            train_examples.append((self.game.obs, (move_probs, None)))
            selected_move = np.random.choice(len(move_probs), p=move_probs)
            # print(f"Taking move {selected_move} with probability {move_probs[selected_move]:.2f}")  # TODO logging
            self.game.step_wrapper(selected_move)
            rewards.append(self.game.reward)
            if self.game.terminated or self.game.truncated:
                break
        else:
            # In this case, we might not have any reward to work with
            warnings.warn(f"`play_single_game` timed out after {max_moves} moves without termination/truncation, returning no training examples")
            return []
        
        # Propagate rewards backwards through steps
        for i in range(len(rewards) - 1, 0, -1):
            rewards[i-1] += self.reward_discount * rewards[i]
        
        # Attach rewards to training examples
        for i in range(len(train_examples)):
            state, (policy, _) = train_examples[i]
            train_examples[i] = (state, (policy, rewards[i]))
        
        return train_examples
    
    def play_and_train(self):
        # Play a bunch of games and keep track of the training examples
        new_train_examples = []  # PERF consider using a deque for efficiency
        for i in range(self.n_games_per_train):
            # print(f"Starting game {i+1} of {self.n_games_per_train}")  # TODO logging
            self.game.reset_wrapper()  # TODO random seed management
            train_examples = self.play_single_game()
            new_train_examples.extend(train_examples)
        
        self.all_training_examples.extend(new_train_examples)
        # TODO currently we never discard old training examples; eventually we probably should

        # Save an old Agent to pit ourselves against, then train the network
        self.game.reset_wrapper()
        self_before_training = copy.deepcopy(self)
        print(f"Training on {len(self.all_training_examples)} examples")
        self.net.train(self.all_training_examples)

        # Play a bunch of games to evaluate new vs. old networks
        old_rewards, new_rewards = [], []
        for i in range(self.n_games_per_eval):
            self.game.reset_wrapper()
            self_before_training.game = copy.deepcopy(self.game)
            
            # NOTE here we are using the final reward to compare performance -- we may in
            # fact want to use a (weighted?) sum of stepwise rewards
            self_before_training.play_single_game()
            reward_from_old = self_before_training.game.reward
            self.play_single_game()
            reward_from_new = self.game.reward

            old_rewards.append(reward_from_old)
            new_rewards.append(reward_from_new)
        
        # Print stats, keep new network iff it wins >= threshold_to_keep fraction of games
        old_rewards = np.array(old_rewards)
        new_rewards = np.array(new_rewards)

        print(f"Old network average reward: {old_rewards.mean()}")
        print(f"New network average reward: {new_rewards.mean()}")
        
        new_wins = np.sum(new_rewards > old_rewards)  # NOTE ties currently go to the old network
        print(f"New network won {new_wins} out of {self.n_games_per_eval} games ({new_wins / self.n_games_per_eval:.2%})")
        if new_wins / self.n_games_per_eval >= self.threshold_to_keep:
            print("Keeping the new network")
        else:
            print("Reverting to the old network")
            self.net = self_before_training.net
    
    def play_train_multiple(self, n_trains: int):
        for i in range(n_trains):
            print(f"Training iteration {i+1} of {n_trains}: will play {self.n_games_per_train} games, train, and evaluate on {self.n_games_per_eval} games")
            self.play_and_train()
