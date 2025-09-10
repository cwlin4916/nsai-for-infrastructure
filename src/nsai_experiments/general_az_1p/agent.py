from typing import Any
import warnings
import copy
import time
from multiprocessing import Pool
import logging
import itertools
import os

import numpy as np

from .game import Game
from .policy_value_net import PolicyValueNet
from .mcts import MCTS, entab
from .utils import THREAD_VARS

# Debugging flags:
DETAILED_DEBUG = False
PIT_NO_MCTS = True
PRINT_ALL_EPOCHS = True

class Agent():
    # Constants
    RNG_NAMES = ["mcts", "train", "eval"]  # Names for the random number generators to construct

    # State
    game: Game
    net: PolicyValueNet
    all_training_examples: list[list[tuple[Any, tuple[Any, float]]]] = []  # Accumulated training examples (state, (policy, reward))
    rngs: dict[str, np.random.Generator]

    # Config
    n_games_per_train: int  # Number of games to play for training examples per training call
    n_games_per_eval: int  # Number of games to play in the pitting step per training call
    n_past_iterations_to_train: int | None  # Number of past iterations to use for training, where an iteration contains `n_games_per_train` games; if None, use all past iterations
    threshold_to_keep: float  # Threshold for win rate to keep the new network
    reward_discount: float  # For the case where only reward is at the end, set to 1.0 to use end reward at all steps
    mcts_params: dict  # Passed through to MCTS constructor
    n_procs: int | None  # Number of processes to use in the `multiprocessing.Pool()`; if None, use all available cores, if < 0 do not use multiprocessing

    def __init__(self, game: Game, net: PolicyValueNet,
                 n_games_per_train: int = 100,
                 n_games_per_eval: int = 20,
                 n_past_iterations_to_train: int = 20,
                 threshold_to_keep: float = 0.55,
                 reward_discount: float = 1.0,
                 mcts_params: dict | None = None,
                 n_procs: int | None = None,
                 random_seeds: dict[str, int] | None = None):
        self.game = game
        self.n_games_per_train = n_games_per_train
        self.n_games_per_eval = n_games_per_eval
        self.n_past_iterations_to_train = n_past_iterations_to_train
        self.threshold_to_keep = threshold_to_keep
        self.net = net
        self.reward_discount = reward_discount
        self.mcts_params = mcts_params if mcts_params is not None else {}
        self.n_procs = n_procs
        if self.n_procs is None or self.n_procs >= 0:
            if not all([os.environ.get(thread_var, None) == "1" for thread_var in THREAD_VARS]):
                warnings.warn(f"You have elected to use multiprocessing, but NumPy multithreading is not disabled. This may lead to thread oversubscription. You can disable NumPy multithreading by setting the environment variables {','.join(THREAD_VARS)} to 1 before importing NumPy, or disable multiprocessing by passing n_procs=-1 to the Agent constructor.")
        self._construct_rngs(random_seeds if random_seeds is not None else {})

    def _construct_rngs(self, random_seeds: dict[str, int]):
        self.rngs = {}
        for rng_name in self.RNG_NAMES:
            seed = random_seeds.get(rng_name, None)
            self.rngs[rng_name] = np.random.default_rng(seed)
        if all(rng_name in random_seeds for rng_name in self.RNG_NAMES):
            print(f"RNG seeds are fully specified")
        else:
            print(f"RNG seeds are not fully specified, using nondeterministic seeds for: {', '.join(rng_name for rng_name in self.RNG_NAMES if rng_name not in random_seeds)}")

    def play_single_game(self, max_moves: int = 10_000, random_seed: int | None = None, msg = ""):
        train_examples = []
        rewards = []
        mcts = MCTS(self.game, self.net, **self.mcts_params)
        rng = np.random.default_rng(random_seed)
        for i in range(max_moves):
            if msg: print(msg, f"starting move {i}")
            move_probs = mcts.perform_simulations(entab(msg, f", move {i+1}"))
            self.game = mcts.game  # TODO HACK because MCTS modifies the game state in place
            train_examples.append((self.game.obs, (move_probs, None)))
            selected_move = rng.choice(len(move_probs), p=move_probs)
            if msg: print(msg, "obs", self.game.obs, "hobs", self.game.hashable_obs, "move_probs", move_probs, "selmove", selected_move)
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
    
    def _play_for_examples(self, i, reset_seed, mcts_seed):
        # print(i)
        logging.getLogger().setLevel(logging.WARN)
        self.game.reset_wrapper(seed=reset_seed)
        return self.play_single_game(random_seed=mcts_seed)

    def _play_for_eval(self, i, reset_seed, mcts_seed, self_before_training, try_without_mcts = False):
        if DETAILED_DEBUG: print(i)
        logging.getLogger().setLevel(logging.WARN)
        self.game.reset_wrapper(seed=reset_seed)
        self_before_training.game = copy.deepcopy(self.game)
        if DETAILED_DEBUG: print("obs", self.game.obs, self_before_training.game.obs)
        if try_without_mcts:
            self_before_training_no_mcts = copy.deepcopy(self_before_training)
            self_before_training_no_mcts.mcts_params = {"n_simulations": -1}
            self_no_mcts = copy.deepcopy(self)
            self_no_mcts.mcts_params = {"n_simulations": -1}
        else:
            self_before_training_no_mcts = None
            self_no_mcts = None

        # NOTE here we are using the final reward to compare performance -- we may in
        # fact want to use a (weighted?) sum of stepwise rewards

        results = []
        for (agent, label) in [(self_before_training, "old net"), (self, "new net"), (self_before_training_no_mcts, "old net no MCTS"), (self_no_mcts, "new net no MCTS")]:
            if agent is None:
                results.append(None)
                continue
            agent.play_single_game(random_seed=mcts_seed, msg = f"{label} game {i}" if DETAILED_DEBUG else "")
            results.append(agent.game.reward)

        if DETAILED_DEBUG: print(f"Reward from old: {results[0]:.2f}, Reward from new: {results[1]:.2f}")
        if DETAILED_DEBUG and try_without_mcts:
            print(f"Reward from old no MCTS: {results[2]:.2f}, Reward from new no MCTS: {results[3]:.2f}")
        return results

    def _starmap(self, fn, arg_tuples):
        """
        Call `fn(*args)` for each `args in arg_tuples`. If `self.n_procs` is None or >= 0,
        construct a `multiprocessing.Pool` and use `Pool.starmap`; otherwise use
        `itertools.starmap`.
        """
        if self.n_procs is None or self.n_procs >= 0:
            with Pool(processes=self.n_procs) as pool:
                results = pool.starmap(fn, arg_tuples)
        else:
            results = list(itertools.starmap(fn, arg_tuples))
        return results
    
    def _randseed(self, rng_name: str):
        "Extract a random integer from RNG `rng_name` suitable for seeding another RNG"
        return int(self.rngs[rng_name].integers(2**31-1))

    def play_and_train(self):
        # Play a bunch of games and keep track of the training examples
        new_train_examples = []  # PERF consider using a deque for efficiency

        start_time = time.time()
        multiprocessing_stash = self.net.push_multiprocessing()
        arg_tuples = [(i, self._randseed("train"), self._randseed("mcts")) for i in range(self.n_games_per_train)]
        train_example_sets = self._starmap(self._play_for_examples, arg_tuples)
        self.net.pop_multiprocessing(multiprocessing_stash)
        for train_examples in train_example_sets:
            new_train_examples.extend(train_examples)
        elapsed = time.time() - start_time
        print(f"..games done in {elapsed:.2f} seconds")
        
        self.all_training_examples.append(new_train_examples)
        if self.n_past_iterations_to_train is not None and len(self.all_training_examples) > self.n_past_iterations_to_train:
            self.all_training_examples.pop(0)
        print(f"Training examples lengths: {[len(x) for x in self.all_training_examples]}")
        flat_examples = list(itertools.chain.from_iterable(self.all_training_examples))

        # Save an old Agent to pit ourselves against, then train the network
        self.game.reset_wrapper()
        self_before_training = copy.deepcopy(self)

        # Sanity check: game states and network predictions on current state should be the
        # same before we train, assuming the network is deterministic. PyTorch inherent
        # nondeterminism seems to be large enough that we sometimes need the isclose
        # assert all(self.game.obs == self_before_training.game.obs)
        assert self.game.hashable_obs == self_before_training.game.hashable_obs
        p1, v1 = self.net.predict(self.game.obs)
        p2, v2 = self_before_training.net.predict(self_before_training.game.obs)
        assert all(np.isclose(p1, p2))
        assert np.isclose(v1, v2)

        print(f"Training on {len(flat_examples)} examples")
        start_time = time.time()
        self.net.train(flat_examples, **({"print_all_epochs": True} if PRINT_ALL_EPOCHS else {}))
        elapsed = time.time() - start_time
        print(f"..training done in {elapsed:.2f} seconds")

        score = self.pit(self_before_training)
        if score >= self.threshold_to_keep:
            print("Keeping the new network")
        else:
            print("Reverting to the old network")
            self.net = self_before_training.net

    def pit(self, self_before_training):
        "Play a bunch of games to evaluate new vs. old networks"
        
        old_rewards, new_rewards, old_rewards_no_mcts, new_rewards_no_mcts = [], [], [], []
        start_time = time.time()
        my_multiprocessing_stash = self.net.push_multiprocessing()
        before_multiprocessing_stash = self_before_training.net.push_multiprocessing()
        # print("pred on old", self_before_training.game.obs, self_before_training.net.predict(self_before_training.game.obs))
        # print("pred on new", self.game.obs, self.net.predict(self.game.obs))
        arg_tuples = [(i, self._randseed("eval"), self._randseed("mcts"), self_before_training, PIT_NO_MCTS) for i in range(self.n_games_per_eval)]
        eval_results = self._starmap(self._play_for_eval, arg_tuples)
        self_before_training.net.pop_multiprocessing(before_multiprocessing_stash)
        self.net.pop_multiprocessing(my_multiprocessing_stash)
        for old_reward, new_reward, old_reward_no_mcts, new_reward_no_mcts in eval_results:
            old_rewards.append(old_reward)
            new_rewards.append(new_reward)
            old_rewards_no_mcts.append(old_reward_no_mcts)
            new_rewards_no_mcts.append(new_reward_no_mcts)

        elapsed = time.time() - start_time
        print(f"..evaluation done in {elapsed:.2f} seconds")

        # Print stats, keep new network iff it wins >= threshold_to_keep fraction of games
        old_rewards = np.array(old_rewards)
        new_rewards = np.array(new_rewards)
        print(f"Old network+MCTS average reward: {old_rewards.mean():.2f}, min: {old_rewards.min():.2f}, max: {old_rewards.max():.2f}, stdev: {old_rewards.std():.2f}")
        print(f"New network+MCTS average reward: {new_rewards.mean():.2f}, min: {new_rewards.min():.2f}, max: {new_rewards.max():.2f}, stdev: {new_rewards.std():.2f}")

        old_rewards_no_mcts = np.array(old_rewards_no_mcts)
        new_rewards_no_mcts = np.array(new_rewards_no_mcts)
        print(f"Old bare network average reward: {old_rewards_no_mcts.mean():.2f}, min: {old_rewards_no_mcts.min():.2f}, max: {old_rewards_no_mcts.max():.2f}, stdev: {old_rewards_no_mcts.std():.2f}")
        print(f"New bare network average reward: {new_rewards_no_mcts.mean():.2f}, min: {new_rewards_no_mcts.min():.2f}, max: {new_rewards_no_mcts.max():.2f}, stdev: {new_rewards_no_mcts.std():.2f}")

        wins = np.sum((new_rewards > old_rewards) & ~(np.isclose(new_rewards, old_rewards)))
        ties = np.sum(np.isclose(new_rewards, old_rewards))
        losses = np.sum((new_rewards < old_rewards) & ~(np.isclose(new_rewards, old_rewards)))
        assert wins + ties + losses == self.n_games_per_eval
        score = (wins + ties / 2) / self.n_games_per_eval  # a tie is half a win
        print(f"New network won {wins} and tied {ties} out of {self.n_games_per_eval} games ({score:.2%} wins where ties are half wins)")
        return score
    
    def play_train_multiple(self, n_trains: int):
        for i in range(n_trains):
            print(f"\nTraining iteration {i+1} of {n_trains}: will play {self.n_games_per_train} games, train, and evaluate on {self.n_games_per_eval} games")
            self.play_and_train()
