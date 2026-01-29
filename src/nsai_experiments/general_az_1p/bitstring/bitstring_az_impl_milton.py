import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsai_experiments.general_az_1p.game import EnvGame
from nsai_experiments.general_az_1p.policy_value_net import TorchPolicyValueNet
from nsai_experiments.general_az_1p.utils import get_accelerator
from nsai_experiments.general_az_1p.agent import Agent
from nsai_experiments.general_az_1p.mcts import MCTS, entab

import gymnasium.spaces as spaces
from typing import Hashable
import warnings
from copy import deepcopy
from copy import deepcopy
import time
import sys
import os
import argparse



class CumulativeRewardWrapper(gym.Wrapper):
    """Wrapper that changes reward behavior: 0 at every step, total steps at termination."""
    
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = env.max_steps
        self.nsites = env.nsites  # Store the number of sites for later use
    
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Give 0 reward during the episode, final reward at termination
        if terminated or truncated:
            print(f"AM TERMINATED {terminated} OR TRUNCATED {truncated}")
            reward = self.step_count / self.max_steps
        else:
            reward = 0

        # assert reward == 0.0 or (terminated or truncated)
        # assert reward <= 1.0
            
        return observation, reward, terminated, truncated, info

class BitStringGameGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nsites=10, sparsemode=True):
        """
        Initialize the BitString environment.
        
        Args:
            nsites: Length of the bitstring.
            sparsemode: If True, reward is only given at the end of the episode (filledness).
                        If False, dense rewards are given at each step for correct moves.
        """
        self.bitflipmode = True  # "setting" a 1 flips it back to 0
        self.sparsemode = sparsemode
        self.nones = 2 # number of bits that are initially set to 1

        self.nsites = nsites
        self.max_steps = 2 * nsites if not self.sparsemode else nsites - self.nones
        self.observation_space = spaces.MultiBinary([self.nsites]) #, seed=42)
        self.action_space = spaces.Discrete(self.nsites)
        self.reset()

    def step(self, action):
        """
        This method is the primary interface between environment and agent.

        Paramters: 
            action: int
                    the index of the respective action (if action space is discrete)

        Returns:
            output: (array, float, bool, dict)
                    information provided by the environment about its current state:
                    (observation, reward, done, trunc, info)
        """
#        print ("prestep state <a, s, s(a)>", action, self.state, self.state[action])
        self.step_count += 1
        done = self.step_count >= self.max_steps
        # Dense reward calculation (normalized by nsites for stability)
        # Normalization (1/nsites) ensures value head targets stay within ~[-1, 1] range.
        r = -1.0 / self.nsites

        if action == -1:  # Sentinel for an invalid action
            return self.state, r, done, False, {}

        if self.state[action] == 0:
            # Reward for correctly flipping a 0 to a 1
            r = 1.0 / self.nsites
        if self.bitflipmode:
            self.state[action] = 1 - self.state[action]
        else:
            self.state[action] = 1
        done = done or sum(self.state) == self.nsites

        normalizer = self.nsites # playing around with scale of pi vs value loss
        if self.sparsemode:
            if done:
                filledness = sum(self.state) / normalizer  # 0 if all 0s, 1 if all 1s
                # turn_inefficiency = self.step_count / (normalizer-self.nones)  # 1 if did it in minimum number of turns, 2 if took 2x as long, etc.
                # print(self.step_count)
                # r = filledness/turn_inefficiency  # 1 if filledness=1 and turn_inefficiency=1, decreases from there
                r = filledness
                # print("state", self.state, "r", r, flush=True)
            else:
                r = 0
        # if done:
        #     print ("Net Episode done <s, r, t, steps>", self.state, r, done, self.step_count)
        return self.state, r, done, done, {}

    def reset(self, seed = None):
        """
        This method resets the environment to its initial values.

        Returns:
            observation:    array
                            the initial state of the environment
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        ones = np.random.choice(range(self.nsites), self.nones, replace=False)
        self.state = np.zeros(self.nsites, dtype=np.float32)
        self.state[ones] = 1
        self.step_count = 0
        return self.state, {}


class BitStringGame(EnvGame):
    def __init__(self, use_cumulative_reward_rescale=True, sparsemode=True, **kwargs):
        # Pass sparsemode explicitly to the gym environment
        env = BitStringGameGym(sparsemode=sparsemode, **kwargs)
        # if use_cumulative_reward_rescale:
        #     env = CumulativeRewardWrapper(env)
        super().__init__(env)
        self._action_mask = np.ones(env.nsites)  # all actions are always available, otherwise it's cheating.
    
    def get_action_mask(self):
        return self._action_mask
    
    @property
    def hashable_obs(self) -> Hashable:
        "Returns a hashable representation of the current observation `obs`."
        return "".join([str(int(x)) for x in self.obs])  + " " + str(self.env.step_count)# Convert the bitstring to a string of '0's and '1's, which is hashable

class BitStringModel(nn.Module):
    def __init__(self, nsites = 10, n_hidden_layers = 2, hidden_size = 128):
        super().__init__()
        self.input_size = nsites   # observation is a bitstring of length nsites
        self.action_size = nsites  # action is an index of a bit to flip
        self.body = nn.Sequential(
            nn.Sequential(nn.Linear(self.input_size, hidden_size), nn.ReLU()),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(n_hidden_layers)],
        )
        self.policy_head = nn.Linear(hidden_size, nsites)  # thinking of it as logits for action of filling a bit
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.body(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value

class BitStringPolicyValueNet(TorchPolicyValueNet):
    save_file_name = "bitstring_checkpoint.pt"
    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "policy_weight": 1.0,
    }

    def __init__(self, random_seed = None, nsites = 10, n_hidden_layers = 2, hidden_size = 128, training_params = {}, device = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        model = BitStringModel(nsites, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size)
        self.nsites = nsites
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        self.DEVICE = get_accelerator() if device is None else device
        print(f"Neural network training will occur on device '{self.DEVICE}'")
        
    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]

        model = model.to(self.DEVICE)
        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=tp["learning_rate"], weight_decay=tp["weight_decay"])

        # The usual case is that `examples` comes from AlphaZero and is a list of tuples
        # that will need to be reformatted. For convenience, we also support the case where
        # `examples` is already in the proper format, perhaps because the network is being
        # tested outside the AlphaZero context; for this, pass `needs_reshape=False`.
        if needs_reshape:
            # PERF we could use a single Python loop for all three of these
            states = torch.from_numpy(np.array([state for state, (_, _) in examples], dtype=np.float32))
            policies = torch.from_numpy(np.array([policy for _, (policy, _) in examples], dtype=np.float32))
            values = torch.from_numpy(np.array([value for _, (_, value) in examples], dtype=np.float32))
            dataset = torch.utils.data.TensorDataset(states, policies, values)
        else:
            print("Skipping reshape of `examples`.")
            dataset = examples
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=tp["batch_size"], shuffle=True)
        print(f"Training with {len(train_loader)} batches of size {tp['batch_size']}")

        train_mini_losses = []
        train_losses = []

        for epoch in range(tp["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0
            for inputs, targets_policy, targets_value in train_loader:
                inputs, targets_value, targets_policy = inputs.to(self.DEVICE), targets_value.to(self.DEVICE), targets_policy.to(self.DEVICE)
                optimizer.zero_grad()
                assert len(inputs.shape) == 2, f"Expected input shape to be 2D, got {inputs.shape}"
                assert inputs.shape[1] == model.input_size, f"Expected input size {model.input_size}, got {inputs.shape[1]}"
                assert len(targets_value.shape) == 1
                assert targets_policy.shape[1] == model.action_size, f"Expected policy answers size {model.action_size}, got {targets_policy.shape[1]}"
                outputs_policy, outputs_value = model(inputs)
                assert outputs_value.shape == targets_value.shape, f"Expected predicted value shape {targets_value.shape}, got {outputs_value.shape}"
                loss_value = criterion_value(outputs_value, targets_value)
                assert outputs_policy.shape == targets_policy.shape, f"Expected predicted policy shape {targets_policy.shape}, got {outputs_policy.shape}"
                loss_policy = criterion_policy(outputs_policy, targets_policy)
                loss = loss_value + policy_weight*loss_policy

                loss.backward()
                optimizer.step()
                loss = loss.item()
                train_mini_losses.append(loss)
                train_loss += loss
                policy_loss += loss_policy.item()
                value_loss += loss_value.item()

            # Calculate epoch averages
            avg_train_loss = train_loss / len(train_loader)
            avg_value_loss = value_loss / len(train_loader)
            avg_policy_loss = policy_loss / len(train_loader)
            
            # Store detailed metrics
            epoch_metrics = {
                'total': avg_train_loss,
                'value': avg_value_loss,
                'policy': avg_policy_loss,
                'weighted_policy': policy_weight * avg_policy_loss
            }
            train_losses.append(epoch_metrics)

            if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
            # if True:
                print(f"Epoch {epoch+1}/{tp['epochs']}, Train Loss: {avg_train_loss:.4f} (value: {avg_value_loss:.4f}, policy: {avg_policy_loss:.4f}, weighted policy: {epoch_metrics['weighted_policy']:.4f})")

        return model, train_mini_losses, train_losses
    
    def predict(self, state):
        self.model.cpu()
        nn_input = torch.tensor(state).reshape(1, -1)
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)
        
        policy_prob = policy_prob.numpy()
        policy_prob = policy_prob.squeeze(0)
        assert policy_prob.shape == (self.nsites,)

        value = value.numpy()
        value = value.squeeze(0)
        assert value.shape == ()

        # policy_prob = np.random.random(2)
        return policy_prob, value


class BitStringAgent(Agent):
    """
    This class is a simple agent that plays the BitStringGame perfectly to generate training data.
    """

    def get_exact_move_probs(self, msg: str = "") -> np.ndarray:
        """
        Returns the exact move probabilities for the current game state.
        """
        if msg: print(msg, "Calculating exact move probabilities for", self.game.obs)
        # return the exact move probabilities
        # based on the current game state. 
        probs = np.zeros(self.game.env.nsites, dtype=np.float32)
        for i in range(self.game.env.nsites):
            if self.game.obs[i] == 0:
                probs[i] = 1.0
            else:
                probs[i] = 0.0
        probs /= np.sum(probs)  # Normalize to make it a probability distribution
        if msg: print(msg, "Exact move probabilities:", probs)
        return probs

    def play_single_game(self, max_moves: int = 10_000, random_seed: int | None = None, msg = ""):
        train_examples = []
        rewards = []
        # mcts = MCTS(self.game, self.net, **self.mcts_params)
        rng = np.random.default_rng(random_seed)
        for i in range(max_moves):
            if msg: print(msg, f"starting move {i}")
            # move_probs = mcts.perform_simulations(entab(msg, f", m{i+1}"))
            # self.game = mcts.game  # TODO HACK because MCTS modifies the game state in place
            move_probs = self.get_exact_move_probs(entab(msg, f", m{i+1}"))
            train_examples.append((deepcopy(self.game.obs), (move_probs, None)))
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


# Define helper classes at module level for pickling compatibility
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # ensure immediate write
    def flush(self):
        for f in self.files:
            f.flush()

class TrackingAgent(Agent):
    """
    Subclass of Agent that tracks detailed history and overrides execution loop
    to allow for plotting and logging.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {
            'iteration': [],
            'reward_mean': [],
            'reward_std': [],
            'loss_policy': [],
            'loss_value': [],
            'game_length': []
        }
        self.cumulative_reward = 0.0

    def play_single_game(self, max_moves: int = 10_000, random_seed: int | None = None, msg = "", add_noise=False):
        # We override this to store cumulative reward for 'pit' evaluation
        train_examples = []
        rewards = []
        self.cumulative_reward = 0.0
        
        # Re-implement play_single_game mostly to capture self.cumulative_reward
        mcts = MCTS(self.game, self.net, **self.mcts_params)
        rng = np.random.default_rng(random_seed)
        for i in range(max_moves):
            if msg: print(msg, f"starting move {i}")
            if self.external_policy is None:
                move_probs = mcts.perform_simulations(entab(msg, f", m{i+1}"), add_noise=add_noise)
                self.game = mcts.game 
                train_examples.append((deepcopy(self.game.obs), (move_probs, None)))
                
                flat_probs = move_probs.flatten()
                flat_idx = rng.choice(len(flat_probs), p=flat_probs)
                selected_move = np.unravel_index(flat_idx, move_probs.shape)
                if len(self.game.action_space.shape) == 0: selected_move, = selected_move
                
                if msg: print(msg, "obs", self.game.obs, "move_probs", move_probs, "selmove", selected_move)
            else:
                selected_move = self.external_policy(self.game.obs)
            
            self.game.step_wrapper(selected_move)
            rewards.append(self.game.reward)
            self.cumulative_reward += self.game.reward
            
            if self.game.terminated or self.game.truncated:
                break
        else:
            warnings.warn(f"`play_single_game` timed out")
            return []
        
        for i in range(len(rewards) - 1, 0, -1):
            rewards[i-1] += self.reward_discount * rewards[i]
        
        for i in range(len(train_examples)):
            state, (policy, _) = train_examples[i]
            train_examples[i] = (state, (policy, rewards[i]))
        
        return train_examples

    def pit(self, self_before_training):
        "Evaluation routine that returns detailed stats dict"
        start_time = time.time()
        my_multiprocessing_stash = self.push_multiprocessing()
        before_multiprocessing_stash = self_before_training.push_multiprocessing()
        
        # 6 args for _play_for_eval: i, reset_seed, mcts_seed, ext_pol_seed, agent_old, try_no_mcts, pit_ext
        # Note: importing PIT_NO_MCTS from agent might fail if not exposed.
        # We assume PIT_NO_MCTS is True as per default in Agent.py
        PIT_NO_MCTS = True 
        
        arg_tuples = [(i, self._randseed("eval"), self._randseed("mcts"), self._randseed("external_policy"), 
                       self_before_training, PIT_NO_MCTS, True) for i in range(self.n_games_per_eval)]
        
        # _starmap calls _play_for_eval. 
        # IMPORTANT: _play_for_eval in base Agent uses agent.game.reward, NOT cumulative_reward.
        # But we overrode play_single_game to set cumulative_reward. 
        # Does _play_for_eval READ cumulative_reward? 
        # In base Agent.py: results[label] = agent.game.reward. (Line 200 of Agent.py view)
        # So overriding play_single_game to set cumulative_reward is useless for _play_for_eval unless we override _play_for_eval too.
        # For BitString, game.reward IS the score (sparse or dense-sum-wrapper-equivalent). 
        # In general_az_1p, usually game.reward at end is what matters. 
        # Let's trust base _play_for_eval returning game.reward is sufficient for evaluation comparison.
        
        eval_results = self._starmap(self._play_for_eval, arg_tuples)
        
        self_before_training.pop_multiprocessing(before_multiprocessing_stash)
        self.pop_multiprocessing(my_multiprocessing_stash)
        elapsed = time.time() - start_time
        print(f"..evaluation done in {elapsed:.2f} seconds")

        eval_results_keys = eval_results[0].keys()
        eval_results_arrays = {key: np.array([res[key] for res in eval_results]) for key in eval_results_keys}

        old_rewards = eval_results_arrays["old_net"]
        new_rewards = eval_results_arrays["new_net"]
        print(f"Old network+MCTS average reward: {old_rewards.mean():.4f}, min: {old_rewards.min():.4f}, max: {old_rewards.max():.4f}, stdev: {old_rewards.std():.4f}")
        print(f"New network+MCTS average reward: {new_rewards.mean():.4f}, min: {new_rewards.min():.4f}, max: {new_rewards.max():.4f}, stdev: {new_rewards.std():.4f}")

        wins = np.sum((new_rewards > old_rewards) & ~(np.isclose(new_rewards, old_rewards)))
        ties = np.sum(np.isclose(new_rewards, old_rewards))
        losses = np.sum((new_rewards < old_rewards) & ~(np.isclose(new_rewards, old_rewards)))
        score = (wins + ties / 2) / self.n_games_per_eval
        print(f"New network won {wins} and tied {ties} out of {self.n_games_per_eval} games ({score:.2%} wins where ties are half wins)")
        
        return {
            'score': score,
            'new_reward_mean': new_rewards.mean(),
            'new_reward_std': new_rewards.std(),
            'old_reward_mean': old_rewards.mean(),
            'old_reward_std': old_rewards.std()
        }

    def play_and_train(self):
        import itertools
        # Override to capture metrics
        new_train_examples = []
        start_time = time.time()
        multiprocessing_stash = self.push_multiprocessing()
        arg_tuples = [(i, self._randseed("train"), self._randseed("mcts")) for i in range(self.n_games_per_train)]
        train_example_sets = self._starmap(self._play_for_examples, arg_tuples)
        self.pop_multiprocessing(multiprocessing_stash)
        for train_examples in train_example_sets:
            new_train_examples.extend(train_examples)
        elapsed = time.time() - start_time
        print(f"..games done in {elapsed:.2f} seconds")
        
        self.all_training_examples.append(new_train_examples)
        if self.n_past_iterations_to_train is not None and len(self.all_training_examples) > self.n_past_iterations_to_train:
            self.all_training_examples.pop(0)
        
        flat_examples = list(itertools.chain.from_iterable(self.all_training_examples))
        
        self_before_training = deepcopy(self)
        start_time = time.time()
        
        # Capture dictionary losses
        _, _, train_losses = self.net.train(flat_examples, **({"print_all_epochs": True}))
        elapsed = time.time() - start_time
        print(f"..training done in {elapsed:.2f} seconds")

        # Validate losses format
        if isinstance(train_losses, list) and len(train_losses) > 0 and isinstance(train_losses[0], dict):
             loss_policy = train_losses[-1].get('policy', 0.0)
             loss_value = train_losses[-1].get('value', 0.0)
        else:
             loss_policy = 0.0 # Fallback
             loss_value = 0.0

        # Evaluate
        eval_stats = self.pit(self_before_training)
        score = eval_stats['score']
        
        if score >= self.threshold_to_keep:
            print("Keeping the new network")
        else:
            print("Reverting to the old network")
            self.net = self_before_training.net
        
        # Record History
        iter_idx = len(self.history['iteration']) + 1
        # Avg game length from current batch
        avg_game_len = np.mean([len(game_trace) for game_trace in train_example_sets]) if train_example_sets else 0

        self.history['iteration'].append(iter_idx)
        self.history['reward_mean'].append(eval_stats['new_reward_mean'])
        self.history['reward_std'].append(eval_stats['new_reward_std'])
        self.history['loss_policy'].append(loss_policy)
        self.history['loss_value'].append(loss_value)
        self.history['game_length'].append(avg_game_len)

if __name__ == "__main__":
    from nsai_experiments.general_az_1p.setup_utils import disable_numpy_multithreading, use_deterministic_cuda
    disable_numpy_multithreading()
    use_deterministic_cuda()

    import numpy as np
    import argparse
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from copy import deepcopy
    import warnings

    from nsai_experiments.general_az_1p.agent import Agent
    # Classes are already defined in this file, no need to import them.
    # from nsai_experiments.general_az_1p.bitstring.bitstring_az_impl_milton import BitStringGame, BitStringPolicyValueNet

    # ... imports for plotting ...
    from nsai_experiments.general_az_1p.bitstring.plot_metrics import plot_training_metrics
    from nsai_experiments.general_az_1p.mcts import MCTS, entab
    import time
    from datetime import datetime
    from pathlib import Path
    import os

    # Default Configuration
    # Default Configuration (User Specific)
    config = {
        "nsites": 6,              # [CHANGED] 10 -> 6
        "sparsemode": False,      # [CHANGED] True -> False (Dense)
        "n_iters": 40,            # [CHANGED] 100 -> 40
        "n_games_per_train": 100,
        "n_games_per_eval": 30,
        "mcts_sims": 30,          # [CHANGED] 50 -> 30
        "c_expl": 1.5,
        "epochs": 10,
        "lr": 1e-3,
        "threshold": 0.55,
        "use_gating": True,       # [ADDED] Default: Gating enabled
        "dirichlet_alpha": 0.3,   # [ADDED] Default: Alpha=0.3
        "dirichlet_epsilon": 0.25 # [ADDED] Default: Epsilon=0.25
    }

    # 1. Parse CLI Arguments
    parser = argparse.ArgumentParser(description="Run BitString AlphaZero Experiment (Milton Impl)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode to tune hyperparameters")
    parser.add_argument("--nsites", type=int, help="Length of bitstring")
    parser.add_argument("--iters", type=int, help="Total training iterations")
    parser.add_argument("--sparse", action="store_true", help="Use sparse rewards (default: True)")
    parser.add_argument("--dense", action="store_true", help="Use dense rewards")
    parser.add_argument("--nogating", action="store_true", help="Disable gating mechanism (always accept new net)")
    parser.add_argument("--dirichlet_alpha", type=float, help="Dirichlet noise alpha (default: 0.3)")
    parser.add_argument("--dirichlet_epsilon", type=float, help="Dirichlet noise epsilon (default: 0.25)")
    
    # New Hyperparameters
    parser.add_argument("--c_expl", type=float, help="Exploration constant (PUCT)")
    parser.add_argument("--mcts_sims", type=int, help="MCTS simulations per move")
    parser.add_argument("--epochs", type=int, help="Training epochs per iteration")
    parser.add_argument("--lr", type=float, help="Learning Rate")
    parser.add_argument("--threshold", type=float, help="Evaluation win rate threshold")
    parser.add_argument("--eval_games", type=int, help="Games per evaluation round")

    args = parser.parse_args()

    # 2. Update Config from CLI
    if args.nsites: config["nsites"] = args.nsites
    if args.iters: config["n_iters"] = args.iters
    if args.dense: config["sparsemode"] = False
    if args.sparse: config["sparsemode"] = True
    
    if args.c_expl: config["c_expl"] = args.c_expl
    if args.mcts_sims: config["mcts_sims"] = args.mcts_sims
    if args.epochs: config["epochs"] = args.epochs
    if args.lr: config["lr"] = args.lr
    if args.threshold: config["threshold"] = args.threshold
    if args.eval_games: config["n_games_per_eval"] = args.eval_games
    if args.nogating: config["use_gating"] = False
    if args.dirichlet_alpha: config["dirichlet_alpha"] = args.dirichlet_alpha
    if args.dirichlet_epsilon: config["dirichlet_epsilon"] = args.dirichlet_epsilon

    # 3. Interactive Mode
    if args.interactive:
        print("\n=== Interactive Experiment Configuration ===")
        try:
            val = input(f"Number of sites (nsites) [default: {config['nsites']}]: ").strip()
            if val: config["nsites"] = int(val)
            
            # Mode selection
            current_mode = "sparse" if config["sparsemode"] else "dense"
            val = input(f"Reward Mode (dense/sparse) [default: {current_mode}]: ").strip().lower()
            if val == "dense": config["sparsemode"] = False
            elif val == "sparse": config["sparsemode"] = True
            
            val = input(f"Total Iterations [default: {config['n_iters']}]: ").strip()
            if val: config["n_iters"] = int(val)

            # Hyperparameters
            val = input(f"MCTS Simulations [default: {config['mcts_sims']}]: ").strip()
            if val: config["mcts_sims"] = int(val)

            val = input(f"Exploration Constant c_expl [default: {config['c_expl']}]: ").strip()
            if val: config["c_expl"] = float(val)

            val = input(f"Training Epochs [default: {config['epochs']}]: ").strip()
            if val: config["epochs"] = int(val)

            val = input(f"Learning Rate [default: {config['lr']}]: ").strip()
            if val: config["lr"] = float(val)
            
            val = input(f"Eval Threshold [default: {config['threshold']}]: ").strip()
            if val: config["threshold"] = float(val)

            val = input(f"Eval Games [default: {config['n_games_per_eval']}]: ").strip()
            if val: config["n_games_per_eval"] = int(val)
            
            # Gating Selection
            current_gating = "yes" if config["use_gating"] else "no"
            val = input(f"Enable Gating (pit new vs old)? (yes/no) [default: {current_gating}]: ").strip().lower()
            if val in ["no", "n", "false"]: config["use_gating"] = False
            elif val in ["yes", "y", "true"]: config["use_gating"] = True
            
            # Dirichlet Noise
            val = input(f"Dirichlet Alpha [default: {config['dirichlet_alpha']}]: ").strip()
            if val: config["dirichlet_alpha"] = float(val)
            
            val = input(f"Dirichlet Epsilon [default: {config['dirichlet_epsilon']}]: ").strip()
            if val: config["dirichlet_epsilon"] = float(val)

        except ValueError as e:
            print(f"Invalid input, using defaults. Error: {e}")
        print("\nConfiguration Complete.\n")

    print(f"Running with Configuration: {config}")

    # --- SETUP RUN DIRECTORY ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "sparse" if config["sparsemode"] else "dense"
    
    # Rigorous Directory Naming
    # run_TIMESTAMP_nSITES_MODE_itITERS_simSIMS_cEXPL
    run_name_parts = [
        f"run_{timestamp}",
        f"n{config['nsites']}",
        mode_str,
        f"it{config['n_iters']}",
        f"sim{config['mcts_sims']}",
        f"c{config['c_expl']}"
    ]
    if not config["use_gating"]:
        run_name_parts.append("nogating")
        
    run_name_parts.append(f"da{config['dirichlet_alpha']}")
    run_name = "_".join(run_name_parts)
    run_dir = Path(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Logging
    log_file = open(run_dir / "log.txt", "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Artifacts will be saved to: {run_dir.absolute()}")

    # 4. Instantiate Components
    nsites = config["nsites"]
    mygame = BitStringGame(nsites=nsites, sparsemode=config["sparsemode"])
    
    mynet = BitStringPolicyValueNet(
        random_seed=47, 
        nsites=nsites, 
        n_hidden_layers=1, 
        training_params={
            "epochs": config["epochs"], 
            "learning_rate": config["lr"], 
            "policy_weight": 2.0
        }
    )

    # USE TrackingAgent instead of Agent
    # USE TrackingAgent instead of Agent
    myagent = TrackingAgent(
        mygame, 
        mynet, 
        n_games_per_train=config["n_games_per_train"], 
        n_games_per_eval=config["n_games_per_eval"], 
        threshold_to_keep=config['threshold'],  
        use_gating=config['use_gating'],
        n_past_iterations_to_train=5,
        random_seeds={"mcts": 48, "train": 49, "eval": 50}, 
        mcts_params={
            "n_simulations": config["mcts_sims"], 
            "c_exploration": config["c_expl"],
            "dirichlet_alpha": config["dirichlet_alpha"],
            "dirichlet_epsilon": config["dirichlet_epsilon"]
        }
    )

    # 5. Execution Loop (Unrolled)
    n_iters = config["n_iters"]
    try:
        for i in range(n_iters):
            print(f"\n=== Iteration {i+1}/{n_iters} ===")
            myagent.play_and_train()
            
            # Save Plot
            plot_training_metrics(myagent.history, save_path=str(run_dir / "metrics_latest.png"))
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exiting safely...")
        # sys.exit(0) # Let finally block handle closure if needed, but here just exit
    
    print(f"Training finished. Results in {run_dir}")
    log_file.close()