import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from nsai_experiments.general_az_1p.game import EnvGame
from nsai_experiments.general_az_1p.policy_value_net import TorchPolicyValueNet

from nsai_experiments.zoning_game.zg_gym import ZoningGameEnv, flatten_zg_obs
from nsai_experiments.zoning_game.zg_gym import Tile

# multiprocessing doesn't like anonymous functions so we can't use TransformReward(... lambda...)
class ScaleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale

class ZoningGameGame(EnvGame):
    default_env_kwargs = {
        "populate_info": False,
    }
    def __init__(self, rescale_rewards = True, *args, **kwargs):
        env_kwargs = self.default_env_kwargs | kwargs
        env = ZoningGameEnv(*args, **env_kwargs)
        if rescale_rewards:
            divisor = env.grid_size*env.grid_size*3  # TODO tune empirically
            env = ScaleRewardWrapper(env, 1/divisor)
        super().__init__(env)
    
    def get_action_mask(self):
        tile_grid, _ = self.obs  # type: ignore
        return np.ravel(tile_grid == Tile.EMPTY.value)
    
    # Purely for performance, we override stash_state and unstash_state to avoid generic deepcopy
    # TODO once we have determinism, verify that things run exactly the same with and without this
    def stash_state(self):
        return (
            tuple(x.copy() for x in self.obs),  # type: ignore
            self.reward,
            self.terminated,
            self.truncated,
            copy.deepcopy(self.info),
            self.env.unwrapped.tile_grid.copy(),  # type: ignore
            self.env.unwrapped.tile_queue.copy(),  # type: ignore
            self.env.unwrapped.n_moves  # type: ignore
        )
    
    def unstash_state(self, state):
        """
        Returns this game reverted to the state represented by `state`, which came from
        `stash_state`. After this is called, the object on which it is called should no
        longer be used.
        """
        obs, reward, terminated, truncated, info, tile_grid, tile_queue, n_moves = state
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.env.unwrapped.tile_grid = tile_grid  # type: ignore
        self.env.unwrapped.tile_queue = tile_queue  # type: ignore
        self.env.unwrapped.n_moves = n_moves  # type: ignore
        return self


class ZoningGameModel(nn.Module):
    def __init__(self, grid_size = 6):
        super().__init__()
        self.grid_size = grid_size
        self.grid_length = grid_size*grid_size
        self.input_length = self.grid_length*2
        self.num_classes = len(Tile)

        kernel_size = 3
        out_channels = self.num_classes*kernel_size*kernel_size
        n_into_linear = self.grid_length*out_channels + self.grid_length*self.num_classes
        n_hidden = self.input_length*self.num_classes
        
        self.conv1 = nn.Conv2d(self.num_classes, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.linear_relu_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_into_linear, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(n_hidden, self.grid_length)
        self.value_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        assert x.shape[-1] == self.input_length
        x = F.one_hot(x, num_classes=self.num_classes).to(torch.float32)

        x_grid = x[:, :self.grid_length, :].reshape(-1, self.grid_size, self.grid_size, self.num_classes)
        x_grid = self.conv1(x_grid)
        x_grid = torch.flatten(x_grid, start_dim=1)

        x_queue = x[:, self.grid_length:]
        x_queue = torch.flatten(x_queue, start_dim=1)

        x = torch.cat((x_grid, x_queue), dim=1)
        x = self.linear_relu_stack(x)

        y_value = self.value_head(x)
        y_policy = self.policy_head(x)
        return y_policy, y_value


class ZoningGamePolicyValueNet(TorchPolicyValueNet):
    save_file_name = "zg_checkpoint.pt"
    default_training_params = {
        "epochs": 10,
        "batch_size": 2048,
        "learning_rate": 0.001,
        "l1_lambda": 0,
        "weight_decay": 1e-5,
        "value_weight": 50.0,
        "policy_weight": 1.0,
        "persist_optimizer": True,  # if False, reinitialize the optimizer every train() call
    }

    def __init__(self, grid_size = 6, random_seed = None, training_params = {}, device = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        model = ZoningGameModel(grid_size = grid_size)
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        self.grid_size = grid_size

        self.DEVICE = (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu") if device is None else device
        print(f"Neural network training will occur on device '{self.DEVICE}'")
        self.optimizer = None

    def _reshape_data(self, examples):
        # The usual case is that `examples` comes from AlphaZero and is a list of tuples
        # that will need to be reformatted.

        # PERF we could use a single Python loop for all three of these
        states = torch.from_numpy(np.array([flatten_zg_obs(state) for state, (_, _) in examples], dtype=np.int64))
        policies = torch.from_numpy(np.array([policy for _, (policy, _) in examples], dtype=np.float32))
        values = torch.from_numpy(np.array([value for _, (_, value) in examples], dtype=np.float32))
        dataset = torch.utils.data.TensorDataset(states, policies, values)
        return dataset
    
    def train(self, examples, val_dataset = None, needs_reshape=True, print_all_epochs=False):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params

        if self.optimizer is None or not tp["persist_optimizer"]:
            print("Initializing optimizer")
            self.optimizer = torch.optim.Adam(model.parameters(), lr=tp["learning_rate"], weight_decay=tp["weight_decay"])

        value_weight = tp["value_weight"]
        policy_weight = tp["policy_weight"]
        l1_lambda = tp["l1_lambda"]

        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        dataset = self._reshape_data(examples) if needs_reshape else examples
        if val_dataset is not None and needs_reshape:
            val_dataset = self._reshape_data(val_dataset)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=tp["batch_size"], shuffle=True)
        print(f"Training with {len(train_loader)} batches of size {tp['batch_size']}")

        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=tp["batch_size"], shuffle=False)
            print(f"Validating with {len(val_loader)} batches of size {tp['batch_size']}")

        train_mini_losses = []
        train_losses = []
        train_mean_maxes = []
        val_losses = []

        for epoch in range(tp["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0
            train_mean_max = 0.0
            for inputs, targets_policy, targets_value in train_loader:
                inputs, targets_value, targets_policy = inputs.to(self.DEVICE), targets_value.to(self.DEVICE), targets_policy.to(self.DEVICE)
                self.optimizer.zero_grad()
                outputs_policy, outputs_value = model(inputs)
                loss_value = criterion_value(outputs_value.squeeze(), targets_value)
                mean_max = outputs_policy.softmax(dim=-1).max(dim=-1).values.mean()
                loss_policy = criterion_policy(outputs_policy.squeeze(), targets_policy)
                loss = value_weight*loss_value + policy_weight*loss_policy

                # Add L1 regularization
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                train_mini_losses.append(loss)
                train_loss += loss
                policy_loss += loss_policy
                value_loss += loss_value
                train_mean_max += mean_max

            train_losses.append(train_loss / len(train_loader))
            train_mean_maxes.append(train_mean_max / len(train_loader))

            epoch_msg = f"Epoch {epoch+1}/{tp['epochs']}, Train Loss: {train_losses[-1]:.4f} (value: {value_loss / len(train_loader):.4f}, weighted value: {value_weight * (value_loss / len(train_loader)):.4f}, policy: {policy_loss / len(train_loader):.4f}, weighted policy: {policy_weight * (policy_loss / len(train_loader)):.4f}), Train Mean Max: {train_mean_maxes[-1]:.4f}"
            # Validation phase
            if val_dataset is not None:
                val_loss, val_mean_max = self.validate_inner(val_loader)
                val_losses.append(val_loss)
                
                if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
                    print(f"{epoch_msg}, Val Loss: {val_loss:.4f}, Val Mean Max: {val_mean_max:.4f}")
            else:
                if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
                    print(epoch_msg)
        return model, train_mini_losses, train_losses

    def validate_inner(self, val_loader):
        model = self.model
        model.to(self.DEVICE)
        model.eval()
        
        tp = self.training_params
        policy_weight = tp["policy_weight"]
        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        
        val_loss = 0.0
        val_mean_max = 0.0
        with torch.no_grad():
            for inputs, targets_policy, targets_value in val_loader:
                inputs, targets_value, targets_policy = inputs.to(self.DEVICE), targets_value.to(self.DEVICE), targets_policy.to(self.DEVICE)
                outputs_policy, outputs_value = model(inputs)
                loss_value = criterion_value(outputs_value.squeeze(), targets_value)
                mean_max = outputs_policy.softmax(dim=-1).max(dim=-1).values.mean()
                loss_policy = criterion_policy(outputs_policy.squeeze(), targets_policy)
                loss = loss_value + policy_weight*loss_policy
                val_loss += loss.item()
                val_mean_max += mean_max.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mean_max = val_mean_max / len(val_loader)
        return avg_val_loss, avg_val_mean_max

    def validate(self, examples, needs_reshape=True):
        dataset = self._reshape_data(examples) if needs_reshape else examples
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.training_params["batch_size"], shuffle=False)
        return self.validate_inner(val_loader)

    def predict(self, state):
        self.model.cpu()
        # Here is the place to prepare raw game state observations for the neural network
        nn_input = torch.tensor(flatten_zg_obs(state)).reshape(1, -1)
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)
        return policy_prob.numpy().squeeze(), value.numpy().squeeze()
    
    def push_multiprocessing(self):
        super().push_multiprocessing()
        optimizer = self.optimizer  # the optimizer can't be sent to CPU, so we'll stash it in caller state so we don't try to copy it
        self.optimizer = None
        return optimizer

    def pop_multiprocessing(self, optimizer):
        super().pop_multiprocessing(optimizer)
        self.optimizer = optimizer
