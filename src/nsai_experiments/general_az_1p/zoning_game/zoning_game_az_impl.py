import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsai_experiments.general_az_1p.game import EnvGame
from nsai_experiments.general_az_1p.policy_value_net import TorchPolicyValueNet

from nsai_experiments.zoning_game.zg_gym import ZoningGameEnv, flatten_zg_obs
from nsai_experiments.zoning_game.zg_gym import Tile

class ZoningGameGame(EnvGame):
    def __init__(self, *args, **kwargs):
        env = ZoningGameEnv(*args, **kwargs)
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
            self.env.tile_grid.copy(),  # type: ignore
            self.env.tile_queue.copy(),  # type: ignore
            self.env.n_moves  # type: ignore
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
        self.env.tile_grid = tile_grid  # type: ignore
        self.env.tile_queue = tile_queue  # type: ignore
        self.env.n_moves = n_moves  # type: ignore
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
        "batch_size": 1024,
        "learning_rate": 0.001,
        "l1_lambda": 0,
        "weight_decay": 5e-3,
        "policy_weight": 4.0,
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

    def train(self, examples, needs_reshape=True):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]
        l1_lambda = tp["l1_lambda"]

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
            states = torch.from_numpy(np.array([flatten_zg_obs(state) for state, (_, _) in examples], dtype=np.int64))
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
            for inputs, targets_policy, targets_value in train_loader:
                inputs, targets_value, targets_policy = inputs.to(self.DEVICE), targets_value.to(self.DEVICE), targets_policy.to(self.DEVICE)
                optimizer.zero_grad()
                outputs_policy, outputs_value = model(inputs)
                loss_value = criterion_value(outputs_value.squeeze(), targets_value)
                loss_policy = criterion_policy(outputs_policy.squeeze(), targets_policy)
                loss = loss_value + policy_weight*loss_policy

                # Add L1 regularization
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

                loss.backward()
                optimizer.step()
                loss = loss.item()
                train_mini_losses.append(loss)
                train_loss += loss

            train_losses.append(train_loss / len(train_loader))
            if epoch == 0 or epoch == tp["epochs"] - 1:
                print(f"Epoch {epoch+1}/{tp['epochs']}, Train Loss: {train_losses[-1]:.4f}")

        return model, train_mini_losses, train_losses

    def predict(self, state):
        self.model.cpu()
        # Here is the place to prepare raw game state observations for the neural network
        nn_input = torch.tensor(flatten_zg_obs(state)).reshape(1, -1)
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)
        return policy_prob.numpy().squeeze(), value.item()
