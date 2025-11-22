# TODO there's a good deal of duplication between this and `zoning_game_az_impl.py`; consider a common superclass

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from nsai_experiments.general_az_1p.game import EnvGame
from nsai_experiments.general_az_1p.policy_value_net import TorchPolicyValueNet
from nsai_experiments.general_az_1p.utils import ScaleRewardWrapper

from nsai_experiments.zoning_game.zl_gym import ZoningLangEnv

class ZoningLangGame(EnvGame):
    DEFAULT_RESCALE_REWARDS = 70.0  # TODO tune empirically
    DEFAULT_ENV_KWARGS = {}
    
    def __init__(self, rescale_rewards = None, *args, **kwargs):
        "rescale_rewards: `False` for no rescaling, `None` for default rescaling, or a number to rescale by that number"
        if rescale_rewards is None:
            rescale_rewards = self.DEFAULT_RESCALE_REWARDS
        env_kwargs = self.DEFAULT_ENV_KWARGS | kwargs
        env = ZoningLangEnv(*args, **env_kwargs)
        if rescale_rewards is not False:
            env = ScaleRewardWrapper(env, 1/rescale_rewards)
        super().__init__(env)
    
    def get_action_mask(self):
        # PERF this could be optimized
        """Return binary tensor of shape (max_length, num_productions) indicating valid actions."""
        mask = np.zeros((self.env.unwrapped.max_length, self.env.unwrapped.max_productions), dtype=bool)
        nonterminal_indices = self.env.unwrapped.get_indices_of_nonterminals(self.obs)
        
        for nonterminal_idx in nonterminal_indices:
            nonterminal_token = self.obs[nonterminal_idx]
            productions = self.env.unwrapped.get_productions_for_nonterminal(nonterminal_token)
            for production_idx, _ in productions:
                mask[nonterminal_idx, production_idx] = True
        
        return mask
    
    def stash_state(self):
        return (
            self.obs.copy(),
            self.reward,
            self.terminated,
            self.truncated,
            copy.deepcopy(self.info),
            self.env.unwrapped.current_program.copy(),
            self.env.unwrapped.n_moves
        )
    
    def unstash_state(self, state):
        (
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            current_program,
            n_moves
        ) = state
        self.env.unwrapped.current_program = current_program
        self.env.unwrapped.n_moves = n_moves
        return self

class ZoningLangModel(nn.Module):
    """Encoder-only transformer for CFG-based AlphaZero.
    
    Takes a sequence of tokens (terminals + nonterminals) and outputs:
    - Policy: logits for production rules at each position
    - Value: scalar estimate of position value
    """
    def __init__(self, 
                 num_input_tokens,
                 max_seq_len,
                 num_productions,
                 d_model=256,
                 nhead=4,
                 num_layers=3,
                 dim_feedforward=512):
        super().__init__()
        
        # Input embedding
        self.token_embedding = nn.Embedding(num_input_tokens, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Shared transformer encoder (bidirectional attention, no causality)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Policy head: predict production for each position
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_productions)
        )
        
        # Value head: predict scalar value from sequence
        # Uses mean pooling over sequence
        self.value_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            # TODO think about shape, bounds
        )
    
    def forward(self, x, src_key_padding_mask=None):
        """Forward pass.
        
        Args:
            x: (batch, seq_len) - token indices
            src_key_padding_mask: (batch, seq_len) - True for padding positions to ignore
            
        Returns:
            policy_logits: (batch, seq_len, num_productions)
            value: (batch, 1)
        """
        batch_size, seq_len = x.shape
        embedded = self.token_embedding(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:seq_len, :]
        
        # Transformer encoding (full bidirectional attention)
        encoded = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        
        # Policy head: per-position logits over all productions
        policy_logits = self.policy_head(encoded)  # (batch, seq_len, num_productions)
        
        # Value head: masked mean pooling to exclude padding
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_encoded = encoded * mask
            pooled = masked_encoded.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)
        
        value = self.value_head(pooled)  # (batch, 1)
        
        return policy_logits, value

class ZoningLangPolicyValueNet(TorchPolicyValueNet):
    save_file_name = "zl_checkpoint.pt"
    default_training_params = {
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "l1_lambda": 0,
        "weight_decay": 1e-5,
        "value_weight": 50.0,
        "policy_weight": 1.0,
        "persist_optimizer": True,  # if False, reinitialize the optimizer every train() call
    }

    def __init__(self, random_seed=None, training_params={}, device=None, model_params=None):
        if random_seed is not None:
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.rng = np.random.default_rng(random_seed)
        self._reset_torch_rng()
        
        # Create a temporary env to get dimensions
        temp_env = ZoningLangEnv()  # TODO do this a different way
        default_model_params = {
            "num_input_tokens": temp_env.num_tokens,
            "max_seq_len": temp_env.max_length,
            "num_productions": temp_env.max_productions,
            "d_model": 256,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512
        }
        self.model_params = default_model_params | (model_params or {})
        model = ZoningLangModel(**self.model_params)
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        print("Neural network training params are", self.training_params)
        print("Neural network model params are", self.model_params)

        self.DEVICE = (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu") if device is None else device
        print(f"Neural network training will occur on device '{self.DEVICE}'")
        self.optimizer = None
        self.pad_token_int = temp_env.pad_token_int
    
    def _reset_torch_rng(self):
        seed = int(self.rng.integers(2**31-1))
        torch.manual_seed(seed)

    def _reshape_data(self, examples):
        # The usual case is that `examples` comes from AlphaZero and is a list of tuples
        # that will need to be reformatted.
        # Policy targets are (token_index, production_index) tuples

        states = torch.from_numpy(np.array([state for state, (_, _) in examples], dtype=np.int64))
        # Convert policy tuples to flat indices for CrossEntropyLoss
        # policy is (token_idx, production_idx)
        policies = torch.tensor([policy for _, (policy, _) in examples], dtype=torch.long)
        values = torch.from_numpy(np.array([value for _, (_, value) in examples], dtype=np.float32))
        dataset = torch.utils.data.TensorDataset(states, policies, values)
        return dataset
    
    def train(self, examples, val_dataset = None, needs_reshape=True, print_all_epochs=False):
        self._reset_torch_rng()
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
            for states, target_actions, target_values in train_loader:
                states = states.to(self.DEVICE)
                target_actions = target_actions.to(self.DEVICE)  # (batch, 2) - (token_idx, prod_idx)
                target_values = target_values.to(self.DEVICE)
                
                # Create padding mask: True for padding positions
                padding_mask = (states == self.pad_token_int)
                
                self.optimizer.zero_grad()
                policy_logits, predicted_values = model(states, src_key_padding_mask=padding_mask)
                
                # Policy loss: use the logits at the target token position
                # policy_logits is (batch, seq_len, num_productions)
                # We need to select logits at target_actions[:, 0] and classify as target_actions[:, 1]
                batch_indices = torch.arange(len(target_actions), device=self.DEVICE)
                selected_logits = policy_logits[batch_indices, target_actions[:, 0], :]  # (batch, num_productions)
                loss_policy = criterion_policy(selected_logits, target_actions[:, 1])
                
                loss_value = criterion_value(predicted_values.squeeze(), target_values)
                mean_max = policy_logits.softmax(dim=-1).max(dim=-1).values.mean()
                loss = value_weight * loss_value + policy_weight * loss_policy

                # Add L1 regularization
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                train_mini_losses.append(loss)
                train_loss += loss
                policy_loss += loss_policy.item()
                value_loss += loss_value.item()
                train_mean_max += mean_max.item()

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
        value_weight = tp["value_weight"]
        policy_weight = tp["policy_weight"]
        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        
        val_loss = 0.0
        val_mean_max = 0.0
        with torch.no_grad():
            for states, target_actions, target_values in val_loader:
                states = states.to(self.DEVICE)
                target_actions = target_actions.to(self.DEVICE)
                target_values = target_values.to(self.DEVICE)
                
                # Create padding mask
                padding_mask = (states == self.pad_token_int)
                
                policy_logits, predicted_values = model(states, src_key_padding_mask=padding_mask)
                
                # Policy loss
                batch_indices = torch.arange(len(target_actions), device=self.DEVICE)
                selected_logits = policy_logits[batch_indices, target_actions[:, 0], :]
                loss_policy = criterion_policy(selected_logits, target_actions[:, 1])
                
                loss_value = criterion_value(predicted_values.squeeze(), target_values)
                mean_max = policy_logits.softmax(dim=-1).max(dim=-1).values.mean()
                loss = value_weight * loss_value + policy_weight * loss_policy
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
        # state is already a flat array of token indices
        nn_input = torch.tensor(state, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        
        # Create padding mask
        padding_mask = (nn_input == self.pad_token_int)
        
        with torch.no_grad():
            policy_logits, value = self.model(nn_input, src_key_padding_mask=padding_mask)
            # policy_logits is (1, seq_len, num_productions)
            policy_prob = F.softmax(policy_logits, dim=-1)  # (1, seq_len, num_productions)
        
        # Return (seq_len, num_productions) and scalar value
        return policy_prob.squeeze(0).numpy(), value.numpy().squeeze()
    
    def push_multiprocessing(self):
        self._reset_torch_rng()
        super().push_multiprocessing()
        optimizer = self.optimizer  # the optimizer can't be sent to CPU, so we'll stash it in caller state so we don't try to copy it
        self.optimizer = None
        return optimizer

    def pop_multiprocessing(self, optimizer):
        super().pop_multiprocessing(optimizer)
        self.optimizer = optimizer
