"""
CartPole AlphaZero Implementation (Milton Version)
===================================================
Phase 1: Core Class Migration

This file contains the foundational classes for the CartPole AlphaZero experiment.
It is designed to be self-contained and runnable for verification purposes.

Reward Formulation (Sparse):
    r_t = 0                         for t < T (non-terminal steps)
    r_T = steps_survived / max_steps   at termination

    This normalization ensures:
    1. Value head targets are in [0, 1], preventing gradient explosion.
    2. Interpretability: r=0.8 means 80% of max survival.
    3. MCTS Q-value stability.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsai_experiments.general_az_1p.game import EnvGame
from nsai_experiments.general_az_1p.policy_value_net import TorchPolicyValueNet
from nsai_experiments.general_az_1p.utils import get_accelerator

# =============================================================================
# Sub-Phase 1.2: CumulativeRewardWrapper (Sparse Reward Implementation)
# =============================================================================
class CumulativeRewardWrapper(gym.Wrapper):
    """
    Wrapper that implements SPARSE reward for CartPole:
        - Reward is 0 at every intermediate step.
        - At termination, reward = (steps_survived / max_steps).
    
    Justification:
        AlphaZero's value function predicts cumulative future reward from a state.
        Sparse, episodic rewards align better with this framework than dense +1/step.
        Normalizing by max_steps keeps V(s) targets in [0, 1].
    """
    
    def __init__(self, env, max_steps=100):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = max_steps
    
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Enforce max_steps truncation
        if self.step_count >= self.max_steps:
            truncated = True
        
        # SPARSE REWARD LOGIC
        if terminated or truncated:
            # Final reward = fraction of max steps survived
            reward = self.step_count / self.max_steps
        else:
            reward = 0.0  # No intermediate reward
        
        # Invariant checks
        assert reward == 0.0 or (terminated or truncated), "Non-zero reward only at terminal state"
        assert 0.0 <= reward <= 1.0, f"Reward out of bounds: {reward}"
            
        return observation, reward, terminated, truncated, info


# =============================================================================
# Sub-Phase 1.3: CartPoleGame (Game Adapter)
# =============================================================================
class CartPoleGame(EnvGame):
    """
    Adapts CartPole-v1 (with CumulativeRewardWrapper) to the EnvGame interface.
    
    Justification:
        The generic AlphaZero Agent expects an EnvGame with:
        - get_action_mask() -> np.ndarray
        - step_wrapper(action)
        - Game state properties (obs, reward, terminated, truncated)
    """
    _ACTION_MASK = np.array([True, True])  # Left (0) and Right (1) are always valid

    def __init__(self, use_cumulative_reward_rescale=True, max_steps=100, **kwargs):
        env = gym.make("CartPole-v1", **kwargs)
        if use_cumulative_reward_rescale:
            env = CumulativeRewardWrapper(env, max_steps=max_steps)
        super().__init__(env)
        self.max_steps = max_steps
        
        # --- PHASE 1.4: Configuration Print Statement ---
        mode_str = "SPARSE (reward = steps/max_steps at end)" if use_cumulative_reward_rescale else "DENSE (native +1/step)"
        print(f"[Config] Reward Mode: {mode_str} | max_steps={max_steps}")
    
    def get_action_mask(self):
        return self._ACTION_MASK


# =============================================================================
# Sub-Phase 1.3: CartPoleModel (Neural Network Architecture)
# =============================================================================
class CartPoleModel(nn.Module):
    """
    Policy-Value network for CartPole with LayerNorm for gradient stability.
    
    Architecture:
        1. INPUT PROJECTION:
           - Linear(4 → hidden_size) → LayerNorm → ReLU
           - LayerNorm applied AFTER projection (not on raw input)
           - Rationale: Raw input has disparate units (meters vs radians)
        
        2. HIDDEN TOWER:
           - For each layer: Linear → LayerNorm → ReLU
           - LayerNorm stabilizes optimization by fixing second moment of activations
        
        3. POLICY HEAD:
           - Linear(hidden_size → 2) returning raw logits
           - Softmax applied externally (CrossEntropyLoss expects logits)
        
        4. VALUE HEAD:
           - Linear(hidden_size → 1) → Sigmoid
           - Sigmoid bounds V(s) ∈ [0, 1] to match normalized rewards
    
    Justification:
        - LayerNorm over BatchNorm: works with any batch size, no running stats
        - Sigmoid on value: prevents predicting impossible values outside [0,1]
        - No LN on raw input: preserves physical meaning of state dimensions
    """
    _INPUT_SIZE = 4  # CartPole observation space: [x, x_dot, theta, theta_dot]
    
    def __init__(self, n_hidden_layers=2, hidden_size=128):
        super().__init__()
        
        # 1. Input Embedding (LayerNorm AFTER projection, not on raw input)
        self.input_block = nn.Sequential(
            nn.Linear(self._INPUT_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # 2. Deep Body with LayerNorm
        layers = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        
        # 3. Policy Head (raw logits for CrossEntropyLoss)
        self.policy_head = nn.Linear(hidden_size, 2)  # 2 actions: Left, Right
        
        # 4. Value Head with Sigmoid → [0, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights with rigorous strategy
        self._init_weights()
    
    def _init_weights(self):
        """
        Rigorous Initialization Strategy for Policy-Value Networks.
        
        ═══════════════════════════════════════════════════════════════════════════
        MATHEMATICAL FOUNDATION
        ═══════════════════════════════════════════════════════════════════════════
        
        1. VARIANCE PRESERVATION PRINCIPLE
        ───────────────────────────────────────────────────────────────────────────
        For stable gradient flow, we want:
            Var(activation_L) ≈ Var(activation_0)    (forward pass)
            Var(∂L/∂W_L) ≈ Var(∂L/∂W_0)              (backward pass)
        
        Different nonlinearities require different variance corrections.
        
        2. KAIMING INITIALIZATION (for ReLU layers)
        ───────────────────────────────────────────────────────────────────────────
        ReLU kills ~50% of activations (negative values → 0), halving variance:
            Var(ReLU(z)) = 0.5 · Var(z)
        
        Kaiming compensates by doubling weight variance:
            W ~ N(0, σ²)  where σ² = 2 / fan_out
        
        This ensures:
            Var(h_l) = Var(h_{l-1})  after each ReLU layer
        
        3. XAVIER/GLOROT INITIALIZATION (for heads)
        ───────────────────────────────────────────────────────────────────────────
        For linear or symmetric activations (tanh, sigmoid), we use Xavier:
            W ~ U[-a, a]  where a = √(6 / (fan_in + fan_out))
        
        This maintains variance for both forward AND backward passes.
        
        4. WHY XAVIER FOR HEADS (CRITICAL DESIGN DECISION)
        ───────────────────────────────────────────────────────────────────────────
        
        POLICY HEAD (→ Softmax):
        ─────────────────────────
        Let z = Wh be the logits, softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
        
        With standard Xavier (gain=1):
            Var(z) = fan_in · Var(W) · Var(h) ≈ 2.0 (for fan_in=128)
            std(z) ≈ 1.41
            
        Typical logits: z = [0.3, -0.2] → softmax = [0.62, 0.38]  (BIASED!)
        
        With Xavier (gain=0.1):
            Var(z) ≈ 0.02, std(z) ≈ 0.14
            
        Typical logits: z = [0.05, -0.03] → softmax ≈ [0.52, 0.48]  (NEAR MAX ENTROPY ✓)
        
        IMPLICATION FOR MCTS:
        ─────────────────────
        - Biased prior → MCTS over-exploits one random action → poor exploration
        - Max Entropy prior → MCTS explores all actions fairly → better learning
        
        VALUE HEAD (→ Sigmoid):
        ────────────────────────
        Sigmoid has useful gradients only in [-3, 3]:
            σ(-3) ≈ 0.05,  σ(3) ≈ 0.95
            σ'(x) → 0  as |x| → ∞  (gradient saturation)
        
        With Xavier init:
            std(z) ≈ 1.0 → 99.7% of values in [-3, 3] ✓
            
        With Kaiming init:
            std(z) ≈ 1.41 → ~95% in [-3, 3], risk of saturation ✗
        
        5. LAYERNORM INITIALIZATION
        ───────────────────────────────────────────────────────────────────────────
        LayerNorm: ŷ = γ · (x - μ) / σ + β
        
        Initialize as identity transform: γ = 1, β = 0
        This ensures LayerNorm starts as a pure normalizer, letting the network
        learn optimal scale/shift during training.
        
        ═══════════════════════════════════════════════════════════════════════════
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 1. POLICY HEAD: High constraint (Near Max Entropy)
                if "policy_head" in name:
                    # Gain=0.1 forces logits near 0, ensuring near-uniform probabilities
                    # Note: gain=0.01 is theoretically optimal but causes NaN on MPS
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    nn.init.constant_(module.bias, 0)
                
                # 2. VALUE HEAD: Linear Regime constraint
                elif "value_head" in name:
                    # Gain=1.0 keeps output variance inside Sigmoid's linear region
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    nn.init.constant_(module.bias, 0)
                
                # 3. BODY LAYERS: ReLU constraint
                else:
                    # Gain=sqrt(2) compensates for ReLU killing half the variance
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            
            # 4. LayerNorm/BatchNorm: Identity Init
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.body(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        
        return policy_logits, value

    def print_architecture(self):
        """Print network architecture with initialization strategy."""
        total_params = sum(p.numel() for p in self.parameters())
        
        print("\n" + "=" * 70)
        print("NEURAL NETWORK ARCHITECTURE & INITIALIZATION")
        print("=" * 70)
        print(f"Model: CartPoleModel | Parameters: {total_params:,}")
        print("-" * 70)
        
        # Layer structure with dimensions
        print("LAYER STRUCTURE:")
        print(f"  Input:  Linear(4 → {self.input_block[0].out_features}) → LN → ReLU")
        for i in range(0, len(self.body), 3):
            if i < len(self.body):
                dim = self.body[i].out_features
                print(f"  Body:   Linear({dim} → {dim}) → LN → ReLU")
        print(f"  Policy: Linear({self.policy_head.in_features} → 2) → [Softmax]")
        print(f"  Value:  Linear({self.value_head[0].in_features} → 1) → Sigmoid")
        print()
        
        # Initialization strategy with mathematical rationale
        print("INITIALIZATION STRATEGY:")
        print("-" * 70)
        print("  BODY LAYERS (Input + Hidden) → Kaiming Normal (fan_out)")
        print("    Rationale: ReLU kills ~50% of activations → Var halved")
        print("    Kaiming compensates: W ~ N(0, 2/fan_out) → preserves Var(h)")
        print()
        print("  POLICY HEAD → Xavier Uniform (gain=0.1)")
        print("    Rationale: Softmax gradient vanishes for large |logits|")
        print("    Gain=0.1 forces Var(logit) → 0 → softmax ≈ [0.52, 0.48]")
        print("    MCTS benefit: Near Max Entropy prior enables fair exploration")
        print()
        print("  VALUE HEAD → Xavier Uniform (gain=1.0)")
        print("    Rationale: Sigmoid saturates outside [-3, 3]")
        print("    Xavier keeps std(z) ≈ 1 → 99.7% in linear regime")
        print()
        print("  LAYERNORM → Identity (γ=1, β=0)")
        print("    Rationale: Start as pure normalizer, learn scale/shift")
        print("=" * 70 + "\n")


# =============================================================================
# Sub-Phase 1.3: CartPolePolicyValueNet (Training Logic)
# =============================================================================
class CartPolePolicyValueNet(TorchPolicyValueNet):
    """
    Handles training and inference for the CartPole Policy-Value network.
    
    Loss:
        L = L_value + policy_weight * L_policy
        L_value = MSE(V_pred, V_target)
        L_policy = CrossEntropy(pi_pred, pi_target)
    
    Justification:
        Standard AlphaZero loss. policy_weight balances the two loss terms
        (typically set >1 because CE loss is often larger than MSE).
    """
    save_file_name = "cartpole_checkpoint.pt"
    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "policy_weight": 1.0,
    }

    def __init__(self, random_seed=None, n_hidden_layers=2, hidden_size=128, training_params={}, device=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        model = CartPoleModel(n_hidden_layers=n_hidden_layers, hidden_size=hidden_size)
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        
        self.DEVICE = get_accelerator() if device is None else device
        print(f"[Config] Neural network training device: '{self.DEVICE}'")

        
    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]

        model = model.to(self.DEVICE)
        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=tp["learning_rate"], weight_decay=tp["weight_decay"])

        if needs_reshape:
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
            model.train()
            train_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0
            for inputs, targets_policy, targets_value in train_loader:
                inputs, targets_value, targets_policy = inputs.to(self.DEVICE), targets_value.to(self.DEVICE), targets_policy.to(self.DEVICE)
                optimizer.zero_grad()
                assert len(inputs.shape) == 2, f"Expected input shape to be 2D, got {inputs.shape}"
                assert inputs.shape[1] == CartPoleModel._INPUT_SIZE, f"Expected input size {CartPoleModel._INPUT_SIZE}, got {inputs.shape[1]}"
                assert len(targets_value.shape) == 1
                assert targets_policy.shape[1] == CartPoleGame._ACTION_MASK.shape[0], f"Expected policy answers size {CartPoleGame._ACTION_MASK.shape[0]}, got {targets_policy.shape[1]}"
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
            
            # Store detailed metrics (dict format for TrackingAgent compatibility)
            epoch_metrics = {
                'total': avg_train_loss,
                'value': avg_value_loss,
                'policy': avg_policy_loss,
                'weighted_policy': policy_weight * avg_policy_loss
            }
            train_losses.append(epoch_metrics)

            if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
                print(f"Epoch {epoch+1}/{tp['epochs']}, Train Loss: {avg_train_loss:.4f} (value: {avg_value_loss:.4f}, policy: {avg_policy_loss:.4f}, weighted policy: {epoch_metrics['weighted_policy']:.4f})")

        return model, train_mini_losses, train_losses
    
    def predict(self, state):
        self.model.cpu()
        nn_input = torch.tensor(state).float().reshape(1, -1)  # Ensure float32
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)
        
        policy_prob = policy_prob.numpy()
        policy_prob = policy_prob.squeeze(0)
        assert policy_prob.shape == (2,)

        value = value.numpy()
        value = value.squeeze(0)
        assert value.shape == ()

        return policy_prob, value

# =============================================================================
# PHASE 2: Infrastructure Classes
# =============================================================================

# --- Sub-Phase 2.1: Tee Class (Dual Logging) ---
class Tee:
    """
    A file-like object that writes to multiple streams simultaneously.
    
    Justification:
        During training, we want console output for real-time monitoring AND
        a log file for post-hoc analysis. This class duplicates writes to both.
    
    Usage:
        log_file = open("log.txt", "w")
        sys.stdout = Tee(sys.stdout, log_file)
        print("This goes to both console and log.txt")
    """
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate write (important for crash recovery)
    
    def flush(self):
        for f in self.files:
            f.flush()


# --- Sub-Phase 2.2: TrackingAgent (Agent with History Tracking) ---
# NOTE: The base Agent class already includes `self.history` tracking
# and returns detailed stats from `pit()`. We import it directly.
# If CartPole-specific customizations are needed, we can subclass here.

from nsai_experiments.general_az_1p.agent import Agent
from nsai_experiments.general_az_1p.mcts import MCTS, entab
from copy import deepcopy
import time
import warnings

# For CartPole, the base Agent provides all needed functionality.
# We alias it for clarity and potential future extension.
TrackingAgent = Agent

# =============================================================================
# PHASE 3: CLI & Interactive Mode
# =============================================================================
import argparse

# --- Sub-Phase 3.1: Default Configuration ---
DEFAULT_CONFIG = {
    "max_steps": 100,           # Episode length
    "n_iters": 10,              # Total training iterations
    "n_games_per_train": 50,    # Games per training batch
    "n_games_per_eval": 20,     # Games per evaluation round
    "mcts_sims": 30,            # MCTS simulations per move
    "c_expl": 1.0,              # Exploration constant (PUCT)
    "epochs": 10,               # Training epochs per iteration
    "lr": 1e-3,                 # Learning rate
    "policy_weight": 2.0,       # Policy loss weight
    "threshold": 0.55,          # Eval win rate threshold
    "use_gating": True,         # Enable gating (pit new vs old)
    "random_seed": 47,          # Base random seed
}


def parse_args():
    """
    Sub-Phase 3.2: Parse command-line arguments.
    
    Justification:
        CLI args allow running experiments with different hyperparameters
        without modifying source code. Essential for hyperparameter sweeps.
    """
    parser = argparse.ArgumentParser(
        description="CartPole AlphaZero Experiment (Milton Implementation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"],
                        help="Maximum steps per episode")
    
    # Training loop
    parser.add_argument("--iters", type=int, default=DEFAULT_CONFIG["n_iters"],
                        help="Total training iterations")
    parser.add_argument("--games_train", type=int, default=DEFAULT_CONFIG["n_games_per_train"],
                        help="Games per training batch")
    parser.add_argument("--games_eval", type=int, default=DEFAULT_CONFIG["n_games_per_eval"],
                        help="Games per evaluation round")
    
    # MCTS
    parser.add_argument("--mcts_sims", type=int, default=DEFAULT_CONFIG["mcts_sims"],
                        help="MCTS simulations per move")
    parser.add_argument("--c_expl", type=float, default=DEFAULT_CONFIG["c_expl"],
                        help="Exploration constant (PUCT c)")
    
    # Neural network training
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Training epochs per iteration")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"],
                        help="Learning rate")
    parser.add_argument("--policy_weight", type=float, default=DEFAULT_CONFIG["policy_weight"],
                        help="Policy loss weight relative to value loss")
    
    # Gating
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"],
                        help="Evaluation win rate threshold to keep new network")
    parser.add_argument("--nogating", action="store_true",
                        help="Disable gating (always accept new network)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["random_seed"],
                        help="Base random seed")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode to configure hyperparameters")
    parser.add_argument("--test", action="store_true",
                        help="Run verification tests instead of training")
    
    return parser.parse_args()


def interactive_config(config):
    """
    Sub-Phase 3.3: Interactive configuration wizard.
    
    Justification:
        For exploratory experiments, interactive prompts are more user-friendly
        than remembering dozens of CLI flags.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE CONFIGURATION")
    print("Press Enter to accept [default] values.")
    print("=" * 60)
    
    try:
        val = input(f"Max steps per episode [{config['max_steps']}]: ").strip()
        if val: config["max_steps"] = int(val)
        
        val = input(f"Training iterations [{config['n_iters']}]: ").strip()
        if val: config["n_iters"] = int(val)
        
        val = input(f"Games per training batch [{config['n_games_per_train']}]: ").strip()
        if val: config["n_games_per_train"] = int(val)
        
        val = input(f"Games per evaluation [{config['n_games_per_eval']}]: ").strip()
        if val: config["n_games_per_eval"] = int(val)
        
        val = input(f"MCTS simulations [{config['mcts_sims']}]: ").strip()
        if val: config["mcts_sims"] = int(val)
        
        val = input(f"Exploration constant c_expl [{config['c_expl']}]: ").strip()
        if val: config["c_expl"] = float(val)
        
        val = input(f"Training epochs [{config['epochs']}]: ").strip()
        if val: config["epochs"] = int(val)
        
        val = input(f"Learning rate [{config['lr']}]: ").strip()
        if val: config["lr"] = float(val)
        
        val = input(f"Policy weight [{config['policy_weight']}]: ").strip()
        if val: config["policy_weight"] = float(val)
        
        val = input(f"Eval threshold [{config['threshold']}]: ").strip()
        if val: config["threshold"] = float(val)
        
        current_gating = "yes" if config["use_gating"] else "no"
        val = input(f"Enable gating? (yes/no) [{current_gating}]: ").strip().lower()
        if val in ["no", "n", "false"]: config["use_gating"] = False
        elif val in ["yes", "y", "true"]: config["use_gating"] = True
        
    except ValueError as e:
        print(f"Invalid input: {e}. Using defaults for remaining fields.")
    
    print("\n" + "=" * 60)
    print("Configuration Complete")
    print("=" * 60)
    
    return config


def build_config_from_args(args):
    """Convert parsed args to config dictionary."""
    return {
        "max_steps": args.max_steps,
        "n_iters": args.iters,
        "n_games_per_train": args.games_train,
        "n_games_per_eval": args.games_eval,
        "mcts_sims": args.mcts_sims,
        "c_expl": args.c_expl,
        "epochs": args.epochs,
        "lr": args.lr,
        "policy_weight": args.policy_weight,
        "threshold": args.threshold,
        "use_gating": not args.nogating,
        "random_seed": args.seed,
    }


def print_config(config):
    """Print configuration in a readable format."""
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


# =============================================================================
# VERIFICATION TESTS (Phases 1, 2, 3)
# =============================================================================
def run_verification_tests():
    """Run all verification tests for Phases 1, 2, and 3."""
    import sys
    import os
    import tempfile
    
    print("=" * 60)
    print("PHASE 1 VERIFICATION: Core Class Migration")
    print("=" * 60)
    
    # --- Test 1: CartPoleGame instantiation and config print ---
    print("\n[Test 1] Instantiating CartPoleGame (expect config print)...")
    game = CartPoleGame(max_steps=100)
    print(f"  -> Game created. Observation space: {game.env.observation_space}")
    print(f"  -> Action mask: {game.get_action_mask()}")
    
    # --- Test 2: Reset and step to verify wrapper ---
    print("\n[Test 2] Running a short episode to verify CumulativeRewardWrapper...")
    game.reset_wrapper()
    total_reward = 0.0
    steps = 0
    for i in range(50):
        action = game.env.action_space.sample()
        game.step_wrapper(action)
        total_reward += game.reward
        steps += 1
        if game.terminated or game.truncated:
            break
    
    print(f"  -> Episode ended after {steps} steps.")
    print(f"  -> Final reward: {game.reward:.4f} (expected: {steps}/100 = {steps/100:.4f})")
    print(f"  -> Total accumulated reward: {total_reward:.4f}")
    assert abs(total_reward - steps / 100) < 1e-6, "Reward mismatch!"
    print("  -> PASS: Sparse reward verified.")
    
    # --- Test 3: Model forward pass ---
    print("\n[Test 3] Testing CartPoleModel forward pass...")
    model = CartPoleModel()
    dummy_input = torch.randn(1, 4)
    policy_logits, value = model(dummy_input)
    print(f"  -> Policy logits shape: {policy_logits.shape} (expected: [1, 2])")
    print(f"  -> Value shape: {value.shape} (expected: [1] for batch of 1)")
    assert policy_logits.shape == (1, 2), "Policy shape mismatch!"
    assert value.shape == (1,), "Value shape mismatch!"
    print("  -> PASS: Model forward pass verified.")
    
    # --- Test 4: PolicyValueNet instantiation ---
    print("\n[Test 4] Instantiating CartPolePolicyValueNet...")
    net = CartPolePolicyValueNet(random_seed=42, device="cpu")
    policy_prob, value = net.predict(np.array([0.0, 0.0, 0.0, 0.0]))
    print(f"  -> Predict output: policy={policy_prob}, value={value:.4f}")
    assert policy_prob.shape == (2,), "Predict policy shape mismatch!"
    print("  -> PASS: PolicyValueNet instantiation verified.")
    
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION COMPLETE: All tests passed.")
    print("=" * 60)
    
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2 VERIFICATION: Infrastructure Classes")
    print("=" * 60)
    
    # --- Test 5: Tee class for dual logging ---
    print("\n[Test 5] Testing Tee class for dual logging...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tee = Tee(sys.stdout, tmp_file)
        original_stdout = sys.stdout
        sys.stdout = tee
        
        test_message = "  -> This message should appear in both console and file."
        print(test_message)
        
        sys.stdout = original_stdout
    
    with open(tmp_path, 'r') as f:
        file_contents = f.read()
    os.unlink(tmp_path)
    
    assert test_message in file_contents, "Tee did not write to file!"
    print("  -> PASS: Tee class verified (message written to file).")
    
    # --- Test 6: TrackingAgent instantiation and history structure ---
    print("\n[Test 6] Testing TrackingAgent (base Agent) instantiation...")
    game2 = CartPoleGame(max_steps=100)
    net2 = CartPolePolicyValueNet(random_seed=42, device="cpu")
    agent = TrackingAgent(
        game2, net2,
        n_games_per_train=5,
        n_games_per_eval=3,
        n_procs=-1,
        random_seeds={"mcts": 1, "train": 2, "eval": 3}
    )
    
    print(f"  -> Agent created.")
    print(f"  -> History keys: {list(agent.history.keys())}")
    expected_keys = {'iteration', 'reward_mean', 'reward_std', 'loss_policy', 'loss_value', 'game_length'}
    assert set(agent.history.keys()) == expected_keys, f"History keys mismatch!"
    print("  -> PASS: TrackingAgent history structure verified.")
    
    # --- Test 7: Single play_and_train iteration ---
    print("\n[Test 7] Running one play_and_train iteration (may take ~10s)...")
    print("  -> This tests: game generation, training, pit() evaluation, history logging.")
    agent.play_and_train()
    
    print(f"  -> History after 1 iteration:")
    print(f"     iteration: {agent.history['iteration']}")
    print(f"     reward_mean: {agent.history['reward_mean']}")
    print(f"     loss_policy: {agent.history['loss_policy']}")
    print(f"     loss_value: {agent.history['loss_value']}")
    print(f"     game_length: {agent.history['game_length']}")
    
    assert len(agent.history['iteration']) == 1, "History should have 1 iteration!"
    assert agent.history['iteration'][0] == 1, "First iteration should be 1!"
    assert not np.isnan(agent.history['reward_mean'][0]), "reward_mean should not be NaN!"
    print("  -> PASS: play_and_train() and history logging verified.")
    
    print("\n" + "=" * 60)
    print("PHASE 2 VERIFICATION COMPLETE: All tests passed.")
    print("=" * 60)
    
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION: CLI & Interactive Mode")
    print("=" * 60)
    
    # --- Test 8: parse_args() with custom values ---
    print("\n[Test 8] Testing parse_args() with custom CLI values...")
    import sys
    original_argv = sys.argv
    sys.argv = ["test", "--iters", "5", "--lr", "0.01", "--nogating", "--seed", "123"]
    
    args = parse_args()
    print(f"  -> Parsed: iters={args.iters}, lr={args.lr}, nogating={args.nogating}, seed={args.seed}")
    assert args.iters == 5, "iters not parsed correctly!"
    assert args.lr == 0.01, "lr not parsed correctly!"
    assert args.nogating == True, "nogating not parsed correctly!"
    assert args.seed == 123, "seed not parsed correctly!"
    print("  -> PASS: parse_args() with custom values verified.")
    
    # --- Test 9: build_config_from_args() ---
    print("\n[Test 9] Testing build_config_from_args()...")
    config = build_config_from_args(args)
    print(f"  -> Config: n_iters={config['n_iters']}, lr={config['lr']}, use_gating={config['use_gating']}")
    assert config["n_iters"] == 5, "n_iters not built correctly!"
    assert config["lr"] == 0.01, "lr not built correctly!"
    assert config["use_gating"] == False, "use_gating should be False when --nogating!"
    print("  -> PASS: build_config_from_args() verified.")
    
    sys.argv = original_argv  # Restore
    
    # --- Test 10: print_config() ---
    print("\n[Test 10] Testing print_config()...")
    print_config(config)
    print("  -> PASS: print_config() executed without error.")
    
    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION COMPLETE: All tests passed.")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS PASSED (Phases 1, 2, 3)")
    print("=" * 60)


# =============================================================================
# PHASE 4: Execution Loop & Logging
# =============================================================================
import os
import sys
from datetime import datetime
from pathlib import Path

def create_run_directory(base_dir="runs"):
    """
    Sub-Phase 4.1: Create timestamped run directory.
    
    Justification:
        Each experiment run should have its own directory for logs, checkpoints,
        and plots. Timestamp ensures unique naming and easy chronological sorting.
    
    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}_cartpole"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def plot_training_metrics(history, save_path=None):
    """
    Sub-Phase 4.2: Plot training metrics.
    
    Justification:
        Visualization is essential for diagnosing training issues and comparing runs.
        Plots reward, losses, and game length over iterations.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[Warning] matplotlib/pandas not installed. Skipping plot generation.")
        return
    
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("CartPole AlphaZero Training Metrics", fontsize=14, fontweight='bold')
    
    # Plot 1: Reward mean with std band
    ax1 = axes[0, 0]
    ax1.plot(df['iteration'], df['reward_mean'], 'b-', linewidth=2, label='Mean Reward')
    if 'reward_std' in df.columns:
        ax1.fill_between(df['iteration'], 
                         df['reward_mean'] - df['reward_std'],
                         df['reward_mean'] + df['reward_std'],
                         alpha=0.3, color='blue', label='±1 Std')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward (MCTS Evaluation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)  # Reward is normalized to [0, 1]
    
    # Plot 2: Policy and Value Loss
    ax2 = axes[0, 1]
    ax2.plot(df['iteration'], df['loss_policy'], 'r-', linewidth=2, label='Policy Loss')
    ax2.plot(df['iteration'], df['loss_value'], 'g-', linewidth=2, label='Value Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for losses
    
    # Plot 3: Game Length
    ax3 = axes[1, 0]
    ax3.plot(df['iteration'], df['game_length'], 'm-', linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Avg Game Length')
    ax3.set_title('Average Episode Length')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined (normalized)
    ax4 = axes[1, 1]
    # Normalize each metric to [0, 1] for comparison
    reward_norm = df['reward_mean'] / df['reward_mean'].max() if df['reward_mean'].max() > 0 else df['reward_mean']
    length_norm = df['game_length'] / df['game_length'].max() if df['game_length'].max() > 0 else df['game_length']
    policy_norm = 1 - (df['loss_policy'] / df['loss_policy'].max()) if df['loss_policy'].max() > 0 else df['loss_policy']
    
    ax4.plot(df['iteration'], reward_norm, 'b-', linewidth=2, label='Reward (norm)')
    ax4.plot(df['iteration'], length_norm, 'm-', linewidth=2, label='Game Length (norm)')
    ax4.plot(df['iteration'], policy_norm, 'r--', linewidth=2, label='1 - Policy Loss (norm)')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Combined Metrics (Normalized)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved to {save_path}")
    
    plt.close(fig)  # Close to free memory


def run_training(config):
    """
    Sub-Phase 4.3: Main training execution loop.
    
    Justification:
        Encapsulates the full training pipeline: setup, loop, logging, plotting.
        Uses Tee for dual stdout/file logging.
    """
    # Create run directory
    run_dir = create_run_directory()
    log_file_path = run_dir / "log.txt"
    plot_path = run_dir / "metrics.png"
    
    print(f"[Run] Output directory: {run_dir}")
    
    # Setup Tee logging
    log_file = open(log_file_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    
    try:
        # Print config to log
        print_config(config)
        print(f"[Run] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Run] Log file: {log_file_path}")
        
        # Create game, network, and agent
        game = CartPoleGame(max_steps=config["max_steps"])
        net = CartPolePolicyValueNet(
            random_seed=config["random_seed"],
            training_params={
                "epochs": config["epochs"],
                "learning_rate": config["lr"],
                "policy_weight": config["policy_weight"],
            }
        )
        
        # Print neural network architecture
        net.model.print_architecture()
        
        agent = TrackingAgent(
            game, net,
            n_games_per_train=config["n_games_per_train"],
            n_games_per_eval=config["n_games_per_eval"],
            threshold_to_keep=config["threshold"],
            use_gating=config["use_gating"],
            mcts_params={"n_simulations": config["mcts_sims"], "c_exploration": config["c_expl"]},
            n_procs=-1,  # Use single process for stability
            random_seeds={
                "mcts": config["random_seed"],
                "train": config["random_seed"] + 1,
                "eval": config["random_seed"] + 2,
            }
        )
        
        # Training loop
        print(f"\n[Training] Starting {config['n_iters']} iterations...")
        print("=" * 60)
        
        for i in range(config["n_iters"]):
            print(f"\n--- Iteration {i+1}/{config['n_iters']} ---")
            agent.play_and_train()
            
            # Save intermediate plot every 5 iterations
            if (i + 1) % 5 == 0 or (i + 1) == config["n_iters"]:
                plot_training_metrics(agent.history, save_path=plot_path)
        
        # Final summary
        print("\n" + "=" * 60)
        print("[Training] COMPLETE")
        print("=" * 60)
        print(f"  Final reward mean: {agent.history['reward_mean'][-1]:.4f}")
        print(f"  Final policy loss: {agent.history['loss_policy'][-1]:.4f}")
        print(f"  Final value loss: {agent.history['loss_value'][-1]:.6f}")
        print(f"  Final avg game length: {agent.history['game_length'][-1]:.1f}")
        print(f"  Output directory: {run_dir}")
        print(f"  Log file: {log_file_path}")
        print(f"  Metrics plot: {plot_path}")
        
        return agent.history, run_dir
        
    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def run_phase4_verification():
    """
    Phase 4 verification tests.
    Tests: run directory creation, Tee logging during training, metric plotting.
    """
    import tempfile
    import shutil
    
    print("=" * 60)
    print("PHASE 4 VERIFICATION: Execution Loop & Logging")
    print("=" * 60)
    
    # --- Test 11: create_run_directory() ---
    print("\n[Test 11] Testing create_run_directory()...")
    with tempfile.TemporaryDirectory() as tmp_base:
        run_dir = create_run_directory(base_dir=tmp_base)
        print(f"  -> Created: {run_dir}")
        assert run_dir.exists(), "Run directory was not created!"
        assert "run_" in run_dir.name, "Run directory naming format incorrect!"
        print("  -> PASS: create_run_directory() verified.")
    
    # --- Test 12: plot_training_metrics() ---
    print("\n[Test 12] Testing plot_training_metrics()...")
    mock_history = {
        'iteration': [1, 2, 3],
        'reward_mean': [0.1, 0.3, 0.5],
        'reward_std': [0.05, 0.08, 0.1],
        'loss_policy': [0.7, 0.5, 0.3],
        'loss_value': [0.02, 0.01, 0.005],
        'game_length': [10, 20, 30],
    }
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_plot:
        tmp_plot_path = tmp_plot.name
    
    plot_training_metrics(mock_history, save_path=tmp_plot_path)
    assert os.path.exists(tmp_plot_path), "Plot file was not created!"
    assert os.path.getsize(tmp_plot_path) > 1000, "Plot file seems too small!"
    os.unlink(tmp_plot_path)
    print("  -> PASS: plot_training_metrics() verified.")
    
    # --- Test 13: Short training run (2 iterations) ---
    print("\n[Test 13] Running mini training (2 iterations, ~30s)...")
    print("  -> This tests: run_training(), Tee logging, Agent training, plotting.")
    
    mini_config = {
        "max_steps": 50,
        "n_iters": 2,
        "n_games_per_train": 3,
        "n_games_per_eval": 2,
        "mcts_sims": 10,
        "c_expl": 1.0,
        "epochs": 3,
        "lr": 0.001,
        "policy_weight": 2.0,
        "threshold": 0.55,
        "use_gating": False,  # Disable gating for speed
        "random_seed": 42,
    }
    
    # Run training
    history, run_dir = run_training(mini_config)
    
    # Verify outputs
    assert len(history['iteration']) == 2, "Should have 2 iterations in history!"
    assert (run_dir / "log.txt").exists(), "Log file should exist!"
    assert (run_dir / "metrics.png").exists(), "Metrics plot should exist!"
    
    # Check log file content
    with open(run_dir / "log.txt", 'r') as f:
        log_content = f.read()
    assert "Training" in log_content, "Log should contain training output!"
    assert "COMPLETE" in log_content, "Log should contain completion message!"
    
    print(f"  -> Output directory: {run_dir}")
    print(f"  -> Log file size: {(run_dir / 'log.txt').stat().st_size} bytes")
    print(f"  -> Plot file size: {(run_dir / 'metrics.png').stat().st_size} bytes")
    print("  -> PASS: run_training() mini test verified.")
    
    print("\n" + "=" * 60)
    print("PHASE 4 VERIFICATION COMPLETE: All tests passed.")
    print("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    
    if args.test:
        # Run all verification tests (Phases 1-4)
        run_verification_tests()  # Phases 1, 2, 3
        print("\n")
        run_phase4_verification()  # Phase 4
        
        print("\n" + "=" * 60)
        print("ALL VERIFICATION TESTS PASSED (Phases 1, 2, 3, 4)")
        print("=" * 60)
    else:
        # Build config from args
        config = build_config_from_args(args)
        
        # Interactive mode override
        if args.interactive:
            config = interactive_config(config)
        
        # Print config
        print_config(config)
        
        # Run training
        run_training(config)

