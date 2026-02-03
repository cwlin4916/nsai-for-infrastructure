"""
NSAI Meta-Ablation: CartPole Architecture Search
=================================================
This script runs 3 parallel experiments to isolate the impact of LayerNorm and Sigmoid.
It uses the existing 'CartPoleGame' and 'TrackingAgent' infrastructure but
dynamically builds Neural Networks with configurable architecture.

Conditions:
1. Simple:     No LayerNorm, No Sigmoid (Baseline like File B)
2. LN_Linear:  LayerNorm Body + Linear Value (Isolate LN effect)
3. LN_Sigmoid: LayerNorm Body + Sigmoid Value (Current Milton version)

Usage:
    cd src/nsai_experiments/general_az_1p/cartpole
    PYTHONPATH=../../.. python meta_ablation.py --iters 8
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pathlib import Path

# Import existing infrastructure
from cartpole_az_impl_milton import CartPoleGame, TrackingAgent, Tee
from nsai_experiments.general_az_1p.policy_value_net import TorchPolicyValueNet
from nsai_experiments.general_az_1p.utils import get_accelerator


# =============================================================================
# CONFIGURATION LOGGING
# =============================================================================
DEFAULT_CONFIG = {
    "iters": 8,
    "games_train": 30,
    "games_eval": 20,
    "mcts_sims": 20,
    "c_expl": 1.0,
    "max_steps": 100,
    "seed": 42,
    # Network defaults
    "hidden_size": 128,
    "n_body_layers": 3,
    # Training defaults
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "policy_weight": 2.0,
}

def print_experiment_config(config, conditions):
    """Print comprehensive experiment configuration at startup."""
    print("\n" + "=" * 70)
    print("META-ABLATION EXPERIMENT CONFIGURATION")
    print("=" * 70)
    
    print("\n[EXPERIMENT SETTINGS]")
    print(f"  Iterations per condition:  {config['iters']}")
    print(f"  Random seed:               {config['seed']}")
    print(f"  Output directory:          {config.get('out_dir', 'TBD')}")
    
    print("\n[ENVIRONMENT]")
    print(f"  Game:                      CartPole-v1")
    print(f"  Max steps per episode:     {config['max_steps']}")
    print(f"  Reward mode:               SPARSE (normalized [0,1])")
    
    print("\n[MCTS SETTINGS]")
    print(f"  Simulations per move:      {config['mcts_sims']}")
    print(f"  Exploration constant (c):  {config['c_expl']}")
    
    print("\n[DATA COLLECTION]")
    print(f"  Games per training batch:  {config['games_train']}")
    print(f"  Games per evaluation:      {config['games_eval']}")
    print(f"  Gating:                    DISABLED (for clean comparison)")
    
    print("\n[NETWORK TRAINING]")
    print(f"  Epochs per iteration:      {DEFAULT_CONFIG['epochs']}")
    print(f"  Batch size:                {DEFAULT_CONFIG['batch_size']}")
    print(f"  Learning rate:             {DEFAULT_CONFIG['learning_rate']}")
    print(f"  Weight decay (L2 reg):     {DEFAULT_CONFIG['weight_decay']}")
    print(f"  Policy loss weight:        {DEFAULT_CONFIG['policy_weight']}")
    
    print("\n[CONDITIONS TO TEST]")
    for i, (name, use_ln, use_sig) in enumerate(conditions):
        arch = []
        if use_ln:
            arch.append("LayerNorm")
        else:
            arch.append("No LN")
        if use_sig:
            arch.append("Sigmoid")
        else:
            arch.append("Linear")
        print(f"  {chr(65+i)}. {name}")
        print(f"      Body: {arch[0]} | Value Head: {arch[1]}")
    
    print("\n" + "=" * 70 + "\n")


def log_experiment_metadata(out_dir, config, conditions, results):
    """Save comprehensive metadata to JSON file."""
    import platform
    
    metadata = {
        "experiment": "meta_ablation",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(get_accelerator()),
        },
        "config": {
            "iterations": config["iters"],
            "seed": config["seed"],
            "max_steps": config["max_steps"],
            "mcts_sims": config["mcts_sims"],
            "c_exploration": config["c_expl"],
            "games_per_train": config["games_train"],
            "games_per_eval": config["games_eval"],
        },
        "training": {
            "epochs": DEFAULT_CONFIG["epochs"],
            "batch_size": DEFAULT_CONFIG["batch_size"],
            "learning_rate": DEFAULT_CONFIG["learning_rate"],
            "weight_decay": DEFAULT_CONFIG["weight_decay"],
            "policy_weight": DEFAULT_CONFIG["policy_weight"],
        },
        "conditions": [
            {
                "name": name,
                "use_layernorm": use_ln,
                "use_sigmoid": use_sig,
            }
            for name, use_ln, use_sig in conditions
        ],
        "results_summary": [
            {
                "name": r["name"],
                "final_reward": r["final_reward"],
                "rewards": r["rewards"],
                "value_losses": r["value_losses"],
                "policy_losses": r["policy_losses"],
            }
            for r in results
        ],
    }
    
    metadata_path = out_dir / "experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[Metadata] Saved to {metadata_path}")
    return metadata


# =============================================================================
# DYNAMIC MODEL BUILDER
# =============================================================================
class DynamicCartPoleNet(nn.Module):
    """
    Configurable CartPole network with toggleable LayerNorm and Sigmoid.
    
    Args:
        use_layernorm: If True, add LayerNorm after each Linear layer
        use_sigmoid: If True, use Sigmoid on value head (bounded [0,1])
    """
    def __init__(self, use_layernorm=False, use_sigmoid=False, hidden_size=128):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.use_sigmoid = use_sigmoid
        input_size = 4  # CartPole state dim
        
        # Build Body
        layers = []
        # Input Layer
        layers.append(nn.Linear(input_size, hidden_size))
        if use_layernorm: 
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden Layers (2 total body layers)
        for _ in range(2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_layernorm: 
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
        
        self.body = nn.Sequential(*layers)
        
        # Heads
        self.policy_head = nn.Linear(hidden_size, 2)
        
        if use_sigmoid:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, 1), 
                nn.Sigmoid()
            )
        else:
            self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights (basic Xavier for simplicity)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.body(x)
        p = self.policy_head(x)
        v = self.value_head(x).squeeze(-1)
        return p, v
    
    def print_architecture(self):
        ln_str = "LayerNorm" if self.use_layernorm else "No LN"
        sig_str = "Sigmoid" if self.use_sigmoid else "Linear"
        print(f"\n[Architecture] Body: {ln_str} | Value Head: {sig_str}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")


# =============================================================================
# WRAPPER FOR TRACKING AGENT COMPATIBILITY
# =============================================================================
class AblationNetWrapper(TorchPolicyValueNet):
    """Wrapper to inject DynamicCartPoleNet into existing infrastructure."""
    
    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "policy_weight": 2.0,
    }

    def __init__(self, use_layernorm, use_sigmoid, device=None, random_seed=42):
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        model = DynamicCartPoleNet(use_layernorm=use_layernorm, use_sigmoid=use_sigmoid)
        super().__init__(model)
        self.training_params = self.default_training_params.copy()
        self.DEVICE = get_accelerator() if device is None else device
    
    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        """Training method compatible with TrackingAgent."""
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]

        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=tp["learning_rate"],
            weight_decay=tp["weight_decay"]
        )

        if needs_reshape:
            # Examples format: (state, (policy, value))
            states = torch.from_numpy(np.array([state for state, (_, _) in examples], dtype=np.float32))
            policies = torch.from_numpy(np.array([policy for _, (policy, _) in examples], dtype=np.float32))
            values = torch.from_numpy(np.array([value for _, (_, value) in examples], dtype=np.float32))

        dataset = torch.utils.data.TensorDataset(states, policies, values)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=tp["batch_size"], shuffle=True
        )
        
        print(f"Training with {len(train_loader)} batches of size {tp['batch_size']}")
        
        train_losses = []
        for epoch in range(tp["epochs"]):
            model.train()
            train_loss = 0.0
            value_loss = 0.0
            policy_loss = 0.0

            for batch_states, batch_policies, batch_values in train_loader:
                batch_states = batch_states.to(self.DEVICE)
                batch_policies = batch_policies.to(self.DEVICE)
                batch_values = batch_values.to(self.DEVICE)
                
                optimizer.zero_grad()
                pred_policies, pred_values = model(batch_states)
                
                loss_value = criterion_value(pred_values, batch_values)
                loss_policy = criterion_policy(pred_policies, batch_policies)
                loss = loss_value + policy_weight * loss_policy
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                policy_loss += loss_policy.item()
                value_loss += loss_value.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_value_loss = value_loss / len(train_loader)
            avg_policy_loss = policy_loss / len(train_loader)
            
            epoch_metrics = {
                'total': avg_train_loss,
                'value': avg_value_loss,
                'policy': avg_policy_loss,
                'weighted_policy': policy_weight * avg_policy_loss
            }
            train_losses.append(epoch_metrics)

            if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
                print(f"Epoch {epoch+1}/{tp['epochs']}, Loss: {avg_train_loss:.4f} (v: {avg_value_loss:.4f}, p: {avg_policy_loss:.4f})")

        return model, [], train_losses
    
    def predict(self, state):
        self.model.cpu()
        nn_input = torch.tensor(state).float().reshape(1, -1)
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)
        
        policy_prob = policy_prob.numpy().squeeze(0)
        value = value.numpy().squeeze(0)
        
        return policy_prob, value


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def run_experiment(name, use_ln, use_sig, config):
    """Run single experiment with given configuration."""
    print(f"\n{'='*70}")
    print(f">>> EXPERIMENT: {name}")
    print(f"{'='*70}")
    
    # Print architecture details
    print(f"\n[Architecture Configuration]")
    print(f"  LayerNorm:        {'ENABLED' if use_ln else 'DISABLED'}")
    print(f"  Value Head:       {'Sigmoid [0,1]' if use_sig else 'Linear (unbounded)'}")
    print(f"  Hidden Size:      {DEFAULT_CONFIG['hidden_size']}")
    print(f"  Body Layers:      {DEFAULT_CONFIG['n_body_layers']}")
    print(f"  Initialization:   Xavier Uniform")
    
    # Setup
    game = CartPoleGame(max_steps=config["max_steps"])
    net = AblationNetWrapper(
        use_layernorm=use_ln, 
        use_sigmoid=use_sig,
        random_seed=config["seed"]
    )
    net.model.print_architecture()
    
    agent = TrackingAgent(
        game, net,
        n_games_per_train=config["games_train"],
        n_games_per_eval=config["games_eval"],
        mcts_params={
            "n_simulations": config["mcts_sims"], 
            "c_exploration": config["c_expl"]
        },
        use_gating=False,  # Disable gating for clean comparison
        random_seeds={
            "mcts": config["seed"], 
            "train": config["seed"] + 1, 
            "eval": config["seed"] + 2
        }
    )
    
    # Train
    rewards = []
    value_losses = []
    policy_losses = []
    
    for i in range(config["iters"]):
        agent.play_and_train()
        r = agent.history['reward_mean'][-1]
        vl = agent.history['loss_value'][-1]
        pl = agent.history['loss_policy'][-1]
        rewards.append(r)
        value_losses.append(vl)
        policy_losses.append(pl)
        print(f"   Iter {i+1}: Reward={r:.3f}, VLoss={vl:.4f}, PLoss={pl:.4f}")
        
        # Live plot update every 2 iterations
        if config.get("out_dir") and (i + 1) % 2 == 0:
            current_result = {
                "name": name,
                "rewards": rewards.copy(),
                "value_losses": value_losses.copy(),
                "policy_losses": policy_losses.copy()
            }
            all_results = config.get("all_results", []) + [current_result]
            live_plot_path = config["out_dir"] / f"live_progress.png"
            plot_comparison(all_results, live_plot_path, title_suffix=f" (Iter {i+1})")
    
    return {
        "name": name,
        "rewards": rewards,
        "value_losses": value_losses,
        "policy_losses": policy_losses,
        "final_reward": rewards[-1],
        "config": {"use_layernorm": use_ln, "use_sigmoid": use_sig}
    }


def plot_comparison(results, save_path, title_suffix=""):
    """
    Plot comparison of all experiments with enhanced visualization.
    
    Color Scheme:
    - A (Simple):     Green (#27ae60) - solid line, circle markers
    - B (LN-Only):    Blue  (#2980b9) - dashed line, square markers  
    - C (LN+Sigmoid): Red   (#c0392b) - dotted line, triangle markers
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Ensure non-interactive backend
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plot.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Meta-Ablation: LayerNorm & Sigmoid Impact{title_suffix}", 
                 fontsize=14, fontweight='bold')
    
    # Enhanced styling for each condition
    styles = [
        {'color': '#27ae60', 'marker': 'o', 'linestyle': '-',  'label': 'A: Simple (baseline)'},
        {'color': '#2980b9', 'marker': 's', 'linestyle': '--', 'label': 'B: LN-Only'},
        {'color': '#c0392b', 'marker': '^', 'linestyle': ':',  'label': 'C: LN+Sigmoid'},
    ]
    
    for idx, result in enumerate(results):
        if idx >= len(styles):
            break
        style = styles[idx]
        iters = range(1, len(result["rewards"]) + 1)
        
        # Use short name or style label
        label = result.get("name", style['label'])
        
        # Reward plot
        axes[0].plot(iters, result["rewards"], 
                     color=style['color'], marker=style['marker'], 
                     linestyle=style['linestyle'],
                     markersize=8, linewidth=2.5, label=label,
                     markeredgecolor='white', markeredgewidth=1)
        
        # Value Loss plot
        axes[1].plot(iters, result["value_losses"], 
                     color=style['color'], marker=style['marker'],
                     linestyle=style['linestyle'],
                     markersize=8, linewidth=2.5, label=label,
                     markeredgecolor='white', markeredgewidth=1)
        
        # Policy Loss plot
        axes[2].plot(iters, result["policy_losses"], 
                     color=style['color'], marker=style['marker'],
                     linestyle=style['linestyle'],
                     markersize=8, linewidth=2.5, label=label,
                     markeredgecolor='white', markeredgewidth=1)
    
    # Reward subplot
    axes[0].set_title("Mean Reward (↑ higher is better)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Iteration", fontsize=10)
    axes[0].set_ylabel("Reward", fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#fafafa')
    
    # Value Loss subplot
    axes[1].set_title("Value Loss (↓ lower is better)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Iteration", fontsize=10)
    axes[1].set_ylabel("Value Loss (log scale)", fontsize=10)
    axes[1].set_yscale('log')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#fafafa')
    
    # Policy Loss subplot
    axes[2].set_title("Policy Loss (↓ lower is better)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Iteration", fontsize=10)
    axes[2].set_ylabel("Policy Loss", fontsize=10)
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[Plot] Saved to {save_path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Meta-Ablation Experiment: Isolate LayerNorm & Sigmoid Effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (8 iterations)
  python meta_ablation.py
  
  # Quick test (2 iterations)
  python meta_ablation.py --iters 2 --games_train 10 --mcts_sims 10
  
  # Custom seed
  python meta_ablation.py --seed 123
"""
    )
    parser.add_argument("--iters", type=int, default=DEFAULT_CONFIG["iters"], 
                        help=f"Training iterations per condition (default: {DEFAULT_CONFIG['iters']})")
    parser.add_argument("--games_train", type=int, default=DEFAULT_CONFIG["games_train"], 
                        help=f"Games per training batch (default: {DEFAULT_CONFIG['games_train']})")
    parser.add_argument("--games_eval", type=int, default=DEFAULT_CONFIG["games_eval"], 
                        help=f"Games per evaluation (default: {DEFAULT_CONFIG['games_eval']})")
    parser.add_argument("--mcts_sims", type=int, default=DEFAULT_CONFIG["mcts_sims"], 
                        help=f"MCTS simulations per move (default: {DEFAULT_CONFIG['mcts_sims']})")
    parser.add_argument("--c_expl", type=float, default=DEFAULT_CONFIG["c_expl"], 
                        help=f"MCTS exploration constant (default: {DEFAULT_CONFIG['c_expl']})")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"], 
                        help=f"Max steps per episode (default: {DEFAULT_CONFIG['max_steps']})")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], 
                        help=f"Random seed (default: {DEFAULT_CONFIG['seed']})")
    args = parser.parse_args()
    
    config = vars(args)
    
    # Define conditions
    conditions = [
        ("A: Simple (No LN, No Sig)", False, False),
        ("B: LN-Only (LN, No Sig)",   True,  False),
        ("C: LN+Sigmoid (LN, Sig)",   True,  True),
    ]
    
    # Create output directory FIRST so we can save live plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"meta_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config["out_dir"] = out_dir
    
    # Print comprehensive config at startup
    print_experiment_config(config, conditions)
    
    results = []
    for name, use_ln, use_sig in conditions:
        config["all_results"] = results.copy()  # Pass previous results for live plotting
        result = run_experiment(name, use_ln, use_sig, config)
        results.append(result)
        
        # Save intermediate plot after each experiment completes
        intermediate_path = out_dir / f"progress_{len(results)}_experiments.png"
        plot_comparison(results, intermediate_path, title_suffix=f" ({len(results)}/3 complete)")
    
    # Save final results
    results_path = out_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to {results_path}")
    
    # Final comparison plot
    plot_path = out_dir / "ablation_comparison.png"
    plot_comparison(results, plot_path, title_suffix=" (FINAL)")
    
    # Print summary
    print("\n" + "="*70)
    print("META-ABLATION COMPLETE")
    print("="*70)
    print("\nFinal Results:")
    for r in results:
        print(f"  {r['name']}: Final Reward = {r['final_reward']:.4f}")
    print(f"\nOutput: {out_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
