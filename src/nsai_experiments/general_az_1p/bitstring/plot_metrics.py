import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

def plot_training_metrics(history, save_path=None):
    """
    Plots a 3-panel dashboard of AlphaZero training metrics.
    
    Args:
        history (dict): Dictionary containing lists of metrics:
            - 'iteration'
            - 'reward_mean', 'reward_std'
            - 'loss_policy', 'loss_value'
            - 'game_length'
        save_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    # Set style defaults if seaborn didn't set them (or even if it did, to be sure)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    iterations = history['iteration']
    
    # --- Panel 1: Reliability (Evaluation Reward) ---
    ax1 = axes[0]
    reward_mean = np.array(history['reward_mean'])
    reward_std = np.array(history['reward_std'])
    
    ax1.plot(iterations, reward_mean, label='Mean Reward', color='b', marker='o')
    ax1.fill_between(iterations, reward_mean - reward_std, reward_mean + reward_std, color='b', alpha=0.2, label='Std Dev')
    
    # Theoretical Max Line
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Theoretical Max (1.0)')
    
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Reward")
    ax1.set_title("Evaluation Reward over Time")
    ax1.legend(loc='lower right')
    ax1.set_ylim(-0.1, 1.1)
    
    # --- Panel 2: Learning Dynamics (Loss Convergence) ---
    ax2 = axes[1]
    loss_policy = np.array(history['loss_policy'])
    loss_value = np.array(history['loss_value'])
    
    # Plot Policy Loss (Left Axis)
    ax2.plot(iterations, loss_policy, label='Policy Loss', color='g', marker='s')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Policy Loss (Cross Entropy)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Right Axis: Value Loss (Log Scale)
    ax2_right = ax2.twinx()
    # Handle log(0)
    safe_loss_value = np.where(loss_value <= 0, 1e-10, loss_value)
    ax2_right.plot(iterations, safe_loss_value, label='Value Loss', color='m', marker='^', linestyle='--')
    ax2_right.set_ylabel("Value Loss (MSE) - Log Scale", color='m')
    ax2_right.tick_params(axis='y', labelcolor='m')
    ax2_right.set_yscale('log')
    
    ax2.set_title("Loss Convergence")
    
    # Combined Legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # --- Panel 3: Efficiency (Average Episode Length) ---
    ax3 = axes[2]
    game_lengths = np.array(history['game_length'])
    
    ax3.plot(iterations, game_lengths, label='Avg Game Length', color='orange', marker='d')
    
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Steps")
    ax3.set_title("Average Episode Length")
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Test with dummy data
    dummy_history = {
        'iteration': [1, 2, 3, 4, 5],
        'reward_mean': [0.1, 0.3, 0.6, 0.8, 0.95],
        'reward_std': [0.05, 0.1, 0.15, 0.1, 0.02],
        'loss_policy': [2.5, 2.0, 1.5, 1.0, 0.8],
        'loss_value': [0.5, 0.3, 0.1, 0.01, 0.001],
        'game_length': [20, 15, 12, 10, 8]
    }
    plot_training_metrics(dummy_history, "test_metrics.png")
