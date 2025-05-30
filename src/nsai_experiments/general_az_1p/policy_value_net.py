from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

class PolicyValueNet(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, examples):
        """
        Takes a list of training examples, where each example is a tuple of the form
        `(state, (policy, value))`, where `state` is the state observation in the exact
        format returned by the game.
        """
        pass

    @abstractmethod
    def predict(self, state):
        """
        Takes a state observation in the exact format returned by the game and returns a
        `(policy, value)` tuple.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, save_dir):
        """
        Saves the model to the given directory.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, save_dir):
        """
        Loads the model from the given directory to which `save_checkpoint` has previously
        saved the model.
        """
        pass

class TorchPolicyValueNet(PolicyValueNet):
    """
    Abstract base class for PyTorch-based policy-value networks. Contains a field `model:
    nn.Module` populated in the constructor; implements `save_checkpoint` and
    `load_checkpoint` methods that save and load this `model`. Users still need to implement
    `train` and `predict` methods to define their own training logic and any transformations
    that may need to happen before or after calling `model.forward()`, respectively.
    """

    model: nn.Module
    save_file_name: str = "model_checkpoint.pt"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def save_checkpoint(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / self.save_file_name)

    def load_checkpoint(self, save_dir):
        save_dir = Path(save_dir)
        save_file = save_dir / self.save_file_name
        if not save_file.exists():
            raise FileNotFoundError(f"Checkpoint file {save_file} does not exist.")
        self.model.load_state_dict(torch.load(save_file))
