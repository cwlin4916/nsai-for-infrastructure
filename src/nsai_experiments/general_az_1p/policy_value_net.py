from typing import Any, Iterable
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

class PolicyValueNet(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, examples: Iterable[tuple[Any, tuple[Any, Any]]]):
        """
        Takes a list of training examples, where each example is a tuple of the form
        `(state, (policy, value))`, where `state` is the state observation in the exact
        format returned by the game.
        """
        pass

    @abstractmethod
    def predict(self, state) -> tuple[Any, Any]:
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
    
    @abstractmethod
    def push_multiprocessing(self):
        """
        Prepares the instance for Python multiprocessing. This may involve moving tensors
        from GPU to CPU so they can be automatically copied, etc. Anything that needs to be
        erased from the instance before it is safe for multiprocessing can be returned; the
        caller should then restore these values by calling `pop_multiprocessing` with them.
        """
        pass

    @abstractmethod
    def pop_multiprocessing(self, *args):
        """
        Restores the instance after a call to `push_multiprocessing`. This may involve
        restoring from the arguments state that was erased and returned in
        `push_multiprocessing`.
        """
        pass

class TorchPolicyValueNet(PolicyValueNet):
    """
    Abstract base class for PyTorch-based policy-value networks. Contains a field `model:
    nn.Module` populated in the constructor; implements `save_checkpoint` and
    `load_checkpoint` methods that save and load this `model`. Users still need to implement
    `train` and `predict` methods to define their own training logic and any transformations
    that may need to happen before or after calling `model.forward()`, respectively.
    Functional defaults for `push_multiprocessing` and `pop_multiprocessing` are provided,
    but the user will need to override these if they have state beyond `model` that should
    not be copied across processes (e.g., an optimizer).
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

    def push_multiprocessing(self):
        self.model.cpu()
    
    def pop_multiprocessing(self, *args):
        pass
