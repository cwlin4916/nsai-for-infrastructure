from abc import ABC, abstractmethod
from typing import Tuple, Any, TypeVar, Generic

from gymnasium import Env, Space

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Game(ABC, Generic[ObsType, ActType]):
    """
    Abstract base class for single-player games compatible with this package's AlphaZero
    implementation. Written so that a Farama Gymnasium will automatically be a full
    implementation. In addition to implementing the abstract methods, the class must define
    fields `action_space` and `observation_space`.
    """

    state: ObsType | None

    def __init__(self):
        self.state = None

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, Any, bool, bool, dict[str, Any]]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[ObsType, dict[str, Any]]:
        pass

    def reset_wrapper(self, *args, **kwargs):
        obs, info = self.reset(*args, **kwargs)
        self.state = obs
        return obs, info
    
    def step_wrapper(self, action: ActType) -> Tuple[ObsType, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.step(action)
        self.state = obs
        return obs, reward, terminated, truncated, info
    
    @property
    def get_current_state(self) -> ObsType | None:
        return self.state

class EnvGame(Game[ObsType, ActType]):
    """
    Abstract base class for Farama Gymnasium-based games. Contains a field `env:
    Env[ObsType, ActType]` populated in the constructor; implements `step` and `reset`
    methods that wrap `env`'s `step` and `reset`.
    """

    env: Env[ObsType, ActType]
    action_space: Space[ActType]
    observation_space: Space[ObsType]

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action: ActType) -> Tuple[ObsType, Any, bool, bool, dict[str, Any]]:
        return self.env.step(action)
    
    def reset(self, *args, **kwargs) -> Tuple[ObsType, dict[str, Any]]:
        return self.env.reset(*args, **kwargs)
    
    def render(self, *args, **kwargs):
        "For convenience, we provide a `render` method that calls `env`'s `render`."
        return self.env.render(*args, **kwargs)
