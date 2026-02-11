"Utils that rely on PyTorch and such"

import torch
import gymnasium as gym

def get_accelerator():
    # The following would work on recent PyTorch:
    # (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    # but we want to support PyTorch 2.2, so:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"

# multiprocessing doesn't like anonymous functions so we can't use TransformReward(... lambda...)
class ScaleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class CumulativeRewardWrapper(gym.Wrapper):
    """Wrapper that implements sparse reward: 0 at every step, steps/max_steps at termination."""

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

        if self.step_count >= self.max_steps:
            truncated = True

        if terminated or truncated:
            reward = self.step_count / self.max_steps
        else:
            reward = 0.0

        assert reward == 0.0 or (terminated or truncated), "Non-zero reward only at terminal state"
        assert 0.0 <= reward <= 1.0, f"Reward out of bounds: {reward}"

        return observation, reward, terminated, truncated, info


class Tee:
    """File-like object that writes to multiple streams simultaneously (e.g. stdout + log file)."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
