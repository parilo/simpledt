import torch

from simpledt.rollout import Rollout


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        observation_shape: dict,
        action_shape: tuple,
        info_shape: dict,
    ):
        self.max_size = max_size
        self.observation_keys = list(observation_shape.keys())
        self.observation_shapes = list(observation_shape.values())
        self.action_shape = action_shape
        self.info_keys = list(info_shape.keys())
        self.info_shapes = list(info_shape.values())
        self.observations = {
            key: torch.empty((max_size, *shape))
            for key, shape in observation_shape.items()
        }
        self.actions = torch.empty((max_size, *action_shape))
        self.rewards = torch.empty((max_size, 1))
        self.terminated = torch.empty((max_size, 1), dtype=torch.bool)  # fixed
        self.truncated = torch.empty((max_size, 1), dtype=torch.bool)  # fixed
        self.info = {
            key: torch.empty((max_size, *shape)) for key, shape in info_shape.items()
        }
        self.size = 0
        self.current_index = 0

    def add_rollout(self, rollout: Rollout):
        for i, key in enumerate(self.observation_keys):
            self.observations[key][self.current_index] = rollout.observations[key]
        self.actions[self.current_index] = rollout.actions
        self.rewards[self.current_index] = rollout.rewards
        self.terminated[self.current_index] = rollout.terminated
        self.truncated[self.current_index] = rollout.truncated
        for i, key in enumerate(self.info_keys):
            self.info[key][self.current_index] = rollout.info[key]
        self.size = min(self.size + 1, self.max_size)
        self.current_index = (self.current_index + 1) % self.max_size

    def sample(self, batch_size: int):
        indices = torch.randperm(self.size)[:batch_size]
        observations = {
            key: self.observations[key][indices] for key in self.observation_keys
        }
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminated = self.terminated[indices]
        truncated = self.truncated[indices]
        info = {key: self.info[key][indices] for key in self.info_keys}
        return observations, actions, rewards, terminated, truncated, info
