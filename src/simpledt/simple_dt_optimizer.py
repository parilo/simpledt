from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from simpledt.models.dtpolicy import DTPolicy
from simpledt.rollout import BatchOfSeq


def calculate_reward_to_go(rewards: torch.Tensor, discount_factor: float) -> torch.Tensor:
    """Calculate the reward-to-go for each sequence in a batch of sequences."""
    reward_to_go = rewards.clone()
    for i in range(reward_to_go.shape[1] - 2, -1, -1):
        reward_to_go[:, i] += discount_factor * reward_to_go[:, i + 1]
    return reward_to_go


def normalize_reward_to_go(reward_to_go: torch.Tensor) -> torch.Tensor:
    """Normalize the reward-to-go values of a batch of sequences."""
    mean = reward_to_go.mean()
    std = reward_to_go.std()
    return (reward_to_go - mean) / (std + 1e-5)


class SimpleDTOptimizer:
    def __init__(
        self,
        policy: DTPolicy,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        discount_factor: float,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.discount_factor = discount_factor

    def train_on_batch(self, batch: BatchOfSeq) -> Dict[str, float]:
        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards

        # Move the tensors to the device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        # Get the reward-to-go for each sequence in the batch
        reward_to_go = calculate_reward_to_go(rewards, self.discount_factor)
        reward_to_go = normalize_reward_to_go(reward_to_go)

        # Get the next action predicted by the policy for each sequence
        next_actions = self.policy(observations, reward_to_go, actions)

        # Compute the loss
        loss = self.criterion(next_actions, actions)

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'dt_loss': loss.item()
        }
