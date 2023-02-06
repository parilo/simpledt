from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from simpledt.models.dtpolicy import DTPolicy
from simpledt.rollout import BatchOfSeq


# def calculate_reward_to_go(rewards: torch.Tensor, discount_factor: float) -> torch.Tensor:
#     """Calculate the reward-to-go for each sequence in a batch of sequences."""
#     reward_to_go = rewards.clone()
#     for i in range(reward_to_go.shape[1] - 2, -1, -1):
#         reward_to_go[:, i] += discount_factor * reward_to_go[:, i + 1]
#     return reward_to_go


# def normalize_reward_to_go(reward_to_go: torch.Tensor) -> torch.Tensor:
#     """Normalize the reward-to-go values of a batch of sequences."""
#     mean = reward_to_go.mean()
#     std = reward_to_go.std()
#     return (reward_to_go - mean) / (std + 1e-5)


# def normalize_reward_to_go(reward_to_go: torch.Tensor) -> torch.Tensor:
#     """Normalize the reward-to-go values of a batch of sequences."""
#     rmax = reward_to_go.max()
#     rmin = reward_to_go.min()
#     return (reward_to_go - rmin) / (rmax - rmin + 1e-5)


# def calculate_reward_to_go(
#     rewards: torch.Tensor, discount_factor: float
# ) -> torch.Tensor:
#     reward_to_go = rewards.clone()
#     reward_to_go[:] = reward_to_go.sum(dim=1).unsqueeze(dim=1)
#     return reward_to_go


# def normalize_reward_to_go(reward_to_go: torch.Tensor, debug=False) -> torch.Tensor:
#     # mean = reward_to_go.mean()
#     # rmax = reward_to_go.max()
#     # rmin = reward_to_go.min()
#     # factor = 0.8
#     # redge = factor * rmax + (1 - factor) * rmin
#     # if debug:
#     #     print(f'--- redge {redge}')
#     #     print(f'--- reward_to_go {reward_to_go.mean(1)[..., 0]}')
#     # reward_to_go = torch.where(
#     #     reward_to_go > redge,
#     #     torch.ones_like(reward_to_go),
#     #     torch.zeros_like(reward_to_go)
#     # )
#     return reward_to_go


def calc_reward_to_go(rewards: torch.Tensor):
    reward_to_go = torch.flip(torch.cumsum(torch.flip(rewards, [1]), 1), [1])
    return reward_to_go


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

    def train_on_batch(self, batch: BatchOfSeq, debug=False) -> Dict[str, float]:
        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards

        # Move the tensors to the device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        # Get the reward-to-go for each sequence in the batch
        reward_to_go = calc_reward_to_go(rewards)

        # use only first part
        # observations = observations[:, :100]
        # actions = actions[:, :100]
        # rewards = rewards[:, :100]
        # reward_to_go = reward_to_go[:, :100]

        # reward_to_go_norm = normalize_reward_to_go(reward_to_go)

        # Get the next action predicted by the policy for each sequence
        # if debug:
        #     import numpy
        #     numpy.set_printoptions(suppress=True)
        #     print(f'--- mean {reward_to_go.mean()}')
        #     print(f'--- rate {reward_to_go_norm.mean()}')
        #     print(rewards[0, :2].cpu().numpy())
        #     print(reward_to_go[0, :2].cpu().numpy())
        #     print(reward_to_go_norm[0, :2].cpu().numpy())

        # reward_to_go_norm_mean = reward_to_go_norm.mean(dim=1)[:, 0]
        # good_ind = reward_to_go_norm_mean > 0
        # observations = observations[good_ind]
        # reward_to_go_norm = reward_to_go_norm[good_ind]
        # actions = actions[good_ind]
        # if debug:
        #     print(f'--- good_ind {good_ind}')
        #     print(f'--- good observations {observations.shape}')
        #     print(f'--- good reward_to_go_norm {reward_to_go_norm.shape}')
        #     # print(f'--- good reward_to_go_norm {reward_to_go_norm}')
        #     print(f'--- good actions {actions.shape}')

        # CEM
        # rollout_rewards = reward_to_go_norm[:, 0, 0]
        # inds = torch.argsort(rollout_rewards, descending=True)
        # use_roll_num = int(0.1 * observations.shape[0])
        # inds = inds[:use_roll_num]
        # observations = observations[inds]
        # reward_to_go_norm = reward_to_go_norm[inds]
        # reward_to_go_norm[:] = 1.
        # actions = actions[inds]

        # RCBC
        # rollout_rewards = reward_to_go[:, 0, 0]
        # rollout_rewards_norm = reward_to_go_norm[:, 0, 0]
        # inds = torch.argsort(rollout_rewards, descending=True)
        # best_roll_num = int(0.1 * observations.shape[0])
        # inds = inds[:best_roll_num]

        # if debug:
        #     import numpy

        #     numpy.set_printoptions(suppress=True)
        #     print(
        #         f"--- best mean {rollout_rewards[inds].mean()} / {rollout_rewards.mean()} "
        #         f"rewards {rollout_rewards[inds].cpu().numpy()} "
        #         f"rewards norm {rollout_rewards_norm[inds].cpu().numpy()} "
        #     )

        # next_actions = self.policy(observations, reward_to_go_norm, actions)

        # Compute the loss
        loss = self.criterion(next_actions, actions)

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "dt_loss": loss.item(),
            # 'reward_to_go_mean': reward_to_go.mean(),
            # 'reward_to_go_std': reward_to_go.std(),
        }
