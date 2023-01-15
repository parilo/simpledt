from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from simpledt.models.dtpolicy import DTPolicy
from simpledt.rollout import BatchOfSeq


def get_best_n_rollouts(num:int, rollouts: BatchOfSeq) -> BatchOfSeq:
    r_rewards = rollouts.rewards.sum(1)[..., 0]
    inds = torch.argsort(r_rewards, descending=True)[:num]
    return BatchOfSeq(
        observations={key: val[inds] for key, val in rollouts.observations.items()},
        actions=rollouts.actions[inds],
        rewards=rollouts.rewards[inds],
        terminated=rollouts.terminated[inds],
        truncated=rollouts.truncated[inds],
        info={key: val[inds] for key, val in rollouts.info.items()},
    )


class CEMOptimizer:
    def __init__(
        self,
        policy: DTPolicy,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def _calc_loss(self, batch: BatchOfSeq) -> torch.Tensor:
        observations = batch.observations
        actions = batch.actions

        # Move the tensors to the device
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        observations = observations.reshape(-1, 1, observations.shape[-1])
        actions = actions.reshape(-1, 1, actions.shape[-1])

        next_actions = self.policy(
            observations=observations,
            reward_to_go=None,
            actions=actions,
        )

        # Compute the loss
        return self.criterion(next_actions, actions)

    def train_on_batch(self, batch: BatchOfSeq) -> Dict[str, float]:

        # Compute the loss
        loss = self._calc_loss(batch)

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'cem_loss': {
                'train': loss.item(),
            }
        }

    def validate_on_batch(self, batch: BatchOfSeq) -> Dict[str, float]:

        # Compute the loss
        with torch.no_grad():
            loss = self._calc_loss(batch)

        return {
            'cem_loss': {
                'valid': loss.item(),
            }
        }
