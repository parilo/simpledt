from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from simpledt.models.transformer_decoder import TransformerDecoderPolicy
from simpledt.rollout import BatchOfSeq, Rollout


def get_best_n_rollouts_list(num: int, rollouts: List[Rollout]) -> List[Rollout]:
    # Compute the total reward for each rollout
    total_rewards = [torch.sum(r.rewards).item() for r in rollouts]
    # Sort the rollouts by total reward using argsort
    sorted_indices = torch.argsort(torch.tensor(total_rewards)).tolist()[::-1]
    sorted_rollouts = [rollouts[i] for i in sorted_indices]
    # Take the top num rollouts
    return sorted_rollouts[:num]


class TransformerCEMOptimizer:
    def __init__(
        self,
        policy: TransformerDecoderPolicy,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.device = device

    def _calc_loss(self, batch: BatchOfSeq) -> torch.Tensor:
        observations = batch.observations
        actions = batch.actions

        # Move the tensors to the device
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        # print(f'--- observations {observations.shape} actions {actions.shape}')

        policy_output = self.policy(
            observations=observations,
            actions=actions[:, :-1],
        )

        next_actions = policy_output[:, ::2]

        targets = actions.argmax(-1)
        # print(targets)
        # print(next_actions)
        next_actions_reshaped = next_actions.reshape(-1, next_actions.shape[-1])
        targets = targets.reshape(-1)
        # print(f'--- next_actions {next_actions.shape} next_actions_reshaped {next_actions_reshaped.shape} targets {targets.shape}')

        random_targets = torch.randint_like(targets, 12)
        ce_loss = F.cross_entropy(next_actions_reshaped, targets) + 0.1 * F.cross_entropy(next_actions_reshaped, random_targets)
        return ce_loss

    def train_on_batch(self, batch: BatchOfSeq) -> Dict[str, float]:

        # Compute the loss
        loss = self._calc_loss(batch)

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "ce_loss": {
                "train": loss.item(),
            }
        }

    def validate_on_batch(self, batch: BatchOfSeq) -> Dict[str, float]:

        # Compute the loss
        with torch.no_grad():
            loss = self._calc_loss(batch)

        return {
            "ce_loss": {
                "valid": loss.item(),
            }
        }
