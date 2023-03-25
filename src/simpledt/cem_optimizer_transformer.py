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

    def _calc_loss(self, batch: BatchOfSeq, entropy_reg_weight: float = 0) -> torch.Tensor:
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
        num_action_bins = next_actions.shape[-1]
        next_actions_reshaped = next_actions.reshape(-1, num_action_bins)
        targets = targets.reshape(-1)
        # print(f'--- next_actions {next_actions.shape} next_actions_reshaped {next_actions_reshaped.shape} targets {targets.shape}')

        ce_loss = F.cross_entropy(next_actions_reshaped, targets)
        if entropy_reg_weight > 0:
            # random_targets = torch.randint_like(targets, num_action_bins)
            # ce_loss += entropy_reg_weight * F.cross_entropy(next_actions_reshaped, random_targets)

            # reg_loss = entropy_reg_weight * next_actions_reshaped
            # reg_loss[next_actions_reshaped < 1e-3] = 0
            # ce_loss += reg_loss.mean()

            ce_loss += entropy_reg_weight * F.mse_loss(
                next_actions_reshaped,
                torch.zeros_like(next_actions_reshaped)
            )

        return ce_loss, next_actions

    def train_on_batch(self, batch: BatchOfSeq) -> Dict[str, float]:

        # Compute the loss
        loss, action_bins = self._calc_loss(batch, entropy_reg_weight=0.05)

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "ce_loss": {
                "train": loss.item(),
            },
            "action_logits_min": {
                "train": action_bins.min().item(),
            },
            "action_logits_max": {
                "train": action_bins.max().item(),
            },
        }

    def validate_on_batch(self, batch: BatchOfSeq) -> Dict[str, float]:

        # Compute the loss
        with torch.no_grad():
            loss, action_bins = self._calc_loss(batch)

        return {
            "ce_loss": {
                "valid": loss.item(),
            },
            "action_logits_min": {
                "valid": action_bins.min().item(),
            },
            "action_logits_max": {
                "valid": action_bins.max().item(),
            },
        }
