from typing import Dict, Optional
import torch
from dataclasses import dataclass


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: Dict[str, torch.Tensor]
    size: int
    total_reward: float


@dataclass
class BatchOfSeq:
    observations: torch.Tensor = None
    actions: torch.Tensor = None
    rewards: torch.Tensor = None
    terminated: torch.Tensor = None
    truncated: torch.Tensor = None
    info: Dict[str, torch.Tensor] = None  # fixed


def trim_rollout(rollout: Rollout, zero_action: torch.Tensor) -> Optional[Rollout]:
    # Find the index where the first occurrence of `zero_action` in `actions`
    # print(zero_action.numpy().tolist())
    # print(rollout.actions.numpy().tolist())
    # mask = rollout.actions.ne(zero_action).all(-1)
    mask = rollout.actions.isclose(zero_action).all(-1)
    # print(mask)
    indices = (~mask).nonzero()
    index = indices[0].item() if indices.numel() > 0 else -1
    # print(rollout.actions.isclose(zero_action).all(-1))
    # print(mask)
    # print(indices)
    # print(index)
    if not mask.any():
        # If `given_value` is not found in `actions`, return None
        return None
    else:
        # Otherwise, truncate the rollout at the index and return a new Rollout object
        truncated_info = {}
        for key, value in rollout.info.items():
            truncated_info[key] = value[index:]
        return Rollout(
            observations=rollout.observations[index:],
            actions=rollout.actions[index:],
            rewards=rollout.rewards[index:],
            terminated=rollout.terminated[index:],
            truncated=rollout.truncated[index:],
            info=truncated_info,
            size=rollout.size - index,
            total_reward=rollout.total_reward - rollout.rewards[:index].sum()
        )
