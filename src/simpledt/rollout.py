from typing import Dict
import torch
from dataclasses import dataclass


@dataclass
class Rollout:
    observations: Dict[str, torch.Tensor]  # fixed
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: Dict[str, torch.Tensor]  # fixed
    size: int
    total_reward: float


@dataclass
class BatchOfSeq:
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor = None
    truncated: torch.Tensor = None
    info: Dict[str, torch.Tensor] = None  # fixed
