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
