import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int,
        device: torch.device,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.device = device

        self._model = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        ).to(device)

    def forward(
        self,
        observations: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Convert observations and reward_to_go to tensors and move to the device
        observations = observations.to(self.device)
        next_action = self._model(observations)
        return next_action
