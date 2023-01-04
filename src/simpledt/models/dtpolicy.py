import torch
import torch.nn as nn


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)


import torch
import torch.nn as nn


class DTPolicy(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        output_seq_len: int,
        batch_first: bool,
        device: torch.device,
    ):
        super(DTPolicy, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.output_seq_len = output_seq_len
        self.batch_first = batch_first
        self.device = device

        self.transformer = nn.Transformer(
            d_model=self.hidden_size,
            nhead=self.nhead,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            batch_first=self.batch_first,
            device=self.device,
        )

        self.pos_encoding = nn.Embedding(self.output_seq_len, self.hidden_size).to(
            self.device
        )
        self.input_fc = nn.Linear(self.obs_size + 1, self.hidden_size).to(self.device)
        self.output_fc = nn.Linear(self.hidden_size, self.action_size).to(self.device)
        self.action_fc = nn.Linear(self.action_size, self.hidden_size).to(self.device)
        self._target_mask = generate_square_subsequent_mask(self.output_seq_len).to(
            device
        )

    def forward(
        self,
        observations: torch.Tensor,
        reward_to_go: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Convert observations and reward_to_go to tensors and move to the device
        # observations = torch.tensor(observations, dtype=torch.float).to(self.device)
        # reward_to_go = torch.tensor(reward_to_go, dtype=torch.float).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        # Concatenate observations and reward_to_go and convert to the batch-first format
        x = torch.cat((observations, reward_to_go), dim=2)
        x = self.input_fc(x)

        # Add positional encoding to the input
        x += self.pos_encoding(torch.arange(x.shape[1], dtype=torch.long)).to(
            self.device
        )

        # Use the transformer to process the input
        x = self.transformer(
            src=x, tgt=self.action_fc(actions), tgt_mask=self._target_mask
        )

        # Pass the output through a linear layer to get the next action
        next_action = self.output_fc(x)

        return next_action
