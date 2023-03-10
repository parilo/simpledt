import torch
import torch.nn as nn


def generate_square_subsequent_mask(size: int, diagonal: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=diagonal)


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

        self.pos_encoding = nn.Embedding(self.output_seq_len + 1, self.hidden_size).to(
            self.device
        )
        self.input_fc = nn.Linear(self.obs_size + 1, self.hidden_size).to(self.device)
        self.output_fc = nn.Linear(self.hidden_size, self.action_size).to(self.device)
        self.action_fc = nn.Linear(self.action_size, self.hidden_size).to(self.device)
        self._src_mask = generate_square_subsequent_mask(self.output_seq_len + 1, 2).to(
            device
        )
        self._tgt_mask = generate_square_subsequent_mask(self.output_seq_len + 1, 1).to(
            device
        )

    def forward(
        self,
        observations: torch.Tensor,
        reward_to_go: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Add first zero step
        observations = torch.cat(
            [torch.zeros_like(observations[:, 0:1]), observations], dim=1
        )
        reward_to_go = torch.cat(
            [torch.zeros_like(reward_to_go[:, 0:1]), reward_to_go], dim=1
        )
        actions = torch.cat([torch.zeros_like(actions[:, 0:1]), actions], dim=1)

        # Convert observations and reward_to_go to tensors and move to the device
        observations = observations.to(self.device)
        reward_to_go = reward_to_go.to(self.device)
        actions = actions.to(self.device)

        # Concatenate observations and reward_to_go and convert to the batch-first format
        x = torch.cat((observations, reward_to_go), dim=2)
        x = self.input_fc(x)

        # Add positional encoding to the input
        seq_len = observations.shape[1]
        pos_enc = self.pos_encoding(
            torch.arange(seq_len, dtype=torch.long).to(self.device)
        ).to(self.device)

        # Use the transformer to process the input
        # print(f'--- x {x.shape} pos enc {pos_enc.shape} mask {self._src_mask.shape}')
        # print(f'--- act {actions.shape} pos enc {pos_enc.shape} mask {self._tgt_mask.shape}')
        x = self.transformer(
            src=x + pos_enc,
            tgt=self.action_fc(actions) + pos_enc,
            src_mask=self._src_mask[:seq_len, :seq_len],
            memory_mask=self._tgt_mask[:seq_len, :seq_len],
            tgt_mask=self._tgt_mask[:seq_len, :seq_len],
        )

        # Pass the output through a linear layer to get the next action
        next_action = self.output_fc(x)

        return next_action[:, 1:]
