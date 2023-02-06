import torch
import torch.nn as nn


def generate_square_subsequent_mask(size: int, diagonal: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=diagonal)


class TransformerPolicy(nn.Module):
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
        super().__init__()
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
            dropout=0,
        )

        # self.pos_encoding = nn.Embedding(self.output_seq_len + 1, self.hidden_size).to(
        #     self.device
        # )
        self.observation_fc = nn.Linear(self.obs_size, self.hidden_size).to(self.device)
        self.output_fc = nn.Linear(self.hidden_size, self.action_size).to(self.device)
        self.action_fc = nn.Linear(self.action_size, self.hidden_size).to(self.device)
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
        actions = torch.cat([torch.zeros_like(actions[:, 0:1]), actions], dim=1)[:, :-1]

        # Move to the device
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        # Make tokens
        obs_tok = self.observation_fc(observations)
        actions_tok = self.action_fc(actions)

        # Add positional encoding to the input
        seq_len = observations.shape[1]
        # pos_enc = self.pos_encoding(torch.arange(seq_len, dtype=torch.long).to(self.device)).to(
        #     self.device
        # )

        # Use the transformer to process the input
        out_tok = self.transformer.decoder(
            tgt=actions_tok,
            memory=obs_tok,
            tgt_mask=self._tgt_mask[:seq_len, :seq_len],
            memory_mask=self._tgt_mask[:seq_len, :seq_len],
        )

        # Pass the output through a linear layer to get the next action
        next_action = self.output_fc(out_tok)
        return next_action
