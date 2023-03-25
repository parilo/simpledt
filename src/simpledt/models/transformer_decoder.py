import math
import torch
import torch.nn as nn


def generate_square_subsequent_mask(size: int, diagonal: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=diagonal)
    # return torch.triu(torch.ones(size, size), diagonal=1).bool()

# def generate_custom_mask(size: int) -> torch.Tensor:
#     # Create an empty mask
#     mask = torch.zeros(size, size * 2 - 1)
#     # Iterate through each row of the maswk
#     for i in range(size):
#         # Set the corresponding elements in the mask based on the triangular pattern
#         mask[i, :2*i+1] = 1
#     return mask


def concatenate_obs_act(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """
    Concatenates the observation and action tensors into a single tensor of the form
    [obs1, act1, obs2, act2, ..., obsN].

    Args:
    - obs: A torch.Tensor object representing the observation tensor with dimensions [batch_size, seq_len, hidden_size].
    - act: A torch.Tensor object representing the action tensor with dimensions [batch_size, seq_len - 1, hidden_size].

    Returns:
    - A torch.Tensor object representing the concatenated tensor with dimensions [batch_size, seq_len * 2 - 1, hidden_size].
    """
    # Get the batch size, sequence length, and hidden size of the tensors
    batch_size, seq_len, hidden_size = obs.size()
    # Create an empty tensor to store the concatenated tensor
    concatenated = torch.zeros(batch_size, seq_len * 2 - 1, hidden_size, device=obs.device)
    # Iterate through each sequence in the batch
    for i in range(batch_size):
        # Iterate through each position in the sequence
        for j in range(seq_len):
            # Copy the observation tensor to the corresponding location in the output tensor
            concatenated[i, 2*j] = obs[i, j]
            # If this is not the last position in the sequence, copy the action tensor to the corresponding location in the output tensor
            if j < seq_len - 1:
                concatenated[i, 2*j+1] = act[i, j]
    return concatenated


def create_mlp(input_size, hidden_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        # nn.Linear(hidden_size, hidden_size),
        # nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x  #self.dropout(x)


class TransformerDecoderPolicy(nn.Module):
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
        # self.input_fc = nn.Linear(self.obs_size, self.hidden_size).to(self.device)
        # self.output_fc = nn.Linear(self.hidden_size, self.action_size).to(self.device)
        # self.action_fc = nn.Linear(self.action_size, self.hidden_size).to(self.device)
        self.pos_encoder = PositionalEncoding(self.hidden_size).to(self.device)
        self.input_fc = create_mlp(self.obs_size, self.hidden_size, self.hidden_size).to(self.device)
        self.output_fc = create_mlp(self.hidden_size, self.hidden_size, self.action_size).to(self.device)
        self.action_fc = create_mlp(self.action_size, self.hidden_size, self.hidden_size).to(self.device)
        # init so action are near zeros
        # nn.init.normal_(self.action_fc.weight, mean=0.0, std=0.001)
        # nn.init.constant_(self.action_fc.bias, 0.0)
        self._tgt_mask = generate_square_subsequent_mask(output_seq_len + 1, 1).to(
            device
        )
        print('mask', self._tgt_mask)

    def to(self, device):
        self.device = device
        self.transformer = self.transformer.to(device)
        self.pos_encoder = self.pos_encoder.to(device)
        self.input_fc = self.input_fc.to(device)
        self.output_fc = self.output_fc.to(device)
        self.action_fc = self.action_fc.to(device)
        self._tgt_mask = self._tgt_mask.to(device)
        return self

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Move to the device
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        # Make tokens
        obs_tok = self.input_fc(observations)
        act_tok = self.action_fc(actions)
        tgt_tok = concatenate_obs_act(obs_tok, act_tok)

        batch_size = observations.shape[0]
        seq_len = observations.shape[1]
        seq_len = seq_len * 2 - 1

        # Add positional encoding to the input
        # tgt_pos_enc = self.pos_encoding(torch.arange(seq_len, dtype=torch.long).to(self.device)).to(
        #     self.device
        # )

        inp_tok = torch.zeros((batch_size, 1, self.hidden_size)).float().to(self.device)

        # print(f'--- tgt_tok {tgt_tok.shape} tgt_pos_enc {tgt_pos_enc.shape}')

        # Use the transformer to process the input
        out_tok = self.transformer.decoder(
            # tgt=tgt_tok + tgt_pos_enc,
            tgt=self.pos_encoder(tgt_tok),
            memory=inp_tok,
            tgt_mask=self._tgt_mask[:seq_len, :seq_len],
        )

        # Pass the output through a linear layer to get the next action
        next_tok = self.output_fc(out_tok)
        return next_tok
