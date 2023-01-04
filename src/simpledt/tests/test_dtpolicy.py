import torch

from simpledt.models.dtpolicy import DTPolicy


def test_DTPolicy():
    # Set the random seed for deterministic results
    torch.manual_seed(42)

    # Create a DTPolicy with random weights
    obs_size = 10
    action_size = 5
    hidden_size = 20
    num_layers = 2
    nhead = 4
    dim_feedforward = 512
    batch_first = True
    seq_len = 16
    device = torch.device("cpu")
    model = DTPolicy(
        obs_size=obs_size,
        action_size=action_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        output_seq_len=seq_len,
        batch_first=batch_first,
        device=device,
    )

    # Generate random input tensors
    batch_size = 8
    observations = torch.randn(batch_size, seq_len, obs_size, dtype=torch.float)
    reward_to_go = torch.randn(batch_size, seq_len, 1, dtype=torch.float)
    actions = torch.randn(batch_size, seq_len, action_size, dtype=torch.float)

    # Test the model on the random input tensors
    output = model(observations, reward_to_go, actions)
    assert output.shape == (batch_size, seq_len, action_size)


test_DTPolicy()
