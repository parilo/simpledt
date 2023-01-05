import torch
import torch.nn as nn
import torch.optim as optim

from simpledt.models.dtpolicy import DTPolicy
from simpledt.rollout import BatchOfSeq
from simpledt.simple_dt_optimizer import SimpleDTOptimizer

def test_simple_dt_optimizer():
    # Create a dummy DTPolicy and optimizer
    obs_size = 10
    action_size = 5
    hidden_size = 20
    num_layers = 2
    nhead = 4
    dim_feedforward = 512
    batch_first = True
    seq_len = 16
    device = torch.device("cpu")
    policy = DTPolicy(
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
    optimizer = optim.SGD(policy.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    discount_factor = 0.9

    # Create a SimpleDTOptimizer instance
    optimizer = SimpleDTOptimizer(policy, optimizer, criterion, device, discount_factor)

    # Create a dummy batch of data
    observations = torch.randn(3, seq_len, obs_size)
    actions = torch.randn(3, seq_len, action_size)
    rewards = torch.randn(3, seq_len, 1)
    batch = BatchOfSeq(observations, actions, rewards)

    # Call the train_on_batch method
    train_info = optimizer.train_on_batch(batch)

    # Check if the output is a scalar tensor
    assert isinstance(train_info['dt_loss'], float)

