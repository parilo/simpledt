import torch
from simpledt.rollout import Rollout, trim_rollout


def test_trim_rollout():
    # Create a sample rollout
    rollout = Rollout(
        observations=torch.randn(20, 4).float(),
        actions=torch.tensor([[0, 1] for _ in range(3)] + [[1, 0] for _ in range(7)] + [[0, 1] for _ in range(3)] + [[1, 0] for _ in range(7)]).float(),
        rewards=torch.randn(20).float(),
        terminated=torch.zeros(20, dtype=torch.bool),
        truncated=torch.zeros(20, dtype=torch.bool),
        info={'step': torch.arange(20)},
        size=20,
        total_reward=0.0
    )
    rollout.total_reward = rollout.rewards.sum()

    # Test case 1: action not found in rollout
    zero_action=torch.tensor([1, 1]).float()
    truncated_rollout = trim_rollout(rollout, zero_action=zero_action)
    assert truncated_rollout is None

    # Test case 2: action found in middle of rollout
    zero_action=torch.tensor([0, 1]).float()
    ind1 = 3
    ind2 = 20 - ind1
    truncated_rollout = trim_rollout(rollout, zero_action=zero_action)
    assert truncated_rollout.observations.shape == (ind2, 4)
    assert truncated_rollout.actions.shape == (ind2, 2)
    assert truncated_rollout.rewards.shape == (ind2,)
    assert truncated_rollout.terminated.shape == (ind2,)
    assert truncated_rollout.truncated.shape == (ind2,)
    assert truncated_rollout.info['step'].shape == (ind2,)
    assert truncated_rollout.size == ind2

    assert torch.allclose(truncated_rollout.observations, rollout.observations[ind1:])
    assert torch.allclose(truncated_rollout.actions, rollout.actions[ind1:])
    assert torch.allclose(truncated_rollout.rewards, rollout.rewards[ind1:])
    assert torch.allclose(truncated_rollout.terminated, rollout.terminated[ind1:])
    assert torch.allclose(truncated_rollout.truncated, rollout.truncated[ind1:])
    for key, val in truncated_rollout.info.items():
        assert torch.allclose(val, rollout.info[key][ind1:])
    assert truncated_rollout.size == ind2
    assert torch.allclose(truncated_rollout.total_reward, rollout.rewards[ind1:].sum())

    # Test case 3: action found at beginning of rollout
    zero_action=torch.tensor([1, 0]).float()
    ind1 = 0
    ind2 = 20 - ind1
    truncated_rollout = trim_rollout(rollout, zero_action=zero_action)
    assert truncated_rollout.observations.shape == (ind2, 4)
    assert truncated_rollout.actions.shape == (ind2, 2)
    assert truncated_rollout.rewards.shape == (ind2,)
    assert truncated_rollout.terminated.shape == (ind2,)
    assert truncated_rollout.truncated.shape == (ind2,)
    assert truncated_rollout.info['step'].shape == (ind2,)
    assert truncated_rollout.size == ind2

    assert torch.allclose(truncated_rollout.observations, rollout.observations[ind1:])
    assert torch.allclose(truncated_rollout.actions, rollout.actions[ind1:])
    assert torch.allclose(truncated_rollout.rewards, rollout.rewards[ind1:])
    assert torch.allclose(truncated_rollout.terminated, rollout.terminated[ind1:])
    assert torch.allclose(truncated_rollout.truncated, rollout.truncated[ind1:])
    for key, val in truncated_rollout.info.items():
        assert torch.allclose(val, rollout.info[key][ind1:])
    assert truncated_rollout.size == ind2
    assert torch.allclose(truncated_rollout.total_reward, rollout.rewards.sum())
