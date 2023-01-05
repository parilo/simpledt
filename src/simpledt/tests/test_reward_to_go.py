import torch

from simpledt.simple_dt_optimizer import calculate_reward_to_go, normalize_reward_to_go


def test_calculate_reward_to_go():
    rewards = torch.tensor([
        [1., 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ])
    discount_factor = 0.9
    reward_to_go = [
        1 + 2 * discount_factor + 3 * discount_factor ** 2 + 4 * discount_factor ** 3,
        2 + 3 * discount_factor + 4 * discount_factor ** 2,
        3 + 4 * discount_factor,
        4
    ]
    expected_output = torch.tensor([
        reward_to_go,
        reward_to_go,
        reward_to_go,
    ])
    output = calculate_reward_to_go(rewards, discount_factor)
    assert torch.allclose(output, expected_output, atol=1e-3)


def test_normalize_reward_to_go():
    reward_to_go = torch.tensor([
        [1., 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ])
    mean = reward_to_go.mean()
    std = reward_to_go.std()
    expected_output = (reward_to_go - mean) / (std + 1e-5)
    output = normalize_reward_to_go(reward_to_go)
    assert torch.allclose(output, expected_output, atol=1e-3)
