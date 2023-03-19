import torch

from simpledt.rollout import Rollout
from simpledt.cem_optimizer import get_best_n_rollouts_list


def test_get_best_n_rollouts_list():
    # Create some example rollouts
    rollouts = []
    for i in range(5):
        observations = {'state': torch.randn(4)}
        actions = torch.randn(10, 2)
        rewards = torch.randn(10)
        terminated = torch.zeros(10)
        truncated = torch.zeros(10)
        info = {'episode_length': torch.tensor(10)}
        rollout = Rollout(observations=observations, actions=actions, rewards=rewards, terminated=terminated, truncated=truncated, info=info)
        rollouts.append(rollout)

    # Set the total reward for each rollout
    total_rewards = [i for i in range(5)]
    for i, rollout in enumerate(rollouts):
        rollout.rewards = total_rewards[i] * torch.ones(10)

    # Test the function with different values of num
    for num in range(1, 6):
        best_rollouts = get_best_n_rollouts_list(num, rollouts)
        expected_rollouts = rollouts[-num:][::-1]
        assert len(best_rollouts) == num
        assert all([r1 is r2 for r1, r2 in zip(best_rollouts, expected_rollouts)])