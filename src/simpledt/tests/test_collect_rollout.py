import gymnasium as gym
import torch

from simpledt.collect import collect_rollout
from simpledt.models.dtpolicy import DTPolicy


def test_collect_rollout():
    # Set up the environment and max number of steps
    env = gym.make("Pendulum-v1")
    max_steps = 100

    # Set up the policy
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    policy = DTPolicy(
        obs_size=obs_size,
        action_size=action_size,
        hidden_size=32,
        num_layers=2,
        nhead=4,
        dim_feedforward=64,
        output_seq_len=max_steps,
        batch_first=True,
        device=torch.device("cpu"),
    )

    # Collect a rollout
    rollout = collect_rollout(env, policy, max_steps)

    # Check that the observations, actions, and rewards have the correct shape
    assert rollout.observations.shape == (max_steps + 1, obs_size)
    assert rollout.actions.shape == (max_steps, action_size)
    assert rollout.rewards.shape == (max_steps, 1)

    # Check that the terminated and truncated flags have the correct shape and values
    assert rollout.terminated.shape == (max_steps, 1)
    assert rollout.truncated.shape == (max_steps, 1)
