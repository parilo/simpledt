from typing import Callable
import gymnasium as gym
import torch
import numpy as np

from simpledt.models.dtpolicy import DTPolicy
from simpledt.rollout import Rollout


def collect_rollout(
    env: gym.Env,
    obs_size: int,
    action_size: int,
    policy: DTPolicy,
    max_steps: int,
    num_history_steps: int = 1,
    action_info_to_action: Callable[[torch.Tensor], torch.Tensor] = None,
    action_to_env_action: Callable[[torch.Tensor], np.ndarray] = None,
    action_visualiser: Callable[[torch.Tensor], np.ndarray] = None,
) -> Rollout:
    observations = torch.zeros(1, max_steps + 1, obs_size, dtype=torch.float)
    actions = torch.zeros(1, max_steps, action_size, dtype=torch.float)
    rewards = torch.zeros(1, max_steps, 1, dtype=torch.float)
    terminated = []
    truncated = []
    info = {}

    # Initialize the environment and get the first observation
    observation, _ = env.reset()
    observations[0, 0, :] = torch.tensor(observation, dtype=torch.float)

    for step in range(max_steps):
        # Use the policy to choose an action
        with torch.no_grad():
            start_ind = max(step - num_history_steps + 1, 0)

            end_ind = step + 1
            action_info = policy(
                observations[:, start_ind:end_ind],
                actions[:, start_ind:end_ind][:, :-1],
            )
            policy_action = action_info_to_action(action_info) if action_info_to_action else action_info
            actions[0, step] = policy_action

        # Step the environment and store the results
        action_step = action_to_env_action(actions[0, step]) if action_to_env_action else actions[0, step].cpu().numpy()
        observation, reward, terminated_step, truncated_step, info_step = env.step(
            action_step
        )
        observations[0, step + 1, :] = torch.tensor(observation, dtype=torch.float)
        rewards[0, step, 0] = reward
        terminated.append(torch.tensor(terminated_step, dtype=torch.bool))
        truncated.append(torch.tensor(truncated_step, dtype=torch.bool))

        if action_visualiser:
            info_step['action_vis'] = action_visualiser(action_info)

        # Convert info_step (a dictionary) into a tensor and add it to the info dictionary
        for key, value in info_step.items():
            if isinstance(value, str):
                continue
            if key not in info:
                info[key] = [torch.tensor(value, dtype=torch.float)]
            else:
                info[key].append(torch.tensor(value, dtype=torch.float))

        # If the environment has terminated, stop collecting the rollout
        if terminated_step or truncated_step:
            break

    # Concatenate the tensors in the info dictionary along the time dimension
    for key in info:
        info[key] = torch.stack(info[key], dim=0)

    # Concatenate the lists of terminated and truncated into tensors
    terminated = torch.tensor(terminated).unsqueeze(-1)
    truncated = torch.tensor(truncated).unsqueeze(-1)

    # Create a Rollout object and return it
    size = step + 1
    rollout = Rollout(
        observations[0][:size + 1],
        actions[0][:size],
        rewards[0][:size],
        terminated,
        truncated,
        info,
        size,
        rewards[0][:size].sum().item(),
    )
    return rollout
