import gymnasium as gym
import torch
import numpy as np

from simpledt.models.dtpolicy import DTPolicy
from simpledt.rollout import Rollout


def collect_rollout(
    env: gym.Env,
    policy: DTPolicy,
    max_steps: int,
    history_steps: int = 1,
    exploration: float = 0,
) -> Rollout:
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    observations = torch.zeros(1, max_steps + 1, obs_size, dtype=torch.float)
    actions = torch.zeros(1, max_steps, action_size, dtype=torch.float)
    rewards = torch.zeros(1, max_steps, 1, dtype=torch.float)
    terminated = []
    truncated = []
    info = {}

    # Initialize the environment and get the first observation
    observation, _ = env.reset(
        options={
            "x_init": np.pi / 10,
            "y_init": 0.01,
        }
    )
    observations[0, 0, :] = torch.tensor(observation, dtype=torch.float)
    reward_to_go = torch.ones(1, max_steps, 1, dtype=torch.float)

    for step in range(max_steps):
        # Use the policy to choose an action
        with torch.no_grad():
            start_ind = max(step - history_steps + 1, 0)

            end_ind = step + 1
            # print(f'--- observations[:, start_ind:end_ind] {observations[:, start_ind:end_ind].shape}')
            # print(f'--- reward_to_go[:, start_ind:end_ind] {reward_to_go[:, start_ind:end_ind].shape}')
            # print(f'--- actions[:, start_ind:end_ind] {actions[:, start_ind:end_ind].shape}')
            policy_actions = policy(
                observations[:, start_ind:end_ind],
                reward_to_go[:, start_ind:end_ind],
                actions[:, start_ind:end_ind],
            )
            actions[0, step] = policy_actions[0, history_steps - 1]

        # Step the environment and store the results
        actions[0, step] += torch.randn_like(actions[0, step]) * exploration
        action_step = actions[0, step].cpu().numpy()
        observation, reward, terminated_step, truncated_step, info_step = env.step(
            action_step
        )
        # if step % 25 == 0:
        #     print(f'--- action {action_step}')
        observations[0, step + 1, :] = torch.tensor(observation, dtype=torch.float)
        rewards[0, step, 0] = reward
        terminated.append(torch.tensor(terminated_step, dtype=torch.bool))
        truncated.append(torch.tensor(truncated_step, dtype=torch.bool))

        # Convert info_step (a dictionary) into a tensor and add it to the info dictionary
        for key, value in info_step.items():
            if key not in info:
                info[key] = [torch.tensor(value, dtype=torch.float)]
            else:
                info[key].append(torch.tensor(value, dtype=torch.float))

        # If the environment has terminated, stop collecting the rollout
        if terminated_step or truncated_step:
            break

    # Concatenate the tensors in the info dictionary along the time dimension
    for key in info:
        info[key] = torch.cat(info[key], dim=0)

    # Concatenate the lists of terminated and truncated into tensors
    terminated = torch.tensor(terminated).unsqueeze(-1)
    truncated = torch.tensor(truncated).unsqueeze(-1)

    # Create a Rollout object and return it
    rollout = Rollout(
        observations[0], actions[0], rewards[0], terminated, truncated, info
    )
    return rollout
