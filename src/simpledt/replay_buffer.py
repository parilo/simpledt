import torch
import numpy as np

from simpledt.rollout import BatchOfSeq, Rollout


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        rollout_len: int,
        observation_shape: dict,
        action_shape: tuple,
        info_shape: dict,
    ):
        self.max_size = max_size
        self.observation_keys = list(observation_shape.keys())
        self.observation_shapes = list(observation_shape.values())
        self.action_shape = action_shape
        self.info_keys = list(info_shape.keys())
        self.info_shapes = list(info_shape.values())
        self.observations = {
            key: torch.empty((max_size, rollout_len + 1, *shape))
            for key, shape in observation_shape.items()
        }
        self.actions = torch.empty((max_size, rollout_len, *action_shape))
        self.rewards = torch.empty((max_size, rollout_len, 1))
        self.terminated = torch.empty(
            (max_size, rollout_len, 1), dtype=torch.bool
        )  # fixed
        self.truncated = torch.empty(
            (max_size, rollout_len, 1), dtype=torch.bool
        )  # fixed
        self.info = {
            key: torch.empty((max_size, rollout_len, *shape))
            for key, shape in info_shape.items()
        }
        self.sizes = torch.zeros(self.max_size)
        self.size = 0
        self.current_index = 0

    def add_rollout(self, rollout: Rollout):
        size = rollout.size
        for key in self.observation_keys:
            self.observations[key][self.current_index][:size + 1] = rollout.observations[key]
        self.actions[self.current_index][:size] = rollout.actions
        self.rewards[self.current_index][:size] = rollout.rewards
        self.terminated[self.current_index][:size] = rollout.terminated
        self.truncated[self.current_index][:size] = rollout.truncated
        for key in self.info_keys:
            self.info[key][self.current_index][:size] = rollout.info[key]
        self.sizes[self.current_index] = size
        self.size = min(self.size + 1, self.max_size)
        self.current_index = (self.current_index + 1) % self.max_size

    def sample(self, batch_size: int):
        indices = torch.randperm(self.size)[:batch_size]
        observations = {
            key: self.observations[key][indices] for key in self.observation_keys
        }
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        # terminated = self.terminated[indices]
        # truncated = self.truncated[indices]
        # info = {key: self.info[key][indices] for key in self.info_keys}
        return BatchOfSeq(observations, actions, rewards)

    def sample_batch_of_seqs(self, batch_size: int, seq_len: int):
        rollouts = np.random.choice(self.size, size=batch_size, replace=True)
        batch_obs = {}
        for key in self.observation_keys:
            batch_obs[key] = []
        batch_act = []
        batch_rew = []
        batch_term = []
        for i in rollouts:
            start_idx = np.random.randint(
                # self.observations[self.observation_keys[0]].shape[1] - seq_len
                self.sizes[i] - seq_len
            )
            obs = {key: self.observations[key][i, start_idx : start_idx + seq_len + 1] for key in self.observation_keys}
            act = self.actions[i, start_idx : start_idx + seq_len]
            rew = self.rewards[i, start_idx : start_idx + seq_len]
            term = self.terminated[i, start_idx : start_idx + seq_len]
            batch_act.append(act)
            batch_rew.append(rew)
            batch_term.append(term)
            for key in self.observation_keys:
                batch_obs[key].append(obs[key])
        batch_obs = {key: torch.stack(obs_list) for key, obs_list in batch_obs.items()}
        batch_act = torch.stack(batch_act)
        batch_rew = torch.stack(batch_rew)
        batch_term = torch.stack(batch_term)
        return BatchOfSeq(batch_obs, batch_act, batch_rew, batch_term)

    # def sample_batch_of_seqs(self, batch_size: int, seq_len: int):
    #     indices = torch.randperm(self.size - seq_len + 1)[:batch_size]
    #     end_indices = indices + seq_len
    #     observations = {
    #         key: torch.stack(
    #             [
    #                 self.observations[key][start_idx:end_idx]
    #                 for start_idx, end_idx in zip(indices, end_indices)
    #             ]
    #         )
    #         for key in self.observation_keys
    #     }
    #     actions = torch.stack(
    #         [
    #             self.actions[start_idx:end_idx]
    #             for start_idx, end_idx in zip(indices, end_indices)
    #         ]
    #     )
    #     rewards = torch.stack(
    #         [
    #             self.rewards[start_idx:end_idx]
    #             for start_idx, end_idx in zip(indices, end_indices)
    #         ]
    #     )
    #     terminated = torch.stack(
    #         [
    #             self.terminated[start_idx:end_idx]
    #             for start_idx, end_idx in zip(indices, end_indices)
    #         ]
    #     )
    #     truncated = torch.stack(
    #         [
    #             self.truncated[start_idx:end_idx]
    #             for start_idx, end_idx in zip(indices, end_indices)
    #         ]
    #     )
    #     info = {
    #         key: torch.stack(
    #             [
    #                 self.info[key][start_idx:end_idx]
    #                 for start_idx, end_idx in zip(indices, end_indices)
    #             ]
    #         )
    #         for key in self.info_keys
    #     }
    #     return BatchOfSeq(observations, actions, rewards, terminated, truncated, info)

    def get_content(self) -> BatchOfSeq:
        return BatchOfSeq(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
            info=self.info,
        )

    def set_content(self, data: BatchOfSeq):
        for key in self.observation_keys:
            self.observations[key][:] = data.observations[key]
        self.actions[:] = data.actions
        self.rewards[:] = data.rewards
        self.terminated[:] = data.terminated
        self.truncated[:] = data.truncated
        for key in self.info_keys:
            self.info[key][:] = data.info[key]
        self.size = self.actions.shape[0]  # bug? if episode size is variable
