import functools
import unittest
import torch

from simpledt.replay_buffer import ReplayBuffer
from simpledt.rollout import Rollout


class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.observation_shape = {"obs1": (10,), "obs2": (20,)}
        self.action_shape = (5,)
        self.info_shape = {"info1": (7,), "info2": (3,)}
        self.max_size = 3
        self.rollout_len = 15
        self.batch_size = 2

        self.rollout1 = Rollout(
            observations={"obs1": torch.rand(self.rollout_len + 1, 10), "obs2": torch.rand(self.rollout_len + 1, 20)},
            actions=torch.rand(self.rollout_len, 5),
            rewards=torch.rand(self.rollout_len, 1),
            terminated=torch.rand(self.rollout_len, 1) > 0.5,
            truncated=torch.rand(self.rollout_len, 1) > 0.5,
            info={"info1": torch.rand(self.rollout_len, 7), "info2": torch.rand(self.rollout_len, 3)},
        )
        self.rollout2 = Rollout(
            observations={"obs1": torch.rand(self.rollout_len + 1, 10), "obs2": torch.rand(self.rollout_len + 1, 20)},
            actions=torch.rand(self.rollout_len, 5),
            rewards=torch.rand(self.rollout_len, 1),
            terminated=torch.rand(self.rollout_len, 1) > 0.5,
            truncated=torch.rand(self.rollout_len, 1) > 0.5,
            info={"info1": torch.rand(self.rollout_len, 7), "info2": torch.rand(self.rollout_len, 3)},
        )
        self.rollout3 = Rollout(
            observations={"obs1": torch.rand(self.rollout_len + 1, 10), "obs2": torch.rand(self.rollout_len + 1, 20)},
            actions=torch.rand(self.rollout_len, 5),
            rewards=torch.rand(self.rollout_len, 1),
            terminated=torch.rand(self.rollout_len, 1) > 0.5,
            truncated=torch.rand(self.rollout_len, 1) > 0.5,
            info={"info1": torch.rand(self.rollout_len, 7), "info2": torch.rand(self.rollout_len, 3)},
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=self.max_size,
            rollout_len=self.rollout_len,
            observation_shape=self.observation_shape,
            action_shape=self.action_shape,
            info_shape=self.info_shape,
        )

    def test_add_rollout_and_sample(self):
        # Add the rollouts to the replay buffer
        self.replay_buffer.add_rollout(self.rollout1)
        self.replay_buffer.add_rollout(self.rollout2)
        self.replay_buffer.add_rollout(self.rollout3)

        # Sample a batch of rollouts from the replay buffer
        (
            observations,
            actions,
            rewards,
            terminated,
            truncated,
            info,
        ) = self.replay_buffer.sample(self.batch_size)

        # Check shapes of sampled data
        self.assertEqual(observations["obs1"].shape, (self.batch_size, self.rollout_len + 1, 10))
        self.assertEqual(observations["obs2"].shape, (self.batch_size, self.rollout_len + 1, 20))
        self.assertEqual(actions.shape, (self.batch_size, self.rollout_len, 5))
        self.assertEqual(rewards.shape, (self.batch_size, self.rollout_len, 1))
        self.assertEqual(terminated.shape, (self.batch_size, self.rollout_len, 1))
        self.assertEqual(truncated.shape, (self.batch_size, self.rollout_len, 1))
        self.assertEqual(info["info1"].shape, (self.batch_size, self.rollout_len, 7))
        self.assertEqual(info["info2"].shape, (self.batch_size, self.rollout_len, 3))

        # Check keys in dictionaries
        self.assertEqual(set(observations.keys()), set(self.observation_shape.keys()))
        self.assertEqual(set(info.keys()), set(self.info_shape.keys()))

    def test_size(self):
        # Check initial size of replay buffer
        self.assertEqual(self.replay_buffer.size, 0)

        # Add the rollouts to the replay buffer
        self.replay_buffer.add_rollout(self.rollout1)
        self.replay_buffer.add_rollout(self.rollout2)
        self.replay_buffer.add_rollout(self.rollout3)

        # Check size of replay buffer after adding 3 rollouts
        self.assertEqual(self.replay_buffer.size, 3)

    def test_contents(self):
        # Add the rollouts to the replay buffer
        self.replay_buffer.add_rollout(self.rollout1)
        self.replay_buffer.add_rollout(self.rollout2)
        self.replay_buffer.add_rollout(self.rollout3)

        # Check contents of replay buffer
        assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        assert_equal(
            self.replay_buffer.observations,
            {
                "obs1": torch.stack(
                    [
                        self.rollout1.observations["obs1"],
                        self.rollout2.observations["obs1"],
                        self.rollout3.observations["obs1"],
                    ]
                ),
                "obs2": torch.stack(
                    [
                        self.rollout1.observations["obs2"],
                        self.rollout2.observations["obs2"],
                        self.rollout3.observations["obs2"],
                    ]
                ),
            },
        )
        assert_equal(
            self.replay_buffer.actions,
            torch.stack(
                [self.rollout1.actions, self.rollout2.actions, self.rollout3.actions]
            ),
        )
        assert_equal(
            self.replay_buffer.rewards,
            torch.stack(
                [self.rollout1.rewards, self.rollout2.rewards, self.rollout3.rewards]
            ),
        )
        assert_equal(
            self.replay_buffer.terminated,
            torch.stack(
                [
                    self.rollout1.terminated,
                    self.rollout2.terminated,
                    self.rollout3.terminated,
                ]
            ),
        )
        assert_equal(
            self.replay_buffer.truncated,
            torch.stack(
                [
                    self.rollout1.truncated,
                    self.rollout2.truncated,
                    self.rollout3.truncated,
                ]
            ),
        )
        assert_equal(
            self.replay_buffer.info,
            {
                "info1": torch.stack(
                    [
                        self.rollout1.info["info1"],
                        self.rollout2.info["info1"],
                        self.rollout3.info["info1"],
                    ]
                ),
                "info2": torch.stack(
                    [
                        self.rollout1.info["info2"],
                        self.rollout2.info["info2"],
                        self.rollout3.info["info2"],
                    ]
                ),
            },
        )
