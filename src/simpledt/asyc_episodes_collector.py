import gym
from gym.vector.async_vector_env import AsyncVectorEnv
import numpy as np

class EpisodeCollector:
    def __init__(self, env_name, num_envs):
        self.envs = AsyncVectorEnv([gym.make(env_name) for _ in range(num_envs)])
        self.num_envs = num_envs

    def collect_episodes(self, num_episodes):
        episodes = []
        episode_rewards = np.zeros(self.num_envs)
        for i in range(num_episodes):
            # reset environments
            obs = self.envs.reset()

            while True:
                # take a step in each environment
                actions = [self.envs.action_space.sample() for _ in range(self.num_envs)]
                obs, rewards, dones, infos = self.envs.step_async(actions)

                # update episode rewards
                episode_rewards += rewards

                # check if any environment is done
                for j, done in enumerate(dones):
                    if done:
                        episode = {'obs': [], 'actions': [], 'rewards': []}
                        for t in range(len(infos[j]['episode']['rewards'])):
                            episode['obs'].append(infos[j]['episode']['obs'][t])
                            episode['actions'].append(infos[j]['episode']['actions'][t])
                            episode['rewards'].append(infos[j]['episode']['rewards'][t])
                        episode['total_reward'] = episode_rewards[j]
                        episodes.append(episode)
                        episode_rewards[j] = 0

                # break out of loop if all environments are done
                if all(dones):
                    break

        return episodes

    def close_envs(self):
        self.envs.close()



import pytest
import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from episode_collector import EpisodeCollector

@pytest.fixture(scope="module")
def env_name():
    return "CartPole-v1"

@pytest.fixture(scope="module")
def num_envs():
    return 2

@pytest.fixture(scope="module")
def num_episodes():
    return 2

def test_episode_collector(env_name, num_envs, num_episodes):
    # create instance of episode collector
    collector = EpisodeCollector(env_name, num_envs)

    # collect episodes
    episodes = collector.collect_episodes(num_episodes)

    # check if correct number of episodes are collected
    assert len(episodes) == num_episodes

    # check if episodes are dictionaries
    assert all(isinstance(episode, dict) for episode in episodes)

    # check if episode dictionary keys are correct
    episode_keys = ['obs', 'actions', 'rewards', 'total_reward']
    assert all(set(episode.keys()) == set(episode_keys) for episode in episodes)

    # check if episode observation, action, and reward sequences have correct length
    for episode in episodes:
        assert len(episode['obs']) == len(episode['actions']) == len(episode['rewards'])

    # close environment
    collector.close_envs()
