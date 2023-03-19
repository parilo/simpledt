import gym
import numpy as np


class FlattenImageObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to resize observation image in a Gym environment to a desired size.
    """

    def __init__(self, env):
        super().__init__(env)

        obs_shape = self.observation_space.shape

        # Update observation space to reflect resized image dimensions
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * obs_shape[1] * obs_shape[2],),
            dtype=np.uint8,
        )

    def observation(self, obs):
        """
        Resize observation image to desired size using OpenCV.
        """
        obs = obs.reshape(-1)
        return obs
