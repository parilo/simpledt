import gym
import numpy as np


class ScaleObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to resize observation image in a Gym environment to a desired size.
    """

    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self._scale = scale

        # Update observation space to reflect resized image dimensions
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        """
        Resize observation image to desired size using OpenCV.
        """
        return obs.astype(np.float32) * self._scale
