import gym
import numpy as np
import cv2


class ResizeObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to resize observation image in a Gym environment to a desired size.
    """

    def __init__(self, env, size):
        super().__init__(env)
        self.size = size

        # Update observation space to reflect resized image dimensions
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.size[0], self.size[1], self.observation_space.shape[2]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        """
        Resize observation image to desired size using OpenCV.
        """
        resized_obs = cv2.resize(obs, self.size[::-1], interpolation=cv2.INTER_AREA)
        return resized_obs
