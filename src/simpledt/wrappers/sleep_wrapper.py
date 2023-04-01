import time
import gym


class SleepWrapper(gym.ObservationWrapper):
    """
    Wrapper to insert sleep into env, for example to make keyboard control more convenient.
    """

    def __init__(self, env, dt: float):
        super().__init__(env)
        self._dt = dt

    def observation(self, obs):
        time.sleep(self._dt)
        return obs
