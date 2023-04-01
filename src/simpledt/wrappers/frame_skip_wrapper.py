from typing import Any, Tuple
import gym


class FrameSkipWrapper(gym.Wrapper):

    def __init__(self, env, frame_skip: int = 1):
        super().__init__(env)
        self._frame_skip = frame_skip

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        reward = 0
        for _ in range(self._frame_skip):
            observation, substep_reward, terminated_step, truncated_step, info_step = super().step(
                action
            )
            reward += substep_reward
            if terminated_step or truncated_step:
                break
        return observation, reward, terminated_step, truncated_step, info_step
