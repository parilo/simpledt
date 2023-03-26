import time
import torch
from simpledt.pygame_controller import PyGameController


MARIO_CONTROL = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
    ['B'],
]


class MarioKeyboardPolicy:

    def __init__(self, fps: int = 120) -> None:
        self._controller = PyGameController()
        self._dt = 1./fps

    def __call__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        self._controller.handle_events()
        keys = self._controller.get_keys()
        action_dist = torch.zeros(len(MARIO_CONTROL))

        if keys['right'] and keys['space'] and keys['k_z']:
            action_dist[4] = 1
        elif keys['right'] and keys['space']:
            action_dist[2] = 1
        elif keys['right'] and keys['k_z']:
            action_dist[3] = 1
        elif keys['right']:
            action_dist[1] = 1

        elif keys['left'] and keys['space'] and keys['k_z']:
            action_dist[9] = 1
        elif keys['left'] and keys['space']:
            action_dist[7] = 1
        elif keys['left'] and keys['k_z']:
            action_dist[8] = 1
        elif keys['left']:
            action_dist[6] = 1

        elif keys['space']:
            action_dist[5] = 1

        elif keys['down']:
            action_dist[10] = 1

        elif keys['up']:
            action_dist[11] = 1

        elif keys['k_z']:
            action_dist[12] = 1

        else:
            action_dist[0] = 1

        img = observations.numpy().reshape(240, 256, 3)
        self._controller.show(img)

        time.sleep(self._dt)

        return action_dist
