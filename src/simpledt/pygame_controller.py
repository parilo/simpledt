import pygame
import cv2
import numpy as np
from pygame.locals import QUIT, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_z


class PyGameController:
    def __init__(self):
        pygame.init()

        # Set up the Pygame window
        self._screen_size = (640, 480)
        self._screen = pygame.display.set_mode(self._screen_size)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()

    def get_keys(self):
        keys = pygame.key.get_pressed()
        return {
            "up": keys[K_UP],
            "down": keys[K_DOWN],
            "left": keys[K_LEFT],
            "right": keys[K_RIGHT],
            "space": keys[K_SPACE],
            "k_z": keys[K_z],
        }

    def show(self, img):
        img = np.transpose(img, [1, 0, 2])
        img = cv2.resize(img, self._screen_size[::-1])
        surface = pygame.surfarray.make_surface(img.astype(np.uint8))
        self._screen.blit(surface, (0, 0))
        pygame.display.update()
