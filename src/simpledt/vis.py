import numpy as np
import cv2
from typing import Tuple

import torch


def draw_prob_histogram_on_image(
    probs: np.ndarray,
    size: Tuple[int, int],
    color=(0, 0, 255)
):
    image = np.zeros(tuple(size) + (3,), dtype=np.uint8)
    hist = probs * image.shape[1]
    num_bins = len(probs)
    # Draw the histogram on the image
    bin_width = image.shape[0] // num_bins
    for i in range(num_bins):
        x1 = i * bin_width
        x2 = (i + 1) * bin_width
        y = int(image.shape[1] - hist[i])
        cv2.rectangle(image, (x1, y), (x2, image.shape[1]), color, -1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def add_action_vis(imgs: np.ndarray, actions: torch.Tensor, size: Tuple[int, int]):
    actions_vis = []
    for i in range(len(actions)):
        action_vis = draw_prob_histogram_on_image(
            probs=actions[i],
            size=size,
        )
        actions_vis.append(action_vis)

    actions_vis = np.stack(actions_vis, axis=0)
    return np.concatenate([imgs, actions_vis], axis=1)
