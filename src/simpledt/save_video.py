import cv2
import numpy as np

from simpledt.rollout import Rollout


def save_video_from_images(images: np.ndarray, video_path: str, fps: int):
    """
    Save a series of images as a video using OpenCV.

    Args:
        images: A numpy array of shape (num_frames, height, width, channels)
        video_path: The path to save the video to.
        fps: The frame rate of the video.
    """
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    video_writer.release()


# def save_rollout_video(rollout: Rollout, path: str, fps: int = 24):
#     # data = rollout.observations.reshape(-1, 64, 60, 3) * 255
#     # data = data.numpy().clip(0, 255).astype(np.uint8)
#     # if 'action_vis' in rollout.info:
#     #     action_vis = rollout.info['action_vis'].numpy().astype(np.uint8)
#     #     data = np.concatenate([data[:-1], action_vis], axis=1)
#     save_video_from_images(
#         images=data,
#         video_path=path,
#         fps=fps,
#     )
