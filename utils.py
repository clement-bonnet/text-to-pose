from typing import Optional

from PIL import Image
import numpy as np
from controlnet_aux.dwpose.util import draw_bodypose, draw_facepose, draw_handpose


def draw_pose(
    pose: dict,
    height: int,
    width: int,
    canvas: Optional[np.ndarray] = None,
    only_body: bool = False,
):
    bodies = pose["bodies"]
    if canvas is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, bodies["candidate"], bodies["subset"])
    if not only_body:
        canvas = draw_handpose(canvas, pose["hands"])
        canvas = draw_facepose(canvas, pose["faces"])
    return canvas


def draw_pose_pil_center_crop(
    pose: dict,
    pose_hw_ratio: float,
    resolution: int = 1024,
    canvas: Optional[np.ndarray] = None,
    only_body: bool = False,
) -> Image.Image:
    if pose_hw_ratio > 1:
        height = int(resolution * pose_hw_ratio)
        width = resolution
    else:
        height = resolution
        width = int(resolution / pose_hw_ratio)
    pose_image = Image.fromarray(
        draw_pose(pose, height=height, width=width, canvas=canvas, only_body=only_body)
    )

    left = (width - resolution) / 2
    top = (height - resolution) / 2
    right = (width + resolution) / 2
    bottom = (height + resolution) / 2
    return pose_image.crop((left, top, right, bottom))
