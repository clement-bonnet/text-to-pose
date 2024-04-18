from typing import Optional

import numpy as np
from controlnet_aux.dwpose.util import draw_bodypose, draw_facepose, draw_handpose


def draw_pose(
    pose: dict,
    height: int,
    width: int,
    canvas: Optional[np.ndarray] = None,
):
    bodies = pose["bodies"]
    if canvas is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, bodies["candidate"], bodies["subset"])
    canvas = draw_handpose(canvas, pose["hands"])
    canvas = draw_facepose(canvas, pose["faces"])
    return canvas
