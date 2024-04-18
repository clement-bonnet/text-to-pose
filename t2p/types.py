from typing import TypedDict

import numpy as np


class PoseBodiesDict(TypedDict):
    candidate: np.ndarray
    subset: np.ndarray


class PoseDict(TypedDict):
    bodies: PoseBodiesDict
    faces: np.ndarray
    hands: np.ndarray
