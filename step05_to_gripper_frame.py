"""Step ⑤: shift world coordinates so that the origin is the gripper (motor) center."""

import numpy as np

from .. import settings


def to_motor_origin_frame(pos_world):
    if pos_world is None or np.any(np.isnan(pos_world)):
        return np.full(3, np.nan)
    return np.asarray(pos_world, dtype=float) - settings.MOTOR_CENTER_IN_ID0_CM
