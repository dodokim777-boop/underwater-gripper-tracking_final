# -*- coding: utf-8 -*-
"""
⑤ 그리퍼 중심 좌표 변환

world(바닥 ID0 마커 원점) 좌표에서 모터 중심 좌표 MOTOR_CENTER_IN_ID0_CM를 빼서
원점만 모터 중심으로 옮긴다 (축 방향은 ID0와 같음).
출력(raw 엑셀 이름): Meas_X_cm / Meas_Y_cm / Meas_Z_cm  (Candidate_*, KF_* 열도 같은 변환)
"""

import numpy as np

from .. import settings


def to_motor_origin_frame(pos_world):
    if pos_world is None or np.any(np.isnan(pos_world)):
        return np.full(3, np.nan)
    return np.asarray(pos_world, dtype=float) - settings.MOTOR_CENTER_IN_ID0_CM
