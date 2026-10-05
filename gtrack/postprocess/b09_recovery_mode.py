# -*- coding: utf-8 -*-
"""
⑨ 복원 모드 분류 · Z 하한 적용 상자 통과 판정

복원 모드
  NONE      : 두 시점 관측 또는 복원 대상 아님
  TOP_RAY   : FRONT_MISSING (상단만 검출) → 상단 광선 + Z 추정
  FRONT_RAY : TOP_MISSING   (정면만 검출) → 정면 광선 + Y 추정
Top_Zone_Supported : 상단 검출 중심의 굴절 광선이 그리퍼 작업영역 상자
                     (Z 하한 ZONE_TRACK_CENTER_Z_FLOOR ~ ZONE_Z_HIGH)를 통과하는지
출력: State2, Recovery_Mode, Recovery_Run_ID, Top_Zone_Supported
"""

import numpy as np
import pandas as pd

from .. import settings
from ..geometry import _camera_ray_world, _refract_at_plane
from .geometry import _ray_intersects_axis_aligned_box


def _finite_pair(df, u_col, v_col):
    u = pd.to_numeric(df[u_col], errors='coerce').to_numpy(dtype=float)
    v = pd.to_numeric(df[v_col], errors='coerce').to_numpy(dtype=float)
    return np.isfinite(u) & np.isfinite(v)


def classify_states(df):
    """복원 근거에 따라 TOP_RAY/FRONT_RAY를 명시적으로 분리한다.

    Track_State는 장기 단일시점 구간에서 MAX_3D_GAP에 의해 reset될 수 있으므로
    복원 자격 조건으로 사용하지 않는다. 대신 살아 있는 카메라의 DETECTED 상태,
    유효한 중심 pixel, 명확한 Missing_Reason을 모두 요구한다.
    """
    n = len(df)
    raw_valid = df[settings.MEAS_COLS].notna().all(axis=1).to_numpy()
    top_ok = df['Status_Top'].eq('DETECTED').to_numpy()
    front_ok = df['Status_Front'].eq('DETECTED').to_numpy()
    top_lost = df['Status_Top'].eq('LOST').to_numpy()
    front_lost = df['Status_Front'].eq('LOST').to_numpy()
    top_pixel_ok = _finite_pair(df, 'Top_px_u', 'Top_px_v')
    front_pixel_ok = _finite_pair(df, 'Front_px_u', 'Front_px_v')
    reason = df['Missing_Reason'].fillna('NONE').astype(str).to_numpy()

    recovery_mode = np.full(n, settings.RECOVERY_NONE, dtype=object)
    top_ray = (
        (~raw_valid)
        & top_ok
        & front_lost
        & top_pixel_ok
        & (reason == 'FRONT_MISSING')
    )
    front_ray = (
        (~raw_valid)
        & front_ok
        & top_lost
        & front_pixel_ok
        & (reason == 'TOP_MISSING')
    )
    recovery_mode[top_ray] = settings.RECOVERY_TOP_RAY
    recovery_mode[front_ray] = settings.RECOVERY_FRONT_RAY

    state = np.full(n, 'LOST', dtype=object)
    state[raw_valid] = 'DETECTED'
    state[recovery_mode != settings.RECOVERY_NONE] = 'SUB'

    run_id = np.full(n, -1, dtype=int)
    next_id = 0
    idx = 0
    while idx < n:
        mode = recovery_mode[idx]
        if mode == settings.RECOVERY_NONE:
            idx += 1
            continue
        end = idx
        while end + 1 < n and recovery_mode[end + 1] == mode:
            end += 1
        run_id[idx:end + 1] = next_id
        next_id += 1
        idx = end + 1

    df['State2'] = state
    df['Recovery_Mode'] = recovery_mode
    df['Recovery_Run_ID'] = run_id
    return df


def _top_pixel_ray_intersects_gripper_zone(pixel, calib_top):
    """top 검출 중심의 굴절 광선이 motor-origin gripper zone을 통과하는지 판정한다."""
    C_world, d_world = _camera_ray_world(pixel, calib_top)
    refracted = _refract_at_plane(C_world, d_world, settings.INTERFACE_TOP)
    if refracted is None:
        return False

    P_world, d_ref_world = refracted
    P_motor = P_world - settings.MOTOR_CENTER_IN_ID0_CM

    box_min = np.array([
        -settings.ZONE_X_HALF,
        -settings.ZONE_Y_HALF,
        settings.ZONE_TRACK_CENTER_Z_FLOOR,
    ], dtype=float)
    box_max = np.array([
        settings.ZONE_X_HALF,
        settings.ZONE_Y_HALF,
        settings.ZONE_Z_HIGH,
    ], dtype=float)
    return _ray_intersects_axis_aligned_box(
        P_motor, d_ref_world, box_min, box_max
    )


def classify_top_zone_support(df, calib_top):
    """각 frame의 top 검출 중심이 gripper-zone 투영 부피 안에 있는지 기록한다.

    고정 pixel polygon을 하드코딩하지 않고, 현재 top calibration과 굴절 광선을
    사용한다. 이 값은 새 상태가 아니라 장기 FRONT_MISSING 복원의 근거 기록이다.
    """
    if calib_top is None:
        raise RuntimeError('Top calibration is required for top-zone support classification.')

    n = len(df)
    supported = np.zeros(n, dtype=bool)
    top_detected = df['Status_Top'].eq('DETECTED').to_numpy()
    top_u = pd.to_numeric(df['Top_px_u'], errors='coerce').to_numpy(dtype=float)
    top_v = pd.to_numeric(df['Top_px_v'], errors='coerce').to_numpy(dtype=float)
    candidates = top_detected & np.isfinite(top_u) & np.isfinite(top_v)

    for idx in np.flatnonzero(candidates):
        supported[idx] = _top_pixel_ray_intersects_gripper_zone(
            (top_u[idx], top_v[idx]), calib_top
        )

    df['Top_Zone_Supported'] = supported
    return df
