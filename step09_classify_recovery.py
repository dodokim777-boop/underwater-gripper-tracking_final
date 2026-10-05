"""Step ⑨: recovery mode (TOP_RAY / FRONT_RAY) and top-ray work-zone box test."""

import numpy as np
import pandas as pd

from .. import settings
from ..common.refraction import _camera_ray_world, _refract_at_plane
from .ray_geometry import _ray_intersects_axis_aligned_box


def _finite_pair(df, u_col, v_col):
    u = pd.to_numeric(df[u_col], errors='coerce').to_numpy(dtype=float)
    v = pd.to_numeric(df[v_col], errors='coerce').to_numpy(dtype=float)
    return np.isfinite(u) & np.isfinite(v)


def classify_states(df):
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
