"""Input: read 3D_raw_online.xlsx and the calibration NPZ files."""

import cv2
import numpy as np
import pandas as pd

from .. import settings
from ..common.file_utils import _npz_value


def _build_calib(R_cm, t_cm, K, dist, marker_to_world=None, metadata=None):
    R_cm = np.asarray(R_cm, dtype=float).reshape(3, 3)
    t_cm = np.asarray(t_cm, dtype=float).reshape(3, 1)
    if marker_to_world is None:
        R_w, t_w = R_cm, t_cm
    else:
        R_fw, t_fw = marker_to_world
        R_fw = np.asarray(R_fw, dtype=float).reshape(3, 3)
        t_fw = np.asarray(t_fw, dtype=float).reshape(3, 1)
        R_w = R_cm @ R_fw
        t_w = R_cm @ t_fw + t_cm

    rvec, _ = cv2.Rodrigues(R_w)
    calib = {
        'rvec': rvec,
        'tvec': t_w,
        'cam_matrix': np.asarray(K, dtype=float).reshape(3, 3),
        'dist_coeffs': np.asarray(dist, dtype=float).reshape(1, -1),
    }
    calib.update(metadata or {})
    return calib


def load_calib_top(path=None):
    if path is None:
        path = settings.MANUAL_CALIB_TOP_PATH
    try:
        data = np.load(path, allow_pickle=False)
        if 'R' not in data.files or 't' not in data.files:
            raise KeyError('NPZ must contain R and t')
        stored_model = str(_npz_value(data, ('camera_model',), settings.TOP_CAMERA_MODEL))
        metadata = {
            'camera_role': 'top',
            'camera_model': stored_model,
            'projection_mode': str(_npz_value(data, ('projection_mode',), 'unknown')),
            'intrinsic_rms_px': _npz_value(
                data, ('intrinsic_rms_px', 'rms', 'reprojection_rms_px'), np.nan
            ),
            'intrinsic_source': f'hardcoded: {settings.TOP_CAMERA_MODEL}',
        }
        calib = _build_calib(
            data['R'], data['t'], settings.K_MATRIX_TOP, settings.DIST_COEFFS_TOP,
            marker_to_world=None, metadata=metadata,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Top dry-calibration NPZ required: {path}') from exc
    except Exception as exc:
        raise RuntimeError(f'Failed to load top calibration NPZ: {path}: {exc}') from exc
    print(
        f"top calib: {path} | model={calib.get('camera_model', settings.TOP_CAMERA_MODEL)} | "
        f"intrinsic={calib.get('intrinsic_source', 'hardcoded')}"
    )
    return calib


def load_calib_front(path=None):
    if path is None:
        path = settings.MANUAL_CALIB_FRONT_PATH
    try:
        data = np.load(path, allow_pickle=False)
        if 'R' not in data.files or 't' not in data.files:
            raise KeyError('NPZ must contain R and t')
        use_wall = bool(_npz_value(data, ('use_wall_transform',), False))
        marker_to_world = (settings.R_FW, settings.t_FW) if use_wall else None
        stored_model = str(_npz_value(data, ('camera_model',), settings.FRONT_CAMERA_MODEL))
        metadata = {
            'camera_role': 'front',
            'camera_model': stored_model,
            'projection_mode': str(_npz_value(data, ('projection_mode',), 'unknown')),
            'intrinsic_rms_px': _npz_value(
                data, ('intrinsic_rms_px', 'rms', 'reprojection_rms_px'), np.nan
            ),
            'intrinsic_source': f'hardcoded: {settings.FRONT_CAMERA_MODEL}',
            'use_wall_transform': use_wall,
        }
        calib = _build_calib(
            data['R'], data['t'], settings.K_MATRIX_FRONT, settings.DIST_COEFFS_FRONT,
            marker_to_world=marker_to_world, metadata=metadata,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Front dry-calibration NPZ required: {path}') from exc
    except Exception as exc:
        raise RuntimeError(f'Failed to load front calibration NPZ: {path}: {exc}') from exc
    print(
        f"front calib: {path} | model={calib.get('camera_model', settings.FRONT_CAMERA_MODEL)} | "
        f"use_wall_transform={calib.get('use_wall_transform', False)} | "
        f"intrinsic={calib.get('intrinsic_source', 'hardcoded')}"
    )
    return calib


def read_raw_trajectory(raw_path):
    try:
        df = pd.read_excel(raw_path, sheet_name='Trajectory')
    except ValueError:
        df = pd.read_excel(raw_path)
    df = df[[c for c in df.columns if not str(c).startswith('｜')]].reset_index(drop=True)
    df = df.rename(columns=settings.RAW_COLUMN_RENAME_INVERSE)
    return df, set(df.columns)


def ensure_missing_reason(df):
    if 'Missing_Reason' in df.columns:
        df['Missing_Reason'] = df['Missing_Reason'].fillna('NONE').astype(str)
        return df

    reason = np.array(['NONE'] * len(df), dtype=object)
    has_meas = df[settings.MEAS_COLS].notna().all(axis=1).to_numpy()
    top_ok = df.get('Status_Top', pd.Series(['LOST'] * len(df))).eq('DETECTED').to_numpy()
    front_ok = df.get('Status_Front', pd.Series(['LOST'] * len(df))).eq('DETECTED').to_numpy()
    reason[(~has_meas) & (~top_ok) & (~front_ok)] = 'BOTH_MISSING'
    reason[(~has_meas) & (~top_ok) & front_ok] = 'TOP_MISSING'
    reason[(~has_meas) & top_ok & (~front_ok)] = 'FRONT_MISSING'
    reason[(~has_meas) & top_ok & front_ok] = 'TRIANGULATION_FAILED'
    if 'Meas_Status' in df.columns:
        status_map = {
            'REJECTED_JUMP': 'REJECTED_JUMP',
            'REJECTED_GATE': 'REJECTED_MAHALANOBIS',
            'REJECTED_MAHALANOBIS': 'REJECTED_MAHALANOBIS',
            'TENTATIVE_HOLD': 'TENTATIVE_HOLD',
            'REJECT_STREAK_RESET': 'REJECT_STREAK_RESET',
        }
        for status, mapped in status_map.items():
            mask = df['Meas_Status'].eq(status).to_numpy()
            reason[mask] = mapped
    df['Missing_Reason'] = reason
    return df
