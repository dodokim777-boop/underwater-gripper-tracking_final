# -*- coding: utf-8 -*-
"""
⑧ 과정 잡음 σa 추정

연속 두 시점 관측 구간(4프레임 이상)의 축별 2차 차분에 대한 강건 표준편차(1.4826·MAD)를
frame 단위 σa로 쓴다. 추정 불가 축은 SIGMA_A_FALLBACK을 쓴다.
- build_cv_process_Q : σa로 등속 모델 Q를 만든다 (⑬ RTS에서 사용)
"""

import numpy as np

from .. import settings


def _robust_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def estimate_process_scale(df):
    """축별 2차 차분의 robust scale을 frame-unit σ_a로 추정한다."""
    valid = df[settings.MEAS_COLS].notna().all(axis=1).to_numpy()
    if 'Missing_Reason' in df.columns:
        valid &= df['Missing_Reason'].eq('NONE').to_numpy()
    if 'Meas_Status' in df.columns:
        valid &= df['Meas_Status'].eq('TRIANGULATED').to_numpy()
    frames = df['Frame'].to_numpy(dtype=float)
    run_ids = np.full(len(df), -1, dtype=int)
    run_id = -1
    previous_frame = None
    for i, is_valid in enumerate(valid):
        if not is_valid:
            previous_frame = None
            continue
        if previous_frame is None or frames[i] - previous_frame != 1:
            run_id += 1
        run_ids[i] = run_id
        previous_frame = frames[i]
    usable_runs = [
        rid for rid in range(run_id + 1)
        if len(np.flatnonzero(run_ids == rid)) >= 4
    ]
    sigma_a = np.full(3, np.nan, dtype=float)
    for axis_idx, col in enumerate(settings.MEAS_COLS):
        second_differences = []
        values = df[col].to_numpy(dtype=float)
        for rid in usable_runs:
            idx = np.flatnonzero(run_ids == rid)
            second_differences.extend(np.diff(values[idx], n=2).tolist())
        sigma_a[axis_idx] = _robust_std(second_differences)
    fallback_axes = ~np.isfinite(sigma_a) | (sigma_a <= 0.0)
    if fallback_axes.any():
        sigma_a[fallback_axes] = settings.SIGMA_A_FALLBACK[fallback_axes]
    return np.maximum(sigma_a, 1e-6), len(usable_runs), bool(fallback_axes.any())


def build_cv_process_Q(sigma_a, dt=1.0):
    """Q = G Σ_a G^T, G=[0.5 dt² I; dt I]."""
    sigma_a = np.asarray(sigma_a, dtype=float).reshape(3)
    Sigma_a = np.diag(sigma_a**2)
    return np.block([
        [0.25 * dt**4 * Sigma_a, 0.5 * dt**3 * Sigma_a],
        [0.5 * dt**3 * Sigma_a, dt**2 * Sigma_a],
    ])
