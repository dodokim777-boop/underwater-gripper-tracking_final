# -*- coding: utf-8 -*-
"""
⑬ 표시용 3차원 RTS 평활화 (Plot_Aux 전용)

두 시점 관측 구간만 3차원 등속 RTS로 평활화하고, 복원 프레임의 X·Y·Z는 그대로 보존한다.
GRAB/LOST와 표시용 연결 프레임은 입력에서 뺀다.
출력: X_vis_cm / Y_vis_cm / Z_vis_cm, Vis_Std_*_cm (Plot_Aux)
      코드 내부 이름은 X_cm / Y_cm / Z_cm, RTS_Std_* 이다 (Result의 X_cm과 다름, docs/blocks.md 참고)
"""

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

from .. import settings
from .geometry import _in_zone_3d
from .b08_process_noise import build_cv_process_Q


def _measurement_class(filled_type):
    """RTS measurement source를 raw/top-ray/front-ray로 분리한다."""
    value = str(filled_type)
    if value == 'NONE':
        return 'RAW_TRIANGULATED'
    if 'TOP_RAY' in value:
        return 'RECONSTRUCTED_TOP_RAY'
    if 'FRONT_RAY' in value:
        return 'RECONSTRUCTED_FRONT_RAY'
    raise RuntimeError(f'Unknown Filled_Type for RTS: {value}')


def _measurement_R_and_sigma(filled_type):
    source = _measurement_class(filled_type)
    if source == 'RAW_TRIANGULATED':
        sigma = settings.SIG_RTS_RAW
    elif source == 'RECONSTRUCTED_TOP_RAY':
        sigma = settings.SIG_RTS_RECON_TOP
    elif source == 'RECONSTRUCTED_FRONT_RAY':
        sigma = settings.SIG_RTS_RECON_FRONT
    else:  # pragma: no cover - _measurement_class에서 이미 차단
        raise RuntimeError(f'Unknown RTS measurement source: {source}')
    return np.diag(sigma**2), sigma, source


def _rts_valid_runs(
    valid_mask, frames, measurement_class, stable_raw_anchor
):
    """RTS run과 hard boundary를 만든다.

    좌표 결측/GRAB/LOST 및 Frame 단절에서는 항상 분리한다. TOP_RAY와 FRONT_RAY
    사이에는 trusted raw stereo anchor가 존재할 때만 같은 RTS segment를 허용한다.
    실제 Hampel-filtered raw stereo frame은 sparse-anchor 정책과 동일하게 mode 간 직접 관측 근거가 된다.
    """
    n = len(valid_mask)
    runs = []
    hard_boundary = np.zeros(n, dtype=bool)
    segment_id = np.full(n, -1, dtype=int)
    next_segment_id = 0

    idx = 0
    while idx < n:
        if not valid_mask[idx]:
            idx += 1
            continue

        start = idx
        end = idx
        active_recovery = None
        first_class = measurement_class[idx]
        if first_class in {
            'RECONSTRUCTED_TOP_RAY',
            'RECONSTRUCTED_FRONT_RAY',
        }:
            active_recovery = first_class
        elif first_class == 'RAW_TRIANGULATED' and stable_raw_anchor[idx]:
            active_recovery = None

        while end + 1 < n and valid_mask[end + 1]:
            nxt = end + 1
            if frames[nxt] - frames[end] != 1:
                hard_boundary[nxt] = True
                break

            next_class = measurement_class[nxt]
            if next_class == 'RAW_TRIANGULATED':
                # 실제 raw stereo 관측이 들어오면 앞선 복원 mode 문맥을 끊는다.
                if stable_raw_anchor[nxt]:
                    active_recovery = None
                end = nxt
                continue

            if next_class not in {
                'RECONSTRUCTED_TOP_RAY',
                'RECONSTRUCTED_FRONT_RAY',
            }:
                raise RuntimeError(
                    f'Unexpected RTS measurement class: {next_class}'
                )

            if active_recovery is None:
                active_recovery = next_class
                end = nxt
                continue

            if next_class != active_recovery:
                hard_boundary[nxt] = True
                break

            end = nxt

        runs.append((start, end))
        segment_id[start:end + 1] = next_segment_id
        next_segment_id += 1
        idx = end + 1

    return runs, hard_boundary, segment_id


def rts_smooth(df, sigma_a):
    for col in [
        'X_rts', 'Y_rts', 'Z_rts',
        'RTS_Std_X', 'RTS_Std_Y', 'RTS_Std_Z',
        'Assigned_Sigma_X', 'Assigned_Sigma_Y', 'Assigned_Sigma_Z',
    ]:
        df[col] = np.nan
    df['RTS_Measurement_Source'] = 'NONE'
    df['RTS_Measurement_Class'] = 'NONE'

    finite_coords = df[settings.MEAS_COLS].notna().all(axis=1).to_numpy()
    if 'Coord_State' in df.columns:
        allowed_state = df['Coord_State'].isin([
            'DETECTED', 'SUB', 'DETECTED_ZONE', 'SUB_ZONE'
        ]).to_numpy()
    else:
        # 직접 호출 호환용 fallback. 정규 pipeline에서는 Coord_State가 반드시 존재한다.
        allowed_state = np.ones(len(df), dtype=bool)
    valid_mask = finite_coords & allowed_state
    df['RTS_Input_Valid'] = valid_mask

    # 표시용 bridge는 시각화만 위한 것이므로 RTS 입력으로 절대 들어가면 안 된다.
    if 'Display_Zone_Bridge' in df.columns:
        display_bridge = (
            df['Display_Zone_Bridge'].fillna(False).to_numpy(dtype=bool)
        )
        if (display_bridge & valid_mask).any():
            raise RuntimeError('Display-zone bridge leaked into RTS input.')

    exclusion_reason = np.full(len(df), 'NONE', dtype=object)
    exclusion_reason[~finite_coords] = 'NONFINITE_COORD'
    if 'Coord_State' in df.columns:
        exclusion_reason[df['Coord_State'].eq('LOST').to_numpy()] = 'LOST'
        exclusion_reason[df['Coord_State'].eq('GRAB').to_numpy()] = 'GRAB'
    df['RTS_Exclusion_Reason'] = exclusion_reason

    frames = pd.to_numeric(df['Frame'], errors='coerce').to_numpy(dtype=float)
    measurement_class = np.full(len(df), 'NONE', dtype=object)
    for idx in np.flatnonzero(valid_mask):
        measurement_class[idx] = _measurement_class(df.at[idx, 'Filled_Type'])
    df['RTS_Measurement_Class'] = measurement_class

    stable_raw_anchor = df.get(
        'Anchor_Raw_Valid', pd.Series(False, index=df.index)
    ).fillna(False).to_numpy(dtype=bool)
    runs, hard_boundary, segment_id = _rts_valid_runs(
        valid_mask, frames, measurement_class, stable_raw_anchor
    )
    df['RTS_Hard_Boundary'] = hard_boundary
    df['RTS_Segment_ID'] = segment_id

    Q = build_cv_process_Q(sigma_a, dt=1.0)
    for start, end in runs:
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.block([[np.eye(3), np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        kf.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        kf.Q = Q
        first = df.loc[start, settings.MEAS_COLS].to_numpy(dtype=float)
        kf.x = np.r_[first, np.zeros(3)]
        kf.P = np.eye(6) * 50.0
        states, covariances, sigmas, sources = [], [], [], []
        for offset, i in enumerate(range(start, end + 1)):
            if offset > 0:
                kf.predict()
            R, sigma, source = _measurement_R_and_sigma(df.at[i, 'Filled_Type'])
            kf.R = R
            kf.update(df.loc[i, settings.MEAS_COLS].to_numpy(dtype=float))
            states.append(kf.x.copy())
            covariances.append(kf.P.copy())
            sigmas.append(sigma.copy())
            sources.append(source)
        xs, ps, _, _ = kf.rts_smoother(
            np.asarray(states), np.asarray(covariances)
        )
        idx = np.arange(start, end + 1)
        df.loc[idx, ['X_rts', 'Y_rts', 'Z_rts']] = xs[:, :3]
        std = np.sqrt(
            np.maximum(np.diagonal(ps, axis1=1, axis2=2)[:, :3], 0.0)
        )
        df.loc[idx, ['RTS_Std_X', 'RTS_Std_Y', 'RTS_Std_Z']] = std
        df.loc[
            idx,
            ['Assigned_Sigma_X', 'Assigned_Sigma_Y', 'Assigned_Sigma_Z'],
        ] = np.asarray(sigmas)
        df.loc[idx, 'RTS_Measurement_Source'] = sources

        # [REV-03 | 2026-09-13] 최종 3D RTS는 시각적 평활화를 위한 단계로 한정한다.
        # 단안 복원 프레임은 광선-평면 기하와 1D 복원 결과를 그대로 보존하기 위해
        # X/Y/Z 전체를 pre-RTS 복원 좌표(Meas_*)로 되돌린다.
        # 따라서 최종 3D RTS는 주변 궤적의 미세 변동 완화에는 사용되지만,
        # 이미 생성된 단안 복원 좌표 자체를 다시 이동시키지 않는다.
        recovery_keep = (
            df.loc[idx, 'Recovery_Applied'].fillna(False).to_numpy(dtype=bool)
        )
        if recovery_keep.any():
            recovery_idx = idx[recovery_keep]
            df.loc[recovery_idx, ['X_rts', 'Y_rts', 'Z_rts']] = (
                df.loc[recovery_idx, settings.MEAS_COLS].to_numpy(dtype=float)
            )

            # 복원 프레임의 최종 좌표는 3D RTS posterior가 아니므로,
            # 3D RTS 표준편차를 최종 좌표의 불확실성처럼 해석하지 않도록 비운다.
            df.loc[recovery_idx, ['RTS_Std_X', 'RTS_Std_Y', 'RTS_Std_Z']] = np.nan

            top_keep = (
                df.loc[recovery_idx, 'Recovery_Mode'].eq(settings.RECOVERY_TOP_RAY).to_numpy()
            )
            front_keep = (
                df.loc[recovery_idx, 'Recovery_Mode'].eq(settings.RECOVERY_FRONT_RAY).to_numpy()
            )
            if top_keep.any():
                top_idx = recovery_idx[top_keep]
                hidden_std = pd.to_numeric(
                    df.loc[top_idx, 'Hidden_Axis_Posterior_Std'], errors='coerce'
                ).to_numpy(dtype=float)
                df.loc[top_idx, 'RTS_Std_Z'] = hidden_std
                df.loc[top_idx, 'Assigned_Sigma_Z'] = hidden_std
            if front_keep.any():
                front_idx = recovery_idx[front_keep]
                hidden_std = pd.to_numeric(
                    df.loc[front_idx, 'Hidden_Axis_Posterior_Std'], errors='coerce'
                ).to_numpy(dtype=float)
                df.loc[front_idx, 'RTS_Std_Y'] = hidden_std
                df.loc[front_idx, 'Assigned_Sigma_Y'] = hidden_std
    return df


def assemble(df):
    df['X_cm'] = df['X_rts']
    df['Y_cm'] = df['Y_rts']
    df['Z_cm'] = df['Z_rts']

    # pre-RTS에서 적용한 TOP_RAY zone-supported Z floor가 RTS 평활화로 다시
    # 3 cm 아래로 내려가지 않도록 최종 출력에서도 동일 조건을 재적용한다.
    # 장기 외삽 허용 여부는 pre-RTS lost_mask 단계에서 이미 확정되어 있다.
    # anchor strategy와 무관하며, 실제로 복원된 TOP_RAY frame에만 적용한다.
    # [2026-10-01] 하한은 pre-RTS 단계에서 실제로 적용된 프레임(Z_Floor_Applied)에만 다시 적용한다.
    top_zone_floor = (
        df['Recovery_Applied'].fillna(False).astype(bool)
        & df['Recovery_Mode'].eq(settings.RECOVERY_TOP_RAY)
        & df['Z_Floor_Applied'].fillna(False).astype(bool)
        & df['Z_cm'].notna()
    )
    df.loc[top_zone_floor, 'Z_cm'] = np.maximum(
        df.loc[top_zone_floor, 'Z_cm'].to_numpy(dtype=float),
        settings.ZONE_TRACK_CENTER_Z_FLOOR,
    )

    blind = df['Coord_State'].isin(['GRAB', 'LOST'])
    df.loc[blind, ['X_cm', 'Y_cm', 'Z_cm']] = np.nan

    # 운영 판정은 pre-RTS 좌표에 고정하되, 최종 RTS 좌표가 zone 경계를 넘었는지는
    # 별도 진단 열로 기록한다. Display_In_Zone을 재계산하거나 재순환시키지 않는다.
    final_coords = df[['X_cm', 'Y_cm', 'Z_cm']].to_numpy(dtype=float)
    final_available = np.isfinite(final_coords).all(axis=1)
    in_zone_final = np.zeros(len(df), dtype=bool)
    for idx in np.flatnonzero(final_available):
        in_zone_final[idx] = _in_zone_3d(*final_coords[idx])

    pre_in_zone = df.get(
        'In_Zone_PreRTS', df.get('In_Zone', pd.Series(False, index=df.index))
    ).fillna(False).to_numpy(dtype=bool)
    pre_available = df.get(
        'Pre_RTS_Coord_Available', pd.Series(False, index=df.index)
    ).fillna(False).to_numpy(dtype=bool)

    df['In_Zone_Final'] = in_zone_final
    df['Final_Coord_Available'] = final_available
    df['Zone_RTS_Class_Changed'] = (
        pre_available
        & final_available
        & (pre_in_zone != in_zone_final)
    )
    return df


def validate_post_rts_invariants(df):
    """RTS 이후에도 표시 bridge와 TOP_RAY 물리 제약이 유지되는지 검사한다."""
    if 'Display_Zone_Bridge' in df.columns:
        bridge = df['Display_Zone_Bridge'].fillna(False).to_numpy(dtype=bool)
        if (bridge & df['RTS_Input_Valid'].fillna(False).to_numpy(dtype=bool)).any():
            raise RuntimeError('Display-zone bridge entered RTS input.')
        final_finite = df[['X_cm', 'Y_cm', 'Z_cm']].notna().all(axis=1).to_numpy()
        if (bridge & final_finite).any():
            raise RuntimeError('Display-zone bridge unexpectedly created final coordinates.')

    # [REV-03 | 2026-09-13] 단안 복원 프레임의 최종 X/Y/Z가 pre-RTS 복원 좌표와
    # 동일하게 보존되는지 검증한다. 최종 3D RTS는 이 좌표들을 변경하지 않는다.
    preserved_recovery = (
        df['Recovery_Applied'].fillna(False).astype(bool)
        & ~df['Coord_State'].isin(['GRAB', 'LOST'])
        & df[settings.MEAS_COLS].notna().all(axis=1)
    )
    if preserved_recovery.any():
        pre = df.loc[preserved_recovery, settings.MEAS_COLS].to_numpy(dtype=float)
        final = df.loc[preserved_recovery, ['X_cm', 'Y_cm', 'Z_cm']].to_numpy(dtype=float)
        if not np.allclose(pre, final, rtol=0.0, atol=1e-9, equal_nan=True):
            raise RuntimeError(
                'A reconstructed coordinate was changed by the final visual 3D RTS.'
            )

    # [2026-10-01] 하한 검사는 pre-RTS에서 하한이 적용된 프레임에만 한다.
    top_zone_floor = (
        df['Recovery_Applied'].fillna(False).astype(bool)
        & df['Recovery_Mode'].eq(settings.RECOVERY_TOP_RAY)
        & df['Z_Floor_Applied'].fillna(False).astype(bool)
        & df['Z_cm'].notna()
        & ~df['Coord_State'].isin(['GRAB', 'LOST'])
    )
    bad_floor = top_zone_floor & (
        pd.to_numeric(df['Z_cm'], errors='coerce') < settings.ZONE_TRACK_CENTER_Z_FLOOR - 1e-9
    )
    if bad_floor.any():
        raise RuntimeError(
            f'Final TOP_RAY zone-supported Z floor violated in {int(bad_floor.sum())} frames.'
        )
    return df
