"""Step ⑬: display-only 3D RTS smoothing (Plot_Aux)."""

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

from .. import settings
from .ray_geometry import _in_zone_3d
from .step08_estimate_noise import build_cv_process_Q


def _measurement_class(filled_type):
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
    else:
        raise RuntimeError(f'Unknown RTS measurement source: {source}')
    return np.diag(sigma**2), sigma, source


def _rts_valid_runs(
    valid_mask, frames, measurement_class, stable_raw_anchor
):
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
        allowed_state = np.ones(len(df), dtype=bool)
    valid_mask = finite_coords & allowed_state
    df['RTS_Input_Valid'] = valid_mask

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

        recovery_keep = (
            df.loc[idx, 'Recovery_Applied'].fillna(False).to_numpy(dtype=bool)
        )
        if recovery_keep.any():
            recovery_idx = idx[recovery_keep]
            df.loc[recovery_idx, ['X_rts', 'Y_rts', 'Z_rts']] = (
                df.loc[recovery_idx, settings.MEAS_COLS].to_numpy(dtype=float)
            )

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
    if 'Display_Zone_Bridge' in df.columns:
        bridge = df['Display_Zone_Bridge'].fillna(False).to_numpy(dtype=bool)
        if (bridge & df['RTS_Input_Valid'].fillna(False).to_numpy(dtype=bool)).any():
            raise RuntimeError('Display-zone bridge entered RTS input.')
        final_finite = df[['X_cm', 'Y_cm', 'Z_cm']].notna().all(axis=1).to_numpy()
        if (bridge & final_finite).any():
            raise RuntimeError('Display-zone bridge unexpectedly created final coordinates.')

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
