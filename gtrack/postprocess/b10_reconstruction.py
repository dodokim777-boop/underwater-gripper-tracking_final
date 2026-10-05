# -*- coding: utf-8 -*-
"""
⑩ 단일시점 구간 좌표 복원 (다이어그램 2쪽 흐름 전체)

결측 구조 판정
  - 양측 단일 결측(300프레임 이하) : 비관측 축을 양측 기준 관측 사이에서 선형 보간
  - 반복 결측 · 300프레임 초과 양측 결측 : 1차원 Kalman/RTS episode (b10_hidden_axis_model)
  - 단측 결측 : 말단 순방향 / 선두 시간 역전 1차원 Kalman · RTS
  - 기준 관측 없음 : LOST (NO_RAW_ANCHOR)
→ 비관측 축 값 확정 (TOP_RAY → Z, FRONT_RAY → Y)
→ 복원 Z 하한 (TOP_RAY만, 상자 통과 + 기준 관측 Z ≥ 3 cm일 때 Z ← max(Z, 3 cm))
→ 유지된 카메라의 굴절 광선 × 비관측 축 평면 교점 (TOP_RAY → X,Y / FRONT_RAY → X,Z)
→ 기준 위치 오프셋 보정
→ 광선 교점 유효성 · FRONT_RAY X 일관성 검사 → 실패 시 LOST (사유는 Hidden_Stop_Reason)
출력: Coord_Type의 근거(Filled_Type), Hidden_Axis_Posterior_Std(→Hidden_Axis_Std_cm), Z_Floor_Applied 등
"""

import numpy as np
import pandas as pd

from .. import settings
from .geometry import top_ray_plane_z, front_ray_plane_y, _in_zone_3d
from .b10_hidden_axis_model import _select_sparse_anchor_sets, _run_1d_smoother_with_direction


def _long_both_lost_barrier(df, min_len=None):
    """GRAB 후보가 될 정도로 긴 양안 LOST run은 anchor 검색 hard barrier로 둔다."""
    if min_len is None:
        min_len = settings.GRAB_MIN
    both_lost = (
        df['Status_Top'].eq('LOST').to_numpy()
        & df['Status_Front'].eq('LOST').to_numpy()
    )
    barrier = np.zeros(len(df), dtype=bool)
    idx = 0
    while idx < len(df):
        if not both_lost[idx]:
            idx += 1
            continue
        end = idx
        while end + 1 < len(df) and both_lost[end + 1]:
            end += 1
        if end - idx + 1 >= int(min_len):
            barrier[idx:end + 1] = True
        idx = end + 1
    return barrier


def _recovery_runs_for_mode(recovery_mode, mode):
    runs = []
    idx = 0
    while idx < len(recovery_mode):
        if recovery_mode[idx] != mode:
            idx += 1
            continue
        end = idx
        while end + 1 < len(recovery_mode) and recovery_mode[end + 1] == mode:
            end += 1
        runs.append((idx, end))
        idx = end + 1
    return runs


def _max_consecutive_true(mask):
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _group_recovery_runs_into_episodes(
    runs, frames, recovery_mode, raw_valid, hard_barrier, mode,
    max_link_frames=None,
    stable_reacquire_min_raw=None,
):
    """불안정한 짧은 raw island만 같은 hidden-axis episode 안에 묶는다.

    두 same-mode 결측 run 사이에 ``stable_reacquire_min_raw`` 이상의 연속 실제 raw
    stereo가 있으면 안정적인 재획득으로 보고 episode를 분리한다. 따라서 2+3처럼
    끊긴 짧은 raw island는 기존 상태의 position update로만 반영되고, 새 velocity
    prior를 만들지 않는다.
    """
    if max_link_frames is None:
        max_link_frames = settings.HIDDEN_EPISODE_LINK_MAX
    if stable_reacquire_min_raw is None:
        stable_reacquire_min_raw = settings.HIDDEN_REACQUIRE_MIN_CONSECUTIVE_RAW
    if not runs:
        return []
    groups = [[runs[0]]]
    for current in runs[1:]:
        prev = groups[-1][-1]
        gap_start = prev[1] + 1
        gap_end = current[0] - 1
        gap_len = max(gap_end - gap_start + 1, 0)
        can_link = gap_len <= int(max_link_frames)
        if can_link and gap_len > 0:
            if np.any(hard_barrier[gap_start:gap_end + 1]):
                can_link = False
            if np.any(recovery_mode[gap_start:gap_end + 1] != settings.RECOVERY_NONE):
                can_link = False
            local_frames = frames[prev[1]:current[0] + 1]
            if np.any(np.diff(local_frames) != 1):
                can_link = False
            if _max_consecutive_true(raw_valid[gap_start:gap_end + 1]) >= int(stable_reacquire_min_raw):
                can_link = False
        if can_link:
            groups[-1].append(current)
        else:
            groups.append([current])
    return groups


def _front_x_sanity_failed_targets(df, frames, target_indices, x_values, anchor_indices):
    """임의 target frame의 Front-ray X를 주변 실제 raw X 궤적과 비교한다."""
    target_indices = np.asarray(target_indices, dtype=int)
    x_values = np.asarray(x_values, dtype=float)
    anchors = np.asarray(anchor_indices, dtype=int)
    failed = ~np.isfinite(x_values)
    if len(target_indices) == 0:
        return failed
    if len(anchors) == 0:
        return np.ones(len(target_indices), dtype=bool)

    anchor_frames = frames[anchors]
    anchor_x = df.loc[anchors, 'Anchor_Raw_X'].to_numpy(dtype=float)
    valid = np.isfinite(anchor_frames) & np.isfinite(anchor_x)
    anchor_frames, anchor_x = anchor_frames[valid], anchor_x[valid]
    if len(anchor_frames) == 0:
        return np.ones(len(target_indices), dtype=bool)

    order = np.argsort(anchor_frames)
    anchor_frames, anchor_x = anchor_frames[order], anchor_x[order]
    target_frames = frames[target_indices]
    if len(anchor_frames) >= 2:
        expected = np.interp(target_frames, anchor_frames, anchor_x)
    else:
        expected = np.full(len(target_frames), anchor_x[0], dtype=float)

    nearest_age = np.min(
        np.abs(target_frames[:, None] - anchor_frames[None, :]), axis=1
    )
    allowed = settings.FRONT_X_SANITY_MARGIN + settings.V_MAX_X_FRONT_RAY * nearest_age
    failed |= np.abs(x_values - expected) > allowed
    return failed


def _top_anchor_offset(df, anchor_set, calib_top):
    """여러 raw anchor의 (실제 XY - Top-ray XY) 보정량 median."""
    offset = settings.MOTOR_CENTER_IN_ID0_CM
    residuals = []
    for idx in np.asarray(anchor_set, dtype=int):
        u = pd.to_numeric(pd.Series([df.at[idx, 'Top_px_u']]), errors='coerce').iloc[0]
        v = pd.to_numeric(pd.Series([df.at[idx, 'Top_px_v']]), errors='coerce').iloc[0]
        z_motor = float(df.at[idx, 'Anchor_Raw_Z'])
        if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(z_motor):
            continue
        point_world = top_ray_plane_z((u, v), z_motor + offset[2], calib_top, settings.INTERFACE_TOP)
        if point_world is None:
            continue
        ray_xy = point_world[:2] - offset[:2]
        raw_xy = df.loc[idx, ['Anchor_Raw_X', 'Anchor_Raw_Y']].to_numpy(dtype=float)
        if np.isfinite(raw_xy).all() and np.isfinite(ray_xy).all():
            residuals.append(raw_xy - ray_xy)
    if not residuals:
        return None
    return np.median(np.asarray(residuals, dtype=float), axis=0)


def _front_anchor_offset(df, anchor_set, calib_front):
    """여러 raw anchor의 (실제 XZ - Front-ray XZ) 보정량 median."""
    offset = settings.MOTOR_CENTER_IN_ID0_CM
    residuals = []
    for idx in np.asarray(anchor_set, dtype=int):
        u = pd.to_numeric(pd.Series([df.at[idx, 'Front_px_u']]), errors='coerce').iloc[0]
        v = pd.to_numeric(pd.Series([df.at[idx, 'Front_px_v']]), errors='coerce').iloc[0]
        y_motor = float(df.at[idx, 'Anchor_Raw_Y'])
        if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(y_motor):
            continue
        point_world = front_ray_plane_y((u, v), y_motor + offset[1], calib_front, settings.INTERFACE_FRONT)
        if point_world is None:
            continue
        ray_xz = point_world[[0, 2]] - offset[[0, 2]]
        raw_xz = df.loc[idx, ['Anchor_Raw_X', 'Anchor_Raw_Z']].to_numpy(dtype=float)
        if np.isfinite(raw_xz).all() and np.isfinite(ray_xz).all():
            residuals.append(raw_xz - ray_xz)
    if not residuals:
        return None
    return np.median(np.asarray(residuals, dtype=float), axis=0)


def _blend_anchor_relative(forward, backward):
    length = len(forward)
    tau = (np.arange(length, dtype=float) + 1.0) / (length + 1.0)
    output = np.full_like(forward, np.nan, dtype=float)
    for k in range(length):
        f_ok = np.isfinite(forward[k]).all()
        b_ok = np.isfinite(backward[k]).all()
        if f_ok and b_ok:
            output[k] = (1.0 - tau[k]) * forward[k] + tau[k] * backward[k]
        elif f_ok:
            output[k] = forward[k]
        elif b_ok:
            output[k] = backward[k]
    return output


def _accumulate_xy_top(df, start, end, left_set, right_set, z_segment, calib_top):
    if calib_top is None:
        raise RuntimeError('Top calibration is required for TOP_RAY reconstruction.')

    indices = np.arange(start, end + 1)
    length = len(indices)
    offset = settings.MOTOR_CENTER_IN_ID0_CM

    def top_only(idx, z_motor):
        u = pd.to_numeric(pd.Series([df.at[idx, 'Top_px_u']]), errors='coerce').iloc[0]
        v = pd.to_numeric(pd.Series([df.at[idx, 'Top_px_v']]), errors='coerce').iloc[0]
        if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(z_motor):
            return None
        point_world = top_ray_plane_z(
            (u, v), float(z_motor) + offset[2], calib_top, settings.INTERFACE_TOP
        )
        return None if point_world is None else point_world[:2] - offset[:2]

    points = [top_only(idx, z_segment[k]) for k, idx in enumerate(indices)]
    left_offset = _top_anchor_offset(df, left_set, calib_top) if len(left_set) else None
    right_offset = _top_anchor_offset(df, right_set, calib_top) if len(right_set) else None

    forward = np.full((length, 2), np.nan)
    backward = np.full((length, 2), np.nan)
    for k, point in enumerate(points):
        if point is None:
            continue
        if left_offset is not None:
            forward[k] = point + left_offset
        if right_offset is not None:
            backward[k] = point + right_offset
    return _blend_anchor_relative(forward, backward)


def _accumulate_xz_front(df, start, end, left_set, right_set, y_segment, calib_front):
    if calib_front is None:
        raise RuntimeError('Front calibration is required for FRONT_RAY reconstruction.')

    indices = np.arange(start, end + 1)
    length = len(indices)
    offset = settings.MOTOR_CENTER_IN_ID0_CM

    def front_only(idx, y_motor):
        u = pd.to_numeric(pd.Series([df.at[idx, 'Front_px_u']]), errors='coerce').iloc[0]
        v = pd.to_numeric(pd.Series([df.at[idx, 'Front_px_v']]), errors='coerce').iloc[0]
        if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(y_motor):
            return None
        point_world = front_ray_plane_y(
            (u, v), float(y_motor) + offset[1], calib_front, settings.INTERFACE_FRONT
        )
        if point_world is None:
            return None
        return point_world[[0, 2]] - offset[[0, 2]]

    points = [front_only(idx, y_segment[k]) for k, idx in enumerate(indices)]
    left_offset = _front_anchor_offset(df, left_set, calib_front) if len(left_set) else None
    right_offset = _front_anchor_offset(df, right_set, calib_front) if len(right_set) else None

    forward = np.full((length, 2), np.nan)
    backward = np.full((length, 2), np.nan)
    for k, point in enumerate(points):
        if point is None:
            continue
        if left_offset is not None:
            forward[k] = point + left_offset
        if right_offset is not None:
            backward[k] = point + right_offset
    return _blend_anchor_relative(forward, backward)


def interpolate_missing(df, calib_top, calib_front, sigma_a):
    """clean gap은 선형 내삽, intermittent/tail은 1D Kalman/RTS로 복원한다."""
    n = len(df)
    frames = pd.to_numeric(df['Frame'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(frames).all() or np.any(np.diff(frames) <= 0):
        raise ValueError('Frame must be finite and strictly increasing.')
    if 'Anchor_Raw_Valid' not in df.columns:
        raise RuntimeError('freeze_anchor_raw_snapshot() must run before interpolation.')

    sigma_a = np.asarray(sigma_a, dtype=float).reshape(3)
    raw_valid = df['Anchor_Raw_Valid'].fillna(False).to_numpy(dtype=bool)
    hard_barrier = _long_both_lost_barrier(df, min_len=settings.GRAB_MIN)
    recovery_mode = df['Recovery_Mode'].astype(str).to_numpy()
    state = df['State2'].to_numpy(dtype=object).copy()

    filled_type = np.full(n, 'NONE', dtype=object)
    front_x_sanity_failed = np.zeros(n, dtype=bool)
    recovery_applied = np.zeros(n, dtype=bool)
    anchor_left_index = np.full(n, -1, dtype=int)
    anchor_right_index = np.full(n, -1, dtype=int)
    anchor_right_used_index = np.full(n, -1, dtype=int)
    anchor_left_count = np.zeros(n, dtype=int)
    anchor_right_count = np.zeros(n, dtype=int)
    anchor_left_window = np.zeros(n, dtype=int)
    anchor_right_window = np.zeros(n, dtype=int)
    anchor_strategy = np.full(n, 'NONE', dtype=object)

    hidden_axis_model = np.full(n, 'NONE', dtype=object)
    hidden_episode_id = np.full(n, -1, dtype=int)
    hidden_filtered_value = np.full(n, np.nan, dtype=float)
    hidden_smoothed_value = np.full(n, np.nan, dtype=float)
    hidden_filtered_velocity = np.full(n, np.nan, dtype=float)
    hidden_smoothed_velocity = np.full(n, np.nan, dtype=float)
    hidden_posterior_std = np.full(n, np.nan, dtype=float)
    hidden_initial_velocity = np.full(n, np.nan, dtype=float)
    hidden_initial_velocity_std = np.full(n, np.nan, dtype=float)
    hidden_innovation_weight = np.full(n, np.nan, dtype=float)
    hidden_anchor_weight_median = np.full(n, np.nan, dtype=float)
    hidden_anchor_weight_min = np.full(n, np.nan, dtype=float)
    hidden_stop_reason = np.full(n, 'NONE', dtype=object)
    hidden_uncertainty_limit = np.full(n, np.nan, dtype=float)
    hidden_has_future_raw = np.zeros(n, dtype=bool)
    hidden_q_sigma_a = np.full(n, np.nan, dtype=float)
    hidden_r_sigma = np.full(n, np.nan, dtype=float)
    z_floor_applied = np.zeros(n, dtype=bool)
    top_zone_extension_applied = np.zeros(n, dtype=bool)

    # Compatibility diagnostics retained, but Theil-Sen is only the episode-start prior.
    left_model_valid = np.zeros(n, dtype=bool)
    right_model_valid = np.zeros(n, dtype=bool)
    left_slope_raw = np.full(n, np.nan, dtype=float)
    right_slope_raw = np.full(n, np.nan, dtype=float)
    left_slope_used = np.full(n, np.nan, dtype=float)
    right_slope_used = np.full(n, np.nan, dtype=float)
    left_residual_mad = np.full(n, np.nan, dtype=float)
    right_residual_mad = np.full(n, np.nan, dtype=float)
    left_pair_count = np.zeros(n, dtype=int)
    right_pair_count = np.zeros(n, dtype=int)
    left_slope_clipped = np.zeros(n, dtype=bool)
    right_slope_clipped = np.zeros(n, dtype=bool)

    top_supported = df['Top_Zone_Supported'].fillna(False).to_numpy(dtype=bool)
    top_detected = df['Status_Top'].eq('DETECTED').to_numpy()
    reason = df['Missing_Reason'].astype(str).to_numpy()

    stats = {
        'TOP_INTERP_SEGMENTS': 0,
        'TOP_EXTRAP_SEGMENTS': 0,
        'TOP_KALMAN_EPISODES': 0,
        'TOP_ZONE_SEGMENTS': 0,
        'TOP_RECON_FRAMES': 0,
        'FRONT_INTERP_SEGMENTS': 0,
        'FRONT_EXTRAP_SEGMENTS': 0,
        'FRONT_KALMAN_EPISODES': 0,
        'FRONT_RECON_FRAMES': 0,
        'FRONT_X_SANITY_REJECTED_FRAMES': 0,
        'ANCHOR_WINDOW_60_SEGMENTS': 0,
        'INSUFFICIENT_ANCHOR_SEGMENTS': 0,
        'THEIL_SEN_MODELS_FIT': 0,
        'THEIL_SEN_SLOPE_CLIPPED_MODELS': 0,
        'UNCERTAINTY_STOPPED_FRAMES': 0,
        'MAX_LENGTH_STOPPED_FRAMES': 0,
        'LOST_SEGMENTS': 0,
        'LOST_FRAMES': 0,
    }

    def store_initial_model(indices, side, model):
        if model is None or len(indices) == 0:
            return
        stats['THEIL_SEN_MODELS_FIT'] += 1
        if bool(model.get('slope_clipped', False)):
            stats['THEIL_SEN_SLOPE_CLIPPED_MODELS'] += 1
        if side == 'LEFT':
            left_model_valid[indices] = True
            left_slope_raw[indices] = model['slope_raw']
            left_slope_used[indices] = model['slope_used']
            left_residual_mad[indices] = model['residual_mad']
            left_pair_count[indices] = model['pair_count']
            left_slope_clipped[indices] = model['slope_clipped']
        else:
            right_model_valid[indices] = True
            right_slope_raw[indices] = model['slope_raw']
            right_slope_used[indices] = model['slope_used']
            right_residual_mad[indices] = model['residual_mad']
            right_pair_count[indices] = model['pair_count']
            right_slope_clipped[indices] = model['slope_clipped']

    episode_counter = 0
    for mode in (settings.RECOVERY_TOP_RAY, settings.RECOVERY_FRONT_RAY):
        mode_runs = _recovery_runs_for_mode(recovery_mode, mode)
        groups = _group_recovery_runs_into_episodes(
            mode_runs, frames, recovery_mode, raw_valid, hard_barrier, mode,
            max_link_frames=settings.HIDDEN_EPISODE_LINK_MAX,
            stable_reacquire_min_raw=settings.HIDDEN_REACQUIRE_MIN_CONSECUTIVE_RAW,
        )

        axis_name = 'Z' if mode == settings.RECOVERY_TOP_RAY else 'Y'
        axis_idx = 2 if mode == settings.RECOVERY_TOP_RAY else 1
        sigma_meas = float(settings.SIG_RTS_RAW[axis_idx])
        sigma_axis_a = float(sigma_a[axis_idx]) * (
            settings.HIDDEN_Q_SCALE_Z if mode == settings.RECOVERY_TOP_RAY else settings.HIDDEN_Q_SCALE_Y
        )
        vmax = settings.V_MAX_Z if mode == settings.RECOVERY_TOP_RAY else settings.V_MAX_Y
        std_stop = (
            settings.HIDDEN_STD_STOP_Z_CM if mode == settings.RECOVERY_TOP_RAY
            else settings.HIDDEN_STD_STOP_Y_CM
        )
        anchor_col = f'Anchor_Raw_{axis_name}'

        for group in groups:
            episode_id = episode_counter
            episode_counter += 1
            ep_start = int(group[0][0])
            ep_end = int(group[-1][1])
            candidate_indices = np.concatenate([
                np.arange(a, b + 1, dtype=int) for a, b in group
            ])
            hidden_episode_id[candidate_indices] = episode_id

            left_found, right_found, left_win, right_win, strategy = (
                _select_sparse_anchor_sets(
                    raw_valid, frames, hard_barrier, ep_start, ep_end
                )
            )
            left_set = np.asarray(left_found, dtype=int)
            right_set = np.asarray(right_found, dtype=int)
            left = int(left_set[-1]) if len(left_set) else None
            right = int(right_set[0]) if len(right_set) else None

            anchor_left_index[candidate_indices] = -1 if left is None else left
            anchor_right_index[candidate_indices] = -1 if right is None else right
            anchor_right_used_index[candidate_indices] = -1 if right is None else right
            anchor_left_count[candidate_indices] = len(left_set)
            anchor_right_count[candidate_indices] = len(right_set)
            anchor_left_window[candidate_indices] = int(left_win)
            anchor_right_window[candidate_indices] = int(right_win)
            if left_win == settings.ANCHOR_SEARCH_EXTENDED or right_win == settings.ANCHOR_SEARCH_EXTENDED:
                stats['ANCHOR_WINDOW_60_SEGMENTS'] += 1

            clean_two_sided = len(group) == 1 and left is not None and right is not None
            if clean_two_sided:
                run_start, run_end = group[0]
                segment = np.arange(run_start, run_end + 1, dtype=int)
                f0, f1 = float(frames[left]), float(frames[right])
                h0 = float(df.at[left, anchor_col])
                h1 = float(df.at[right, anchor_col])
                tau = np.clip((frames[segment] - f0) / (f1 - f0), 0.0, 1.0)
                hidden = (1.0 - tau) * h0 + tau * h1
                hidden_std = np.sqrt(
                    ((1.0 - tau)**2 + tau**2) * sigma_meas**2
                )
                velocity = (h1 - h0) / (f1 - f0)

                if mode == settings.RECOVERY_TOP_RAY:
                    support = (
                        top_supported[segment]
                        & top_detected[segment]
                        & (reason[segment] == 'FRONT_MISSING')
                        & (min(h0, h1) >= settings.ZONE_TRACK_CENTER_Z_FLOOR)   # [2026-10-01] 기준 관측 Z 조건
                    )
                    floor = support & (hidden < settings.ZONE_TRACK_CENTER_Z_FLOOR)
                    hidden[support] = np.maximum(
                        hidden[support], settings.ZONE_TRACK_CENTER_Z_FLOOR
                    )
                    z_floor_applied[segment[floor]] = True
                    if floor.any():
                        stats['TOP_ZONE_SEGMENTS'] += 1
                    observed = _accumulate_xy_top(
                        df, run_start, run_end, left_set, right_set, hidden, calib_top
                    )
                    coords = np.column_stack([observed[:, 0], observed[:, 1], hidden])
                    stats['TOP_INTERP_SEGMENTS'] += 1
                else:
                    observed = _accumulate_xz_front(
                        df, run_start, run_end, left_set, right_set, hidden, calib_front
                    )
                    coords = np.column_stack([observed[:, 0], hidden, observed[:, 1]])
                    x_failed = _front_x_sanity_failed_targets(
                        df, frames, segment, coords[:, 0],
                        np.r_[left_set, right_set],
                    )
                    front_x_sanity_failed[segment] = x_failed
                    coords[x_failed] = np.nan
                    stats['FRONT_X_SANITY_REJECTED_FRAMES'] += int(x_failed.sum())
                    stats['FRONT_INTERP_SEGMENTS'] += 1

                finite = np.isfinite(coords).all(axis=1)
                kept = segment[finite]
                rejected = segment[~finite]
                if len(kept):
                    df.loc[kept, settings.MEAS_COLS] = coords[finite]
                    recovery_applied[kept] = True
                    base_type = f'LINEAR_INTERP_{mode}'
                    filled_type[kept] = base_type
                    floor_kept = z_floor_applied[kept]
                    filled_type[kept[floor_kept]] = f'{base_type}_ZONE_FLOOR'
                    hidden_axis_model[kept] = 'LINEAR_INTERPOLATION'
                    hidden_filtered_value[kept] = hidden[finite]
                    hidden_smoothed_value[kept] = hidden[finite]
                    hidden_filtered_velocity[kept] = velocity
                    hidden_smoothed_velocity[kept] = velocity
                    hidden_posterior_std[kept] = hidden_std[finite]
                    hidden_initial_velocity[kept] = velocity
                    hidden_initial_velocity_std[kept] = np.nan
                    hidden_uncertainty_limit[kept] = std_stop
                    hidden_q_sigma_a[kept] = sigma_axis_a
                    hidden_r_sigma[kept] = sigma_meas
                    hidden_has_future_raw[kept] = True
                    anchor_strategy[kept] = 'TWO_SIDED'
                    if mode == settings.RECOVERY_TOP_RAY:
                        stats['TOP_RECON_FRAMES'] += len(kept)
                    else:
                        stats['FRONT_RECON_FRAMES'] += len(kept)
                if len(rejected):
                    state[rejected] = 'LOST'
                    hidden_stop_reason[rejected] = 'RAY_OR_SANITY_FAILURE'
                    stats['LOST_FRAMES'] += len(rejected)
                    stats['LOST_SEGMENTS'] += 1
                continue

            # Intermittent episode or one-sided tail: one continuous 1D state model.
            internal_raw = np.flatnonzero(
                raw_valid & (np.arange(n) >= ep_start) & (np.arange(n) <= ep_end)
            )
            all_anchor_indices = np.unique(np.r_[left_set, internal_raw, right_set]).astype(int)
            if len(all_anchor_indices) == 0:
                state[candidate_indices] = 'LOST'
                anchor_strategy[candidate_indices] = 'INSUFFICIENT'
                hidden_stop_reason[candidate_indices] = 'NO_RAW_ANCHOR'
                stats['INSUFFICIENT_ANCHOR_SEGMENTS'] += 1
                stats['LOST_SEGMENTS'] += 1
                stats['LOST_FRAMES'] += len(candidate_indices)
                continue

            seq_start = int(min(ep_start, all_anchor_indices[0]))
            seq_end = int(max(ep_end, all_anchor_indices[-1]))
            seq = np.arange(seq_start, seq_end + 1, dtype=int)
            obs_mask = raw_valid[seq]
            obs_values = df.loc[seq, anchor_col].to_numpy(dtype=float)

            has_left = np.any(all_anchor_indices < ep_start)
            has_right = np.any(all_anchor_indices > ep_end)
            reverse = (not has_left) and has_right
            if reverse:
                history_idx = right_set if len(right_set) else all_anchor_indices[:settings.ANCHOR_MODEL_MIN_POINTS]
            else:
                history_idx = left_set if len(left_set) else all_anchor_indices[:settings.ANCHOR_MODEL_MIN_POINTS]

            result = _run_1d_smoother_with_direction(
                frames[seq], obs_values, obs_mask,
                frames[history_idx], df.loc[history_idx, anchor_col].to_numpy(dtype=float),
                sigma_a=sigma_axis_a,
                sigma_meas=sigma_meas,
                vmax=vmax,
                reverse=reverse,
            )
            if result is None:
                state[candidate_indices] = 'LOST'
                anchor_strategy[candidate_indices] = 'INSUFFICIENT'
                hidden_stop_reason[candidate_indices] = 'STATE_INIT_FAILED'
                stats['INSUFFICIENT_ANCHOR_SEGMENTS'] += 1
                stats['LOST_SEGMENTS'] += 1
                stats['LOST_FRAMES'] += len(candidate_indices)
                continue

            pos = candidate_indices - seq_start
            hidden = result['smoothed_state'][pos, 0].copy()
            filtered = result['filtered_state'][pos, 0]
            smooth_vel = result['smoothed_state'][pos, 1]
            filt_vel = result['filtered_state'][pos, 1]
            std = np.sqrt(np.maximum(result['smoothed_cov'][pos, 0, 0], 0.0))

            # 각 candidate frame에서 가장 가까운 과거/미래 실제 raw를 따로 찾는다.
            # 내부 raw island가 있으면 이후 tail의 age와 zone anchor는 그 최신 raw를 기준으로 한다.
            raw_frames = frames[all_anchor_indices]
            cand_frames = frames[candidate_indices]
            past_anchor_idx = np.full(len(candidate_indices), -1, dtype=int)
            future_anchor_idx = np.full(len(candidate_indices), -1, dtype=int)
            for local_i, frame_value in enumerate(cand_frames):
                past = all_anchor_indices[raw_frames < frame_value]
                future = all_anchor_indices[raw_frames > frame_value]
                if len(past):
                    past_anchor_idx[local_i] = int(past[-1])
                if len(future):
                    future_anchor_idx[local_i] = int(future[0])

            has_past_raw = past_anchor_idx >= 0
            has_future_raw = future_anchor_idx >= 0
            hidden_has_future_raw[candidate_indices] = has_future_raw

            lost_mask = ~np.isfinite(hidden) | ~np.isfinite(std)
            std_fail = std > float(std_stop)
            lost_mask |= std_fail
            hidden_stop_reason[candidate_indices[std_fail]] = 'POSTERIOR_STD_LIMIT'
            stats['UNCERTAINTY_STOPPED_FRAMES'] += int(std_fail.sum())

            # One-sided emergency length cap. 각 frame의 가장 가까운 실제 raw를 기준으로 age를 계산한다.
            length_fail = np.zeros(len(candidate_indices), dtype=bool)
            forward_tail = has_past_raw & ~has_future_raw
            backward_tail = ~has_past_raw & has_future_raw
            if forward_tail.any():
                local = np.flatnonzero(forward_tail)
                age = cand_frames[local] - frames[past_anchor_idx[local]]
                length_fail[local] = age > settings.MAX_EXTRAP_LEN
            if backward_tail.any():
                local = np.flatnonzero(backward_tail)
                age = frames[future_anchor_idx[local]] - cand_frames[local]
                length_fail[local] = age > settings.MAX_EXTRAP_LEN

            anchor_strategy[candidate_indices[has_past_raw & has_future_raw]] = 'INTERMITTENT_TWO_SIDED'
            anchor_strategy[candidate_indices[forward_tail]] = 'LEFT_ONLY'
            anchor_strategy[candidate_indices[backward_tail]] = 'RIGHT_ONLY'
            anchor_strategy[candidate_indices[~has_past_raw & ~has_future_raw]] = 'INSUFFICIENT'

            # Top-zone long-tail policy:
            # 일반 150-frame cap을 무제한 해제하지 않고, 조건을 만족하는 TOP_RAY 전방 말단에만
            # 300-frame hard cap을 적용한다. posterior std stop은 그대로 유지한다.
            if mode == settings.RECOVERY_TOP_RAY:
                extension = np.zeros(len(candidate_indices), dtype=bool)
                offset = 0
                for a, b in group:
                    run_len = b - a + 1
                    sl = slice(offset, offset + run_len)
                    local_idx = candidate_indices[sl]
                    local_forward_tail = forward_tail[sl]
                    support_local = (
                        top_supported[local_idx]
                        & top_detected[local_idx]
                        & (reason[local_idx] == 'FRONT_MISSING')
                    )
                    # 같은 candidate run에서 가장 가까운 과거 raw가 in-zone일 때만 확장 정책을 적용한다.
                    if local_forward_tail.any():
                        first_tail_local = int(np.flatnonzero(local_forward_tail)[0])
                        anchor_idx = int(past_anchor_idx[sl][first_tail_local])
                        anchor_in_zone = bool(
                            anchor_idx >= 0
                            and _in_zone_3d(
                                df.at[anchor_idx, 'Anchor_Raw_X'],
                                df.at[anchor_idx, 'Anchor_Raw_Y'],
                                df.at[anchor_idx, 'Anchor_Raw_Z'],
                            )
                        )
                        continuous = np.logical_and.accumulate(support_local)
                        extension[sl] = anchor_in_zone & local_forward_tail & continuous
                    offset += run_len

                top_zone_extension_applied[candidate_indices[extension]] = True
                if extension.any():
                    ext_local = np.flatnonzero(extension)
                    ext_age = (
                        cand_frames[ext_local]
                        - frames[past_anchor_idx[ext_local]]
                    )
                    length_fail[ext_local] = (
                        ext_age > settings.TOP_ZONE_MAX_EXTRAP_LEN
                    )

            length_only = length_fail & (~lost_mask)
            lost_mask |= length_fail
            hidden_stop_reason[candidate_indices[length_only]] = 'MAX_EXTRAP_LEN'
            stats['MAX_LENGTH_STOPPED_FRAMES'] += int(length_only.sum())

            if mode == settings.RECOVERY_TOP_RAY:
                # [2026-10-01] 시간상 가장 가까운 기준 관측(과거 또는 미래)의 Z가 하한 이상일 때만 적용
                past_gap = np.where(past_anchor_idx >= 0, cand_frames - frames[np.maximum(past_anchor_idx, 0)], np.inf)
                future_gap = np.where(future_anchor_idx >= 0, frames[np.maximum(future_anchor_idx, 0)] - cand_frames, np.inf)
                ref_anchor = np.where(past_gap <= future_gap, past_anchor_idx, future_anchor_idx)
                anchor_z = np.array([
                    float(df.at[int(i), 'Anchor_Raw_Z']) if i >= 0 else np.nan for i in ref_anchor
                ])
                support = (
                    top_supported[candidate_indices]
                    & top_detected[candidate_indices]
                    & (reason[candidate_indices] == 'FRONT_MISSING')
                    & np.isfinite(hidden)
                    & (anchor_z >= settings.ZONE_TRACK_CENTER_Z_FLOOR)   # [2026-10-01] 기준 관측 Z 조건
                )
                floor = support & (hidden < settings.ZONE_TRACK_CENTER_Z_FLOOR)
                hidden[support] = np.maximum(hidden[support], settings.ZONE_TRACK_CENTER_Z_FLOOR)
                z_floor_applied[candidate_indices[floor]] = True
                if floor.any() or top_zone_extension_applied[candidate_indices].any():
                    stats['TOP_ZONE_SEGMENTS'] += 1

            # Reconstruct ray-derived coordinates using a single robust offset from all actual raw anchors.
            # TOP_RAY Z-floor가 적용된 candidate 값도 ray intersection에 동일하게 사용한다.
            hidden_full = result['smoothed_state'][:, 0].copy()
            hidden_full[pos] = hidden
            if mode == settings.RECOVERY_TOP_RAY:
                observed_all = _accumulate_xy_top(
                    df, seq_start, seq_end,
                    all_anchor_indices, np.array([], dtype=int),
                    hidden_full, calib_top,
                )
                obs = observed_all[pos]
                coords = np.column_stack([obs[:, 0], obs[:, 1], hidden])
            else:
                observed_all = _accumulate_xz_front(
                    df, seq_start, seq_end,
                    all_anchor_indices, np.array([], dtype=int),
                    hidden_full, calib_front,
                )
                obs = observed_all[pos]
                coords = np.column_stack([obs[:, 0], hidden, obs[:, 1]])
                x_failed = _front_x_sanity_failed_targets(
                    df, frames, candidate_indices, coords[:, 0], all_anchor_indices
                )
                front_x_sanity_failed[candidate_indices] = x_failed
                x_only = x_failed & (~lost_mask)
                lost_mask |= x_failed
                hidden_stop_reason[candidate_indices[x_only]] = 'FRONT_X_SANITY'
                stats['FRONT_X_SANITY_REJECTED_FRAMES'] += int(x_only.sum())

            ray_failed = ~np.isfinite(coords).all(axis=1)
            ray_only = ray_failed & (~lost_mask)
            lost_mask |= ray_failed
            hidden_stop_reason[candidate_indices[ray_only]] = 'RAY_GEOMETRY_FAILURE'

            kept_mask = ~lost_mask
            kept = candidate_indices[kept_mask]
            rejected = candidate_indices[lost_mask]
            if len(kept):
                df.loc[kept, settings.MEAS_COLS] = coords[kept_mask]
                recovery_applied[kept] = True

                local_method = np.full(len(candidate_indices), 'KALMAN_RTS_EPISODE', dtype=object)
                local_method[has_past_raw & ~has_future_raw] = 'KALMAN_FORWARD_TAIL'
                local_method[~has_past_raw & has_future_raw] = 'KALMAN_BACKWARD_TAIL'
                hidden_axis_model[candidate_indices] = local_method
                filled = np.array([f'{m}_{mode}' for m in local_method], dtype=object)
                floor_local = z_floor_applied[candidate_indices]
                filled[floor_local] = np.array([
                    f'{name}_ZONE_FLOOR' for name in filled[floor_local]
                ], dtype=object)
                filled_type[kept] = filled[kept_mask]

                hidden_filtered_value[kept] = filtered[kept_mask]
                hidden_smoothed_value[kept] = hidden[kept_mask]
                hidden_filtered_velocity[kept] = filt_vel[kept_mask]
                hidden_smoothed_velocity[kept] = smooth_vel[kept_mask]
                hidden_posterior_std[kept] = std[kept_mask]
                hidden_uncertainty_limit[kept] = std_stop
                hidden_q_sigma_a[kept] = sigma_axis_a
                hidden_r_sigma[kept] = sigma_meas
                # Candidate frames themselves have no raw measurement update. Store the
                # episode-level robust anchor-weight diagnostics instead of a meaningless NaN.
                raw_weights = result['innovation_weight'][np.isfinite(result['innovation_weight'])]
                if len(raw_weights):
                    weight_median = float(np.median(raw_weights))
                    weight_min = float(np.min(raw_weights))
                    hidden_innovation_weight[kept] = weight_median
                    hidden_anchor_weight_median[kept] = weight_median
                    hidden_anchor_weight_min[kept] = weight_min

                init_model = result.get('initial_model')
                if init_model is not None:
                    hidden_initial_velocity[kept] = init_model['slope_used']
                    hidden_initial_velocity_std[kept] = init_model['velocity_std']
                    if reverse:
                        store_initial_model(kept, 'RIGHT', init_model)
                    else:
                        store_initial_model(kept, 'LEFT', init_model)

                if mode == settings.RECOVERY_TOP_RAY:
                    stats['TOP_RECON_FRAMES'] += len(kept)
                    stats['TOP_KALMAN_EPISODES'] += 1
                    if np.any(local_method == 'KALMAN_FORWARD_TAIL') or np.any(local_method == 'KALMAN_BACKWARD_TAIL'):
                        stats['TOP_EXTRAP_SEGMENTS'] += 1
                else:
                    stats['FRONT_RECON_FRAMES'] += len(kept)
                    stats['FRONT_KALMAN_EPISODES'] += 1
                    if np.any(local_method == 'KALMAN_FORWARD_TAIL') or np.any(local_method == 'KALMAN_BACKWARD_TAIL'):
                        stats['FRONT_EXTRAP_SEGMENTS'] += 1

            if len(rejected):
                df.loc[rejected, settings.MEAS_COLS] = np.nan
                state[rejected] = 'LOST'
                stats['LOST_FRAMES'] += len(rejected)
                stats['LOST_SEGMENTS'] += 1

    df['State2'] = state
    df['Filled_Type'] = filled_type
    df['Recovery_Applied'] = recovery_applied
    df['Front_X_Sanity_Failed'] = front_x_sanity_failed
    df['Recovery_Left_Anchor_Index'] = anchor_left_index
    df['Recovery_Right_Anchor_Index'] = anchor_right_index
    df['Recovery_Right_Used_Index'] = anchor_right_used_index
    df['Anchor_Left_Count'] = anchor_left_count
    df['Anchor_Right_Count'] = anchor_right_count
    df['Anchor_Left_Search_Window'] = anchor_left_window
    df['Anchor_Right_Search_Window'] = anchor_right_window
    df['Anchor_Strategy'] = anchor_strategy
    df['Hidden_Axis_Model'] = hidden_axis_model
    df['Hidden_Episode_ID'] = hidden_episode_id
    df['Hidden_Axis_Filtered_Value'] = hidden_filtered_value
    df['Hidden_Axis_Smoothed_Value'] = hidden_smoothed_value
    df['Hidden_Axis_Filtered_Velocity'] = hidden_filtered_velocity
    df['Hidden_Axis_Smoothed_Velocity'] = hidden_smoothed_velocity
    df['Hidden_Axis_Posterior_Std'] = hidden_posterior_std
    df['Hidden_Axis_Initial_Velocity'] = hidden_initial_velocity
    df['Hidden_Axis_Initial_Velocity_Std'] = hidden_initial_velocity_std
    df['Hidden_Axis_Innovation_Weight'] = hidden_innovation_weight
    df['Hidden_Anchor_Weight_Median'] = hidden_anchor_weight_median
    df['Hidden_Anchor_Weight_Min'] = hidden_anchor_weight_min
    df['Hidden_Stop_Reason'] = hidden_stop_reason
    df['Hidden_Uncertainty_Limit_cm'] = hidden_uncertainty_limit
    df['Hidden_Has_Future_Raw'] = hidden_has_future_raw
    df['Hidden_Q_SigmaA'] = hidden_q_sigma_a
    df['Hidden_R_Sigma'] = hidden_r_sigma
    df['Z_Floor_Applied'] = z_floor_applied
    df['Top_Zone_Extension_Applied'] = top_zone_extension_applied

    df['Hidden_Axis_Velocity_cm_per_frame'] = hidden_smoothed_velocity
    df['Anchor_Left_Model_Valid'] = left_model_valid
    df['Anchor_Right_Model_Valid'] = right_model_valid
    df['Anchor_Left_TheilSen_Slope_Raw'] = left_slope_raw
    df['Anchor_Right_TheilSen_Slope_Raw'] = right_slope_raw
    df['Anchor_Left_TheilSen_Slope_Used'] = left_slope_used
    df['Anchor_Right_TheilSen_Slope_Used'] = right_slope_used
    df['Anchor_Left_TheilSen_Residual_MAD_cm'] = left_residual_mad
    df['Anchor_Right_TheilSen_Residual_MAD_cm'] = right_residual_mad
    df['Anchor_Left_TheilSen_Pair_Count'] = left_pair_count
    df['Anchor_Right_TheilSen_Pair_Count'] = right_pair_count
    df['Anchor_Left_TheilSen_Slope_Clipped'] = left_slope_clipped
    df['Anchor_Right_TheilSen_Slope_Clipped'] = right_slope_clipped
    return df, stats


def validate_reconstruction_invariants(df):
    """hybrid hidden-axis 복원의 핵심 불변조건을 검사한다."""
    mode = df['Recovery_Mode'].astype(str)
    top_mode = mode.eq(settings.RECOVERY_TOP_RAY)
    front_mode = mode.eq(settings.RECOVERY_FRONT_RAY)
    applied = df['Recovery_Applied'].fillna(False).astype(bool)
    filled = df['Filled_Type'].fillna('NONE').astype(str)

    bad_top_mode = top_mode & ~(
        df['Missing_Reason'].eq('FRONT_MISSING')
        & df['Status_Top'].eq('DETECTED')
        & df['Status_Front'].eq('LOST')
    )
    bad_front_mode = front_mode & ~(
        df['Missing_Reason'].eq('TOP_MISSING')
        & df['Status_Top'].eq('LOST')
        & df['Status_Front'].eq('DETECTED')
    )
    if bad_top_mode.any() or bad_front_mode.any():
        raise RuntimeError(
            'Recovery_Mode contradicts camera state: '
            f'TOP_RAY={int(bad_top_mode.sum())}, '
            f'FRONT_RAY={int(bad_front_mode.sum())}'
        )

    if not (applied == filled.ne('NONE')).all():
        raise RuntimeError('Recovery_Applied and Filled_Type are inconsistent.')
    if df.loc[applied, settings.MEAS_COLS].isna().any(axis=None):
        raise RuntimeError('A reconstructed frame contains non-finite coordinates.')
    if (applied & top_mode & ~filled.str.contains('TOP_RAY', regex=False)).any():
        raise RuntimeError('A TOP_RAY frame has an incompatible Filled_Type.')
    if (applied & front_mode & ~filled.str.contains('FRONT_RAY', regex=False)).any():
        raise RuntimeError('A FRONT_RAY frame has an incompatible Filled_Type.')

    models = df['Hidden_Axis_Model'].fillna('NONE').astype(str)
    allowed_models = {
        'NONE', 'LINEAR_INTERPOLATION',
        'KALMAN_RTS_EPISODE', 'KALMAN_FORWARD_TAIL', 'KALMAN_BACKWARD_TAIL',
    }
    unexpected = sorted(set(models.tolist()) - allowed_models)
    if unexpected:
        raise RuntimeError(f'Unexpected Hidden_Axis_Model values: {unexpected}')
    if (applied & models.eq('NONE')).any():
        raise RuntimeError('Applied reconstruction lacks a hidden-axis model.')

    stable = df['Anchor_Raw_Valid'].fillna(False).astype(bool).to_numpy()
    for col in ['Recovery_Left_Anchor_Index', 'Recovery_Right_Anchor_Index']:
        indices = pd.to_numeric(df[col], errors='coerce').fillna(-1).to_numpy(dtype=int)
        used = indices >= 0
        if used.any():
            if (indices[used] >= len(df)).any() or not stable[indices[used]].all():
                raise RuntimeError(f'{col} references a non-raw anchor.')

    posterior_std = pd.to_numeric(
        df['Hidden_Axis_Posterior_Std'], errors='coerce'
    )
    if (applied & (~np.isfinite(posterior_std) | (posterior_std < 0))).any():
        raise RuntimeError('Applied hidden-axis estimate lacks valid posterior std.')

    if (df['Front_X_Sanity_Failed'].fillna(False).astype(bool) & ~front_mode).any():
        raise RuntimeError('Front_X_Sanity_Failed appeared outside FRONT_RAY mode.')

    floor = df['Z_Floor_Applied'].fillna(False).astype(bool)
    if (floor & ~top_mode).any():
        raise RuntimeError('Z floor appeared outside TOP_RAY mode.')
    if (
        pd.to_numeric(df.loc[floor, 'Meas_Z'], errors='coerce')
        < settings.ZONE_TRACK_CENTER_Z_FLOOR - 1e-9
    ).any():
        raise RuntimeError('A Z-floor frame is below 3 cm.')

    # Reconstructed values must never become immutable raw anchors.
    if (applied & df['Anchor_Raw_Valid'].fillna(False).astype(bool)).any():
        raise RuntimeError('A reconstructed frame leaked into Anchor_Raw_Valid.')

    forbidden_front_zone_cols = [
        c for c in df.columns if str(c).startswith('Front_Zone')
    ]
    if forbidden_front_zone_cols:
        raise RuntimeError(
            f'Front-zone logic must not be present: {forbidden_front_zone_cols}'
        )
    return df
