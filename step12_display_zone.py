"""Step ⑫: display-only work-zone bridging for plots (Plot_Aux)."""

import numpy as np
import pandas as pd

from .. import settings


def _display_nearest_side_zone_evidence(
    base_in_zone, coord_available, start, end, side, search_frames, min_points
):
    n = len(base_in_zone)
    search_frames = int(search_frames)
    min_points = int(min_points)
    if search_frames <= 0 or min_points <= 0:
        return np.array([], dtype=int), False

    if side == 'left':
        lo = max(0, start - search_frames)
        candidates = np.arange(start - 1, lo - 1, -1, dtype=int)
    elif side == 'right':
        hi = min(n, end + 1 + search_frames)
        candidates = np.arange(end + 1, hi, dtype=int)
    else:
        raise ValueError(f'Unknown side: {side}')

    valid = candidates[coord_available[candidates]]
    chosen = valid[:min_points]
    stable = bool(
        len(chosen) >= min_points
        and base_in_zone[chosen].all()
    )
    return chosen, stable


def build_display_zone_state(
    df,
    max_gap_frames=None,
    stable_search_frames=None,
    stable_min_points=None,
):
    if max_gap_frames is None:
        max_gap_frames = settings.DISPLAY_ZONE_BRIDGE_MAX
    if stable_search_frames is None:
        stable_search_frames = settings.DISPLAY_ZONE_STABLE_SEARCH
    if stable_min_points is None:
        stable_min_points = settings.DISPLAY_ZONE_STABLE_MIN_POINTS_PER_SIDE
    required = [
        'Frame', 'In_Zone', 'Pre_RTS_Coord_Available', 'Coord_State',
        'Status_Top', 'Status_Front',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'Missing display-zone columns: {missing}')

    max_gap_frames = int(max_gap_frames)
    stable_search_frames = int(stable_search_frames)
    stable_min_points = int(stable_min_points)
    if max_gap_frames < 0:
        raise ValueError('max_gap_frames must be >= 0')
    if stable_search_frames <= 0:
        raise ValueError('stable_search_frames must be > 0')
    if stable_min_points <= 0:
        raise ValueError('stable_min_points must be > 0')

    n = len(df)
    frames = pd.to_numeric(df['Frame'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(frames).all() or np.any(np.diff(frames) <= 0):
        raise ValueError('Frame must be finite and strictly increasing.')

    base_in_zone = df['In_Zone'].fillna(False).to_numpy(dtype=bool).copy()
    coord_available = (
        df['Pre_RTS_Coord_Available'].fillna(False).to_numpy(dtype=bool).copy()
    )
    coord_state = df['Coord_State'].fillna('LOST').astype(str).to_numpy().copy()
    top_detected = df['Status_Top'].eq('DETECTED').to_numpy(dtype=bool)
    front_detected = df['Status_Front'].eq('DETECTED').to_numpy(dtype=bool)
    any_camera_detected = top_detected | front_detected

    candidate = (~coord_available) & (~base_in_zone) & (coord_state == 'LOST')
    bridge = np.zeros(n, dtype=bool)
    bridge_type = np.full(n, 'NONE', dtype=object)
    bridge_run_id = np.full(n, -1, dtype=int)
    bridge_run_length = np.zeros(n, dtype=int)
    left_evidence_count = np.zeros(n, dtype=int)
    right_evidence_count = np.zeros(n, dtype=int)

    next_run_id = 0
    idx = 0
    while idx < n:
        if not candidate[idx]:
            idx += 1
            continue

        end = idx
        while (
            end + 1 < n
            and candidate[end + 1]
            and frames[end + 1] - frames[end] == 1
        ):
            end += 1

        run_length = end - idx + 1
        left = idx - 1
        right = end + 1
        bounded_by_original_zone = bool(
            left >= 0
            and right < n
            and base_in_zone[left]
            and base_in_zone[right]
            and frames[idx] - frames[left] == 1
            and frames[right] - frames[end] == 1
        )

        accepted_type = 'NONE'
        left_points = np.array([], dtype=int)
        right_points = np.array([], dtype=int)

        if bounded_by_original_zone and run_length <= max_gap_frames:
            accepted_type = 'SHORT'
        elif bounded_by_original_zone and any_camera_detected[idx:end + 1].all():
            left_points, left_stable = _display_nearest_side_zone_evidence(
                base_in_zone, coord_available, idx, end, 'left',
                stable_search_frames, stable_min_points,
            )
            right_points, right_stable = _display_nearest_side_zone_evidence(
                base_in_zone, coord_available, idx, end, 'right',
                stable_search_frames, stable_min_points,
            )
            if left_stable and right_stable:
                accepted_type = 'STABLE_OBSERVED'

        if accepted_type != 'NONE':
            bridge[idx:end + 1] = True
            bridge_type[idx:end + 1] = accepted_type
            bridge_run_id[idx:end + 1] = next_run_id
            bridge_run_length[idx:end + 1] = run_length
            left_evidence_count[idx:end + 1] = len(left_points)
            right_evidence_count[idx:end + 1] = len(right_points)
            next_run_id += 1

        idx = end + 1

    display_in_zone = base_in_zone | bridge
    display_state = np.full(n, 'OUTSIDE_OR_UNKNOWN', dtype=object)
    display_state[base_in_zone] = 'IN_ZONE_3D'
    display_state[bridge] = 'IN_ZONE_BRIDGED'

    df['In_Zone_PreRTS'] = base_in_zone
    df['Display_In_Zone'] = display_in_zone
    df['Display_Zone_Bridge'] = bridge
    df['Display_Zone_Bridge_Type'] = bridge_type
    df['Display_Zone_State'] = display_state
    df['Display_Zone_Bridge_Run_ID'] = bridge_run_id
    df['Display_Zone_Bridge_Length'] = bridge_run_length
    df['Display_Zone_Left_Evidence_Count'] = left_evidence_count
    df['Display_Zone_Right_Evidence_Count'] = right_evidence_count
    return df


def validate_display_zone_invariants(
    df,
    max_gap_frames=None,
    stable_search_frames=None,
    stable_min_points=None,
):
    if max_gap_frames is None:
        max_gap_frames = settings.DISPLAY_ZONE_BRIDGE_MAX
    if stable_search_frames is None:
        stable_search_frames = settings.DISPLAY_ZONE_STABLE_SEARCH
    if stable_min_points is None:
        stable_min_points = settings.DISPLAY_ZONE_STABLE_MIN_POINTS_PER_SIDE
    base = df['In_Zone_PreRTS'].fillna(False).to_numpy(dtype=bool)
    bridge = df['Display_Zone_Bridge'].fillna(False).to_numpy(dtype=bool)
    bridge_type = df['Display_Zone_Bridge_Type'].fillna('NONE').astype(str).to_numpy()
    display = df['Display_In_Zone'].fillna(False).to_numpy(dtype=bool)
    available = (
        df['Pre_RTS_Coord_Available'].fillna(False).to_numpy(dtype=bool)
    )
    state = df['Coord_State'].fillna('LOST').astype(str).to_numpy()
    frames = pd.to_numeric(df['Frame'], errors='coerce').to_numpy(dtype=float)
    top_detected = df['Status_Top'].eq('DETECTED').to_numpy(dtype=bool)
    front_detected = df['Status_Front'].eq('DETECTED').to_numpy(dtype=bool)
    any_camera_detected = top_detected | front_detected

    if not np.array_equal(display, base | bridge):
        raise RuntimeError('Display_In_Zone must equal immutable In_Zone_PreRTS OR bridge.')
    if (bridge & available).any():
        raise RuntimeError('Display bridge must not overwrite a frame with finite pre-RTS coordinates.')
    if (bridge & (state != 'LOST')).any():
        raise RuntimeError('Display bridge must be limited to Coord_State=LOST frames.')
    if ((bridge_type != 'NONE') != bridge).any():
        raise RuntimeError('Display_Zone_Bridge_Type is inconsistent with Display_Zone_Bridge.')

    idx = 0
    while idx < len(df):
        if not bridge[idx]:
            idx += 1
            continue
        end = idx
        while (
            end + 1 < len(df)
            and bridge[end + 1]
            and frames[end + 1] - frames[end] == 1
            and bridge_type[end + 1] == bridge_type[idx]
        ):
            end += 1

        run_length = end - idx + 1
        kind = bridge_type[idx]
        if not (
            idx > 0
            and end + 1 < len(df)
            and base[idx - 1]
            and base[end + 1]
        ):
            raise RuntimeError('Display-zone bridge is not bounded by original in-zone frames.')

        if kind == 'SHORT':
            if run_length > int(max_gap_frames):
                raise RuntimeError(
                    f'SHORT display-zone bridge exceeds {max_gap_frames} frames: {run_length}'
                )
        elif kind == 'STABLE_OBSERVED':
            if not any_camera_detected[idx:end + 1].all():
                raise RuntimeError('STABLE_OBSERVED bridge contains a frame with both cameras LOST.')
            _, left_stable = _display_nearest_side_zone_evidence(
                base, available, idx, end, 'left',
                stable_search_frames, stable_min_points,
            )
            _, right_stable = _display_nearest_side_zone_evidence(
                base, available, idx, end, 'right',
                stable_search_frames, stable_min_points,
            )
            if not (left_stable and right_stable):
                raise RuntimeError('STABLE_OBSERVED bridge lacks stable in-zone evidence on both sides.')
        else:
            raise RuntimeError(f'Unknown Display_Zone_Bridge_Type: {kind}')

        idx = end + 1
    return df
