"""Step ⑪: work-zone state and GRAB labeling on pre-smoothing coordinates."""

import numpy as np
import pandas as pd

from .. import settings
from .ray_geometry import _in_zone_3d


def _first_long_true_run(mask, start, end, min_len):
    idx = start
    while idx <= end:
        if not mask[idx]:
            idx += 1
            continue
        run_end = idx
        while run_end + 1 <= end and mask[run_end + 1]:
            run_end += 1
        if run_end - idx + 1 >= min_len:
            return idx, idx + min_len - 1, run_end
        idx = run_end + 1
    return None, None, None


def classify_zone_grab(df):
    n = len(df)
    state = df['State2'].to_numpy(dtype=object).copy()

    coords = df[settings.MEAS_COLS].to_numpy(dtype=float)
    coord_available = np.isfinite(coords).all(axis=1)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    in_zone = np.array([
        _in_zone_3d(x[idx], y[idx], z[idx]) if coord_available[idx] else False
        for idx in range(n)
    ])

    final = np.full(n, 'LOST', dtype=object)
    detected_valid = coord_available & (state == 'DETECTED')
    sub_valid = coord_available & (state == 'SUB')
    final[detected_valid] = 'DETECTED'
    final[sub_valid] = 'SUB'
    final[detected_valid & in_zone] = 'DETECTED_ZONE'
    final[sub_valid & in_zone] = 'SUB_ZONE'

    both_lost = (
        df['Status_Top'].eq('LOST').to_numpy()
        & df['Status_Front'].eq('LOST').to_numpy()
    )
    unavailable = ~coord_available

    grab_event_id = np.full(n, -1, dtype=int)
    grab_anchor_index = np.full(n, -1, dtype=int)
    grab_event_start_index = np.full(n, -1, dtype=int)
    grab_loss_start_index = np.full(n, -1, dtype=int)
    grab_confirm_index = np.full(n, -1, dtype=int)
    grab_both_lost_end_index = np.full(n, -1, dtype=int)
    grab_run_end_index = np.full(n, -1, dtype=int)
    grab_anchor_frame = np.full(n, np.nan, dtype=float)
    grab_event_start_frame = np.full(n, np.nan, dtype=float)
    grab_loss_start_frame = np.full(n, np.nan, dtype=float)
    grab_confirm_frame = np.full(n, np.nan, dtype=float)
    grab_both_lost_end_frame = np.full(n, np.nan, dtype=float)
    grab_run_end_frame = np.full(n, np.nan, dtype=float)
    grab_inferred_onset = np.zeros(n, dtype=bool)

    n_grab = 0
    idx = 0
    while idx < n:
        if not unavailable[idx]:
            idx += 1
            continue

        end = idx
        while end + 1 < n and unavailable[end + 1]:
            end += 1

        anchor = idx - 1
        anchor_in_zone = bool(
            anchor >= 0
            and coord_available[anchor]
            and in_zone[anchor]
        )
        loss_start, confirm, loss_end = _first_long_true_run(
            both_lost, idx, end, min_len=settings.GRAB_MIN
        )

        if anchor_in_zone and confirm is not None:
            event_id = n_grab + 1
            event_slice = slice(idx, end + 1)

            final[event_slice] = 'GRAB'
            grab_event_id[event_slice] = event_id
            grab_anchor_index[event_slice] = anchor
            grab_event_start_index[event_slice] = idx
            grab_loss_start_index[event_slice] = loss_start
            grab_confirm_index[event_slice] = confirm
            grab_both_lost_end_index[event_slice] = loss_end
            grab_run_end_index[event_slice] = end
            if 'Frame' in df.columns:
                grab_anchor_frame[event_slice] = float(df.at[anchor, 'Frame'])
                grab_event_start_frame[event_slice] = float(df.at[idx, 'Frame'])
                grab_loss_start_frame[event_slice] = float(df.at[loss_start, 'Frame'])
                grab_confirm_frame[event_slice] = float(df.at[confirm, 'Frame'])
                grab_both_lost_end_frame[event_slice] = float(df.at[loss_end, 'Frame'])
                grab_run_end_frame[event_slice] = float(df.at[end, 'Frame'])
            grab_inferred_onset[idx] = True
            n_grab += 1

        idx = end + 1

    df['Coord_State'] = final
    df['In_Zone'] = in_zone
    df['Pre_RTS_Coord_Available'] = coord_available
    df['Both_Cameras_Lost'] = both_lost
    df['Grab_Event_ID'] = grab_event_id
    df['Grab_Anchor_Index'] = grab_anchor_index
    df['Grab_Event_Start_Index'] = grab_event_start_index
    df['Grab_Loss_Start_Index'] = grab_loss_start_index
    df['Grab_Confirm_Index'] = grab_confirm_index
    df['Grab_Both_Lost_End_Index'] = grab_both_lost_end_index
    df['Grab_Run_End_Index'] = grab_run_end_index
    df['Grab_Anchor_Frame'] = grab_anchor_frame
    df['Grab_Event_Start_Frame'] = grab_event_start_frame
    df['Grab_Loss_Start_Frame'] = grab_loss_start_frame
    df['Grab_Confirm_Frame'] = grab_confirm_frame
    df['Grab_Both_Lost_End_Frame'] = grab_both_lost_end_frame
    df['Grab_Run_End_Frame'] = grab_run_end_frame
    df['Grab_Inferred_Onset'] = grab_inferred_onset
    return df, n_grab


def validate_grab_invariants(df):
    required = [
        'Frame', 'Coord_State', 'In_Zone', 'Pre_RTS_Coord_Available',
        'Both_Cameras_Lost', 'Grab_Event_ID', 'Grab_Anchor_Index',
        'Grab_Event_Start_Index', 'Grab_Loss_Start_Index',
        'Grab_Confirm_Index', 'Grab_Both_Lost_End_Index',
        'Grab_Run_End_Index', 'Grab_Inferred_Onset',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'Missing GRAB validation columns: {missing}')

    n = len(df)
    frames = pd.to_numeric(df['Frame'], errors='coerce').to_numpy(dtype=float)
    state = df['Coord_State'].fillna('LOST').astype(str).to_numpy()
    grab = state == 'GRAB'
    both_lost = df['Both_Cameras_Lost'].fillna(False).to_numpy(dtype=bool)
    available = df['Pre_RTS_Coord_Available'].fillna(False).to_numpy(dtype=bool)
    event_id = pd.to_numeric(
        df['Grab_Event_ID'], errors='coerce'
    ).fillna(-1).to_numpy(dtype=int)
    onset = df['Grab_Inferred_Onset'].fillna(False).to_numpy(dtype=bool)

    if (grab & available).any():
        raise RuntimeError('GRAB frame unexpectedly has a finite pre-RTS coordinate.')
    if (grab != (event_id >= 0)).any():
        raise RuntimeError('Coord_State=GRAB and Grab_Event_ID assignment are inconsistent.')
    if (onset & ~grab).any():
        raise RuntimeError('Grab_Inferred_Onset appeared outside a GRAB frame.')

    positive_ids = sorted(set(event_id[event_id >= 0].tolist()))
    if positive_ids != list(range(1, len(positive_ids) + 1)):
        raise RuntimeError(f'Grab_Event_ID must be consecutive from 1: {positive_ids}')

    for eid in positive_ids:
        indices = np.flatnonzero(event_id == eid)
        if np.any(np.diff(indices) != 1) or np.any(np.diff(frames[indices]) != 1):
            raise RuntimeError(f'GRAB event {eid} is not a contiguous frame run.')

        start = int(indices[0])
        end = int(indices[-1])
        if available[indices].any():
            raise RuntimeError(f'GRAB event {eid} contains a finite coordinate.')
        if start <= 0:
            raise RuntimeError(f'GRAB event {eid} has no preceding anchor frame.')
        if end + 1 < n and not available[end + 1]:
            raise RuntimeError(f'GRAB event {eid} does not cover the full unavailable run.')

        def unique_metadata(col):
            values = pd.to_numeric(
                df.loc[indices, col], errors='coerce'
            ).to_numpy(dtype=int)
            if len(set(values.tolist())) != 1:
                raise RuntimeError(f'GRAB event {eid} has inconsistent {col}.')
            return int(values[0])

        anchor = unique_metadata('Grab_Anchor_Index')
        event_start = unique_metadata('Grab_Event_Start_Index')
        loss_start = unique_metadata('Grab_Loss_Start_Index')
        confirm = unique_metadata('Grab_Confirm_Index')
        both_lost_end = unique_metadata('Grab_Both_Lost_End_Index')
        run_end = unique_metadata('Grab_Run_End_Index')

        if event_start != start or run_end != end:
            raise RuntimeError(f'GRAB event {eid} event-boundary metadata is inconsistent.')
        if anchor != start - 1:
            raise RuntimeError(f'GRAB event {eid} anchor must immediately precede the outage.')
        if not available[anchor] or not bool(df.at[anchor, 'In_Zone']):
            raise RuntimeError(
                f'GRAB event {eid} anchor must be a finite in-zone pre-RTS frame.'
            )
        if not (start <= loss_start <= confirm <= both_lost_end <= end):
            raise RuntimeError(f'GRAB event {eid} both-lost metadata order is invalid.')
        if confirm != loss_start + settings.GRAB_MIN - 1:
            raise RuntimeError(f'GRAB event {eid} confirmation index is not the GRAB_MIN-th loss frame.')
        if not both_lost[loss_start:confirm + 1].all():
            raise RuntimeError(f'GRAB event {eid} lacks GRAB_MIN consecutive both-lost evidence.')
        if onset[indices].sum() != 1 or not onset[start]:
            raise RuntimeError(f'GRAB event {eid} must have one onset flag at the event start.')

    return df
