"""Step ⑭: frame-by-frame velocity from pre-smoothing coordinates (11-frame local slope)."""

import numpy as np
import pandas as pd

from .. import settings


def _time_seconds(df):
    if 'Time_sec' in df.columns and df['Time_sec'].notna().all():
        return df['Time_sec'].to_numpy(dtype=float), 'Time_sec'
    frame = pd.to_numeric(df['Frame'], errors='coerce').to_numpy(dtype=float)
    return (frame - frame[0]) / settings.VEL_FALLBACK_FPS, f'Frame/{settings.VEL_FALLBACK_FPS}'


def _coord_type_labels(filled_type, has_coord):
    out = np.full(len(filled_type), 'NONE', dtype=object)
    for i, (ft, ok) in enumerate(zip(filled_type, has_coord)):
        if not ok:
            continue
        ft = str(ft)
        if ft == 'NONE':
            out[i] = 'STEREO'
            continue
        side = 'TOP' if 'TOP_RAY' in ft else ('FRONT' if 'FRONT_RAY' in ft else 'UNKNOWN')
        if ft.startswith('LINEAR_INTERP'):
            method = 'LINEAR'
        elif ft.startswith('KALMAN_RTS_EPISODE'):
            method = 'KALMAN'
        elif ft.startswith('KALMAN_FORWARD_TAIL') or ft.startswith('KALMAN_BACKWARD_TAIL'):
            method = 'EXTRAP'
        else:
            method = 'UNKNOWN'
        out[i] = f'{side}_{method}'
    return out


def _axis_source_grades(coord_type):
    grades = np.full((len(coord_type), 3), 3, dtype=int)
    for i, ct in enumerate(coord_type):
        if ct == 'STEREO':
            grades[i] = 0
            continue
        side, _, method = str(ct).partition('_')
        hidden_grade = 2 if method in ('LINEAR', 'KALMAN') else 3
        g = np.array([1, 1, 1])
        if side == 'TOP':
            g[2] = hidden_grade
        elif side == 'FRONT':
            g[1] = hidden_grade
        else:
            g[:] = 3
        grades[i] = g
    return grades


def _local_slope_velocity(time_s, coords, usable, output_mask, half_window, grades=None,
                          max_ratio=None,
                          max_center_offset=None):
    if max_ratio is None:
        max_ratio = settings.VEL_MAX_NOISE_RATIO
    if max_center_offset is None:
        max_center_offset = settings.VEL_MAX_CENTER_OFFSET_FRAMES
    n = len(coords)
    k = int(half_window)
    s_full = k * (k + 1) * (2 * k + 1) / 3.0
    vel = np.full((n, 3), np.nan)
    ratio = np.full(n, np.nan)
    n_used = np.zeros(n, dtype=int)
    source = np.full((n, 3), -1, dtype=int)
    for t in np.flatnonzero(output_mask):
        lo, hi = max(0, t - k), min(n - 1, t + k)
        idx = np.arange(lo, hi + 1)
        idx = idx[usable[idx]]
        if idx.size < 2:
            continue
        j = (idx - t).astype(float)
        if not ((j < 0).any() and (j > 0).any()):
            continue
        j_mean = j.mean()
        s_used = float(np.sum((j - j_mean) ** 2))
        r = np.sqrt(s_full / s_used)
        if r > max_ratio or abs(j_mean) > max_center_offset:
            continue
        tt = time_s[idx]
        dt = tt - tt.mean()
        pts = coords[idx]
        vel[t] = dt @ (pts - pts.mean(axis=0)) / float(np.sum(dt ** 2))
        ratio[t] = r
        n_used[t] = idx.size
        if grades is not None:
            source[t] = grades[idx].max(axis=0)
    return vel, ratio, n_used, source


def compute_velocity_columns(df, half_window=None):
    if half_window is None:
        half_window = settings.VEL_HALF_WINDOW
    df = df.copy()
    time_s, time_source = _time_seconds(df)
    coords = df[settings.MEAS_COLS].to_numpy(dtype=float)
    state = df['Coord_State'].astype(str).to_numpy()
    floor = df['Z_Floor_Applied'].fillna(False).astype(bool).to_numpy()
    has_coord = np.isfinite(coords).all(axis=1) & ~np.isin(state, ['LOST', 'GRAB'])
    coord_type = _coord_type_labels(df['Filled_Type'].astype(str).to_numpy(), has_coord)

    usable_all = has_coord & ~floor
    usable_obs = usable_all & (coord_type == 'STEREO')
    output_mask = usable_all

    grades = _axis_source_grades(coord_type)
    v_all, r_all, n_all, src_all = _local_slope_velocity(
        time_s, coords, usable_all, output_mask, half_window, grades=grades)
    v_obs, r_obs, n_obs, _ = _local_slope_velocity(
        time_s, coords, usable_obs, output_mask, half_window)

    labels = np.array(settings.VEL_SOURCE_LABELS + [''], dtype=object)
    for a, ax in enumerate('xyz'):
        df[f'V{ax}_cm_s'] = v_all[:, a]
        df[f'V{ax}_obs_cm_s'] = v_obs[:, a]
        df[f'V{ax}_Source'] = labels[src_all[:, a]]
    df['Speed_3D_cm_s'] = np.linalg.norm(v_all, axis=1)
    df['Speed_3D_obs_cm_s'] = np.linalg.norm(v_obs, axis=1)
    worst = np.where((src_all >= 0).all(axis=1), src_all.max(axis=1), -1)
    df['Speed_3D_Source'] = labels[worst]
    df['V_Noise_Ratio'] = r_all
    df['V_obs_Noise_Ratio'] = r_obs
    df['V_N_Used'] = n_all
    df['V_obs_N_Used'] = n_obs
    df['Result_Coord_Type'] = coord_type
    df['Result_State'] = pd.Series(state).map(settings.RESULT_STATE_MAP).fillna(pd.Series(state)).to_numpy()
    df.attrs['velocity_time_source'] = time_source

    result_state = df['Result_State'].to_numpy()
    recovered_types = ['TOP_LINEAR', 'TOP_KALMAN', 'TOP_EXTRAP',
                       'FRONT_LINEAR', 'FRONT_KALMAN', 'FRONT_EXTRAP']
    detected_mismatch = np.sum((result_state == 'DETECTED') != (coord_type == 'STEREO'))
    recovered_mismatch = np.sum((result_state == 'RECOVERED') != np.isin(coord_type, recovered_types))
    df.attrs['state_type_mismatch'] = int(detected_mismatch + recovered_mismatch)
    if df.attrs['state_type_mismatch']:
        print(f"WARNING: State/Coord_Type mismatch frames = {df.attrs['state_type_mismatch']}")
    if 'UNKNOWN' in ' '.join(map(str, coord_type)):
        print('WARNING: unknown Filled_Type encountered while building Coord_Type')
    return df


def velocity_summary_rows(df):
    rows = [
        ('Velocity time source', df.attrs.get('velocity_time_source', '')),
        ('Velocity frames (recovered-inclusive)', int(df['Vx_cm_s'].notna().sum())),
        ('Velocity frames (observed-only)', int(df['Vx_obs_cm_s'].notna().sum())),
        ('State/Coord_Type mismatch frames', int(df.attrs.get('state_type_mismatch', 0))),
    ]
    for label, count in df['Result_Coord_Type'].value_counts().items():
        rows.append((f'Coord_Type::{label}', int(count)))
    src = df.loc[df['Speed_3D_cm_s'].notna(), 'Speed_3D_Source']
    for label, count in src.value_counts().items():
        rows.append((f'Speed_3D_Source::{label}', int(count)))
    return rows


def velocity_config_df():
    return pd.DataFrame([
        ('Result coordinates', 'pre-RTS Meas_X/Y/Z (Hampel + recovery); RTS result is Plot_Aux only'),
        ('VEL_HALF_WINDOW_frames', settings.VEL_HALF_WINDOW),
        ('VEL_MAX_NOISE_RATIO', settings.VEL_MAX_NOISE_RATIO),
        ('VEL_MAX_CENTER_OFFSET_frames', settings.VEL_MAX_CENTER_OFFSET_FRAMES),
        ('Velocity method',
         'least-squares slope of pre-RTS coordinates vs Time_s over [t-k, t+k] '
         '(Savitzky-Golay 1st derivative, polynomial order 1); frames missing inside the window are skipped'),
        ('Velocity window acceptance',
         '>=1 used frame on each side of t, noise ratio <= VEL_MAX_NOISE_RATIO, '
         '|mean used-frame offset| <= VEL_MAX_CENTER_OFFSET_frames'),
        ('Velocity input (recovered-inclusive)', 'frames with coordinates, excluding Z_Floor_Applied'),
        ('Velocity input (observed-only)', 'Coord_Type STEREO frames only'),
        ('Velocity output frames', 'State DETECTED/RECOVERED and not Z_Floor_Applied'),
        ('Velocity source rule',
         'per axis, lowest grade among used frames: OBSERVED > RAY_CONSTRAINED > MODEL_INTERP > MODEL_EXTRAP; '
         'Speed_3D_Source = lowest of three axes'),
        ('Velocity k selection basis',
         'largest k that preserves pre-capture rapid motion at the k=3 level within noise; '
         'k=7 underestimated new_3 pre-capture Z velocity by 15-20%'),
        ('Velocity slow-phase fluctuation', settings.VEL_SLOW_PHASE_FLUCTUATION_NOTE),
    ], columns=['Parameter', 'Value'])
