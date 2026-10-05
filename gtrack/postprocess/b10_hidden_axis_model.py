# -*- coding: utf-8 -*-
"""
⑩ 단일시점 구간 좌표 복원 — 비관측 축(숨은 축) 추정 모델 (다이어그램 2쪽)

반복 결측 · 300프레임 초과 양측 결측 · 단측 결측에서 쓰는 1차원 모델 부품
  - _select_sparse_anchor_sets : 결측 앞뒤 30(부족하면 60) 프레임 안의 실제 두 시점 관측 수집
  - _fit_theil_sen_model       : Theil–Sen 강건 기울기 → 초기 속도와 그 불확실성
  - _run_1d_kalman_rts         : 1차원 등속도 Kalman filter + Huber 가중 위치 관측 + 고정구간 RTS
  - _run_1d_smoother_with_direction : 말단(순방향) / 선두(시간 역전) 처리
"""

import numpy as np

from .. import settings


def _collect_sparse_raw_indices(
    raw_valid, frames, hard_barrier, start, end, side, window_frames
):
    """결측 run 주변 window에서 실제 raw stereo frame만 모은다.

    중간의 TENTATIVE_HOLD, REJECTED_*, 다른 single-view 결측, 이미 복원된 frame은
    anchor가 아니므로 단순히 건너뛴다. 단, 장기 양안 LOST barrier나 Frame 단절은
    넘지 않는다.
    """
    out = []
    window_frames = float(window_frames)
    if side == 'LEFT':
        boundary_frame = frames[start]
        prev_frame = boundary_frame
        idx = start - 1
        while idx >= 0:
            if prev_frame - frames[idx] != 1:
                break
            if boundary_frame - frames[idx] > window_frames:
                break
            if hard_barrier[idx]:
                break
            if raw_valid[idx]:
                out.append(idx)
            prev_frame = frames[idx]
            idx -= 1
        out.reverse()
    elif side == 'RIGHT':
        boundary_frame = frames[end]
        prev_frame = boundary_frame
        idx = end + 1
        while idx < len(raw_valid):
            if frames[idx] - prev_frame != 1:
                break
            if frames[idx] - boundary_frame > window_frames:
                break
            if hard_barrier[idx]:
                break
            if raw_valid[idx]:
                out.append(idx)
            prev_frame = frames[idx]
            idx += 1
    else:
        raise ValueError(f'Unknown side: {side}')
    return np.asarray(out, dtype=int)


def _select_sparse_anchor_sets(raw_valid, frames, hard_barrier, start, end):
    """좌우 30 frame에서 raw anchor를 찾고 부족한 쪽만 60까지 확장한다.

    핵심 구분:
    - 양쪽에 실제 Hampel-filtered stereo raw anchor가 하나씩이라도 있으면
      가장 가까운 좌우 raw boundary 사이를 단순 선형 내삽한다.
    - 한쪽밖에 없으면 그 방향에 실제 raw 점이 ``ANCHOR_MODEL_MIN_POINTS``개
      이상일 때만 Theil-Sen 국소 기울기를 적합하여 제한 외삽한다.
    - 중간 결측과 이전 복원값은 anchor로 절대 사용하지 않는다.
    """
    left = _collect_sparse_raw_indices(
        raw_valid, frames, hard_barrier, start, end, 'LEFT', settings.ANCHOR_SEARCH_INITIAL
    )
    right = _collect_sparse_raw_indices(
        raw_valid, frames, hard_barrier, start, end, 'RIGHT', settings.ANCHOR_SEARCH_INITIAL
    )
    left_window = settings.ANCHOR_SEARCH_INITIAL
    right_window = settings.ANCHOR_SEARCH_INITIAL

    # one-sided Theil-Sen 모델 가능성을 확보하기 위해 raw 점이 3개 미만인 쪽만
    # 60 frame까지 한 번 확장한다. two-sided interpolation은 실제 raw 한 점씩이면 된다.
    if len(left) < settings.ANCHOR_MODEL_MIN_POINTS:
        left = _collect_sparse_raw_indices(
            raw_valid, frames, hard_barrier, start, end,
            'LEFT', settings.ANCHOR_SEARCH_EXTENDED,
        )
        left_window = settings.ANCHOR_SEARCH_EXTENDED

    if len(right) < settings.ANCHOR_MODEL_MIN_POINTS:
        right = _collect_sparse_raw_indices(
            raw_valid, frames, hard_barrier, start, end,
            'RIGHT', settings.ANCHOR_SEARCH_EXTENDED,
        )
        right_window = settings.ANCHOR_SEARCH_EXTENDED

    left_present = len(left) >= 1
    right_present = len(right) >= 1
    left_model_ok = len(left) >= settings.ANCHOR_MODEL_MIN_POINTS
    right_model_ok = len(right) >= settings.ANCHOR_MODEL_MIN_POINTS

    if left_present and right_present:
        strategy = 'TWO_SIDED'
    elif left_model_ok:
        strategy = 'LEFT_ONLY'
    elif right_model_ok:
        strategy = 'RIGHT_ONLY'
    else:
        strategy = 'INSUFFICIENT'

    return left, right, left_window, right_window, strategy


def _fit_theil_sen_model(frames, values, vmax, sigma_meas=None, sigma_a=None):
    """실제 raw history에서 episode 시작 velocity prior를 강건하게 추정한다.

    Theil-Sen은 이후 결측을 직선으로 외삽하는 방법이 아니다. episode 시작 시
    velocity mean을 한 번 정하고, pairwise slope dispersion을 초기 velocity
    covariance로 전달하는 용도만 사용한다.
    """
    x = np.asarray(frames, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < settings.ANCHOR_MODEL_MIN_POINTS:
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x, y = unique_x, y[unique_idx]
    if len(x) < settings.ANCHOR_MODEL_MIN_POINTS:
        return None

    slopes = []
    for i in range(len(x) - 1):
        dt = x[i + 1:] - x[i]
        ok = dt > 0
        if ok.any():
            slopes.extend(((y[i + 1:][ok] - y[i]) / dt[ok]).tolist())
    slopes = np.asarray(slopes, dtype=float)
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) == 0:
        return None

    slope_raw = float(np.median(slopes))
    slope_used = float(np.clip(slope_raw, -float(vmax), float(vmax)))
    slope_clipped = bool(not np.isclose(slope_raw, slope_used, rtol=0.0, atol=1e-12))

    slope_center = float(np.median(slopes))
    slope_mad = float(1.4826 * np.median(np.abs(slopes - slope_center)))
    span = float(np.max(x) - np.min(x))
    measurement_floor = 0.0
    if sigma_meas is not None and np.isfinite(sigma_meas):
        measurement_floor = float(np.sqrt(2.0) * float(sigma_meas) / max(span, 1.0))
    process_floor = float(sigma_a) if sigma_a is not None and np.isfinite(sigma_a) else 0.0
    velocity_std = max(slope_mad, measurement_floor, process_floor, 1e-6)

    origin_frame = float(np.median(x))
    origin_value = float(np.median(y - slope_used * (x - origin_frame)))
    fitted = origin_value + slope_used * (x - origin_frame)
    residual = y - fitted
    residual_center = float(np.median(residual))
    residual_mad = float(1.4826 * np.median(np.abs(residual - residual_center)))

    return {
        'slope_raw': slope_raw,
        'slope_used': slope_used,
        'slope_clipped': slope_clipped,
        'slope_mad': slope_mad,
        'velocity_std': velocity_std,
        'origin_frame': origin_frame,
        'origin_value': origin_value,
        'residual_mad': residual_mad,
        'point_count': int(len(x)),
        'pair_count': int(len(slopes)),
        'span_frames': span,
    }


def _cv_1d_transition_and_Q(dt, sigma_a):
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f'Invalid dt for 1D CV model: {dt}')
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    q = float(max(sigma_a, 1e-9)) ** 2
    Q = q * np.array([
        [0.25 * dt**4, 0.5 * dt**3],
        [0.5 * dt**3, dt**2],
    ], dtype=float)
    return F, Q


def _robust_position_update(x_pred, P_pred, measurement, measurement_var):
    """Huber innovation weighting을 적용한 1D position Kalman update."""
    H = np.array([[1.0, 0.0]], dtype=float)
    z = float(measurement)
    R0 = float(max(measurement_var, 1e-12))
    innovation = z - float(H @ x_pred)
    S0 = float(H @ P_pred @ H.T + R0)
    d = innovation / np.sqrt(max(S0, 1e-12))
    abs_d = abs(float(d))
    weight = 1.0 if abs_d <= settings.HIDDEN_HUBER_K else settings.HIDDEN_HUBER_K / abs_d
    R_eff = R0 / max(weight**2, 1e-12)

    S = float(H @ P_pred @ H.T + R_eff)
    K = (P_pred @ H.T) / max(S, 1e-12)
    x_upd = x_pred + K[:, 0] * innovation
    I = np.eye(2)
    KH = K @ H
    # Joseph form for numerical stability.
    P_upd = (I - KH) @ P_pred @ (I - KH).T + K * R_eff @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)
    return x_upd, P_upd, float(weight), float(innovation), float(np.sqrt(max(S, 0.0)))


def _initial_1d_state(
    sequence_frames, observation_values, observation_mask,
    history_frames, history_values, sigma_meas, sigma_a, vmax,
):
    """sequence 시작 상태 [position, velocity]와 covariance를 raw history에서 만든다."""
    sequence_frames = np.asarray(sequence_frames, dtype=float)
    obs_values = np.asarray(observation_values, dtype=float)
    obs_mask = np.asarray(observation_mask, dtype=bool)
    observed = np.flatnonzero(obs_mask & np.isfinite(obs_values))
    if len(observed) == 0:
        return None

    model = _fit_theil_sen_model(
        history_frames,
        history_values,
        vmax=vmax,
        sigma_meas=sigma_meas,
        sigma_a=sigma_a,
    )
    if model is None:
        velocity = 0.0
        velocity_std = max(float(vmax), float(sigma_a), 1e-3)
    else:
        velocity = float(model['slope_used'])
        velocity_std = float(model['velocity_std'])

    first_obs = int(observed[0])
    dt_back = float(sequence_frames[first_obs] - sequence_frames[0])
    position_at_start = float(obs_values[first_obs]) - velocity * dt_back
    position_var = float(sigma_meas)**2 + (dt_back * velocity_std)**2
    if dt_back > 0:
        _, Q_back = _cv_1d_transition_and_Q(dt_back, sigma_a)
        position_var += float(Q_back[0, 0])

    x0 = np.array([position_at_start, velocity], dtype=float)
    P0 = np.diag([
        max(position_var, float(sigma_meas)**2, 1e-8),
        max(velocity_std**2, 1e-8),
    ])
    return x0, P0, model


def _run_1d_kalman_rts(
    sequence_frames, observation_values, observation_mask,
    history_frames, history_values,
    sigma_a, sigma_meas, vmax,
):
    """raw position observation만 사용하는 1D CV Kalman + fixed-interval RTS."""
    t = np.asarray(sequence_frames, dtype=float)
    y = np.asarray(observation_values, dtype=float)
    obs = np.asarray(observation_mask, dtype=bool) & np.isfinite(y)
    n = len(t)
    if n == 0 or len(y) != n or len(obs) != n:
        raise ValueError('Invalid 1D smoother input lengths.')
    if np.any(np.diff(t) <= 0):
        raise ValueError('1D smoother frames must be strictly increasing.')

    init = _initial_1d_state(
        t, y, obs,
        history_frames=history_frames,
        history_values=history_values,
        sigma_meas=sigma_meas,
        sigma_a=sigma_a,
        vmax=vmax,
    )
    if init is None:
        return None
    x0, P0, init_model = init

    x_pred = np.zeros((n, 2), dtype=float)
    P_pred = np.zeros((n, 2, 2), dtype=float)
    x_filt = np.zeros((n, 2), dtype=float)
    P_filt = np.zeros((n, 2, 2), dtype=float)
    transition = np.zeros((max(n - 1, 0), 2, 2), dtype=float)
    innovation_weight = np.full(n, np.nan, dtype=float)
    innovation = np.full(n, np.nan, dtype=float)
    innovation_std = np.full(n, np.nan, dtype=float)

    for k in range(n):
        if k == 0:
            xp, Pp = x0.copy(), P0.copy()
        else:
            F, Q = _cv_1d_transition_and_Q(t[k] - t[k - 1], sigma_a)
            transition[k - 1] = F
            xp = F @ x_filt[k - 1]
            Pp = F @ P_filt[k - 1] @ F.T + Q
            Pp = 0.5 * (Pp + Pp.T)

        x_pred[k], P_pred[k] = xp, Pp
        if obs[k]:
            xu, Pu, w, nu, s = _robust_position_update(
                xp, Pp, y[k], float(sigma_meas)**2
            )
            x_filt[k], P_filt[k] = xu, Pu
            innovation_weight[k] = w
            innovation[k] = nu
            innovation_std[k] = s
        else:
            x_filt[k], P_filt[k] = xp, Pp

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for k in range(n - 2, -1, -1):
        F = transition[k]
        try:
            C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            C = P_filt[k] @ F.T @ np.linalg.pinv(P_pred[k + 1])
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - P_pred[k + 1]) @ C.T
        P_smooth[k] = 0.5 * (P_smooth[k] + P_smooth[k].T)

    return {
        'filtered_state': x_filt,
        'filtered_cov': P_filt,
        'smoothed_state': x_smooth,
        'smoothed_cov': P_smooth,
        'predicted_state': x_pred,
        'predicted_cov': P_pred,
        'innovation_weight': innovation_weight,
        'innovation': innovation,
        'innovation_std': innovation_std,
        'initial_model': init_model,
    }


def _run_1d_smoother_with_direction(
    sequence_frames, observation_values, observation_mask,
    history_frames, history_values,
    sigma_a, sigma_meas, vmax, reverse=False,
):
    """right-only leading gap은 time reversal로 대칭 처리한다."""
    if not reverse:
        return _run_1d_kalman_rts(
            sequence_frames, observation_values, observation_mask,
            history_frames, history_values,
            sigma_a=sigma_a, sigma_meas=sigma_meas, vmax=vmax,
        )

    t = np.asarray(sequence_frames, dtype=float)
    y = np.asarray(observation_values, dtype=float)
    obs = np.asarray(observation_mask, dtype=bool)
    t_rev = -t[::-1]
    y_rev = y[::-1]
    obs_rev = obs[::-1]
    h_frames = -np.asarray(history_frames, dtype=float)[::-1]
    h_values = np.asarray(history_values, dtype=float)[::-1]
    result = _run_1d_kalman_rts(
        t_rev, y_rev, obs_rev,
        h_frames, h_values,
        sigma_a=sigma_a, sigma_meas=sigma_meas, vmax=vmax,
    )
    if result is None:
        return None

    for key in ['filtered_state', 'smoothed_state', 'predicted_state']:
        arr = result[key][::-1].copy()
        arr[:, 1] *= -1.0
        result[key] = arr
    J = np.diag([1.0, -1.0])
    for key in ['filtered_cov', 'smoothed_cov', 'predicted_cov']:
        cov = result[key][::-1].copy()
        result[key] = np.einsum('ab,nbc,cd->nad', J, cov, J.T)
    for key in ['innovation_weight', 'innovation', 'innovation_std']:
        result[key] = result[key][::-1].copy()
    if result['initial_model'] is not None:
        result['initial_model'] = dict(result['initial_model'])
        result['initial_model']['slope_raw'] *= -1.0
        result['initial_model']['slope_used'] *= -1.0
    return result
