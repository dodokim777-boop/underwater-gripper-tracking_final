# -*- coding: utf-8 -*-
"""
④ 후보 수용 판정 및 궤적 확정

순서: 광선 간 거리 게이트(RAY_GAP_MAX_CM) → 변위 게이트(JUMP_LIMIT) → Mahalanobis 게이트(GATE_CHI2)
      → 7-of-10 확정(CONFIRM_M of CONFIRM_N)
거부된 후보의 임시 예측은 폐기한다 (committed state를 앞으로 밀지 않음).
출력(raw 엑셀 이름): Track_State, Meas_Status, Missing_Reason, KF_X/Y/Z_cm, KF_Std_X/Y/Z_cm
"""

from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import inv

from .. import settings


def build_Q(sigma_a, dt=1.0):
    """등속 상태모델의 white-noise-acceleration covariance."""
    q = float(sigma_a) ** 2
    a, b, c = q * dt**4 / 4.0, q * dt**3 / 2.0, q * dt**2
    I3 = np.eye(3)
    return np.block([[a * I3, b * I3],
                     [b * I3, c * I3]])


class OnlineTracker:
    """시간 인지형 measurement validation gate.

    필터는 결측 좌표를 출력하거나 online 궤적을 외삽하지 않는다. 마지막으로
    수용된 3D 후보의 state와 covariance를 committed state로 보존하고, 새 후보가
    들어왔을 때만 기준 FPS로 환산한 실제 경과시간만큼 임시 전진시켜 게이트를 계산한다.

    - 후보가 없으면 committed state와 covariance를 앞으로 밀지 않는다.
    - 한쪽 카메라만 소실된 경우에는 track identity를 유지하며 reset하지 않는다.
      즉 single-view 상태가 오래 지속되어도 3D-gap만으로 CONFIRMED track을 버리지 않는다.
    - 양안 소실이 MAX_BOTH_LOST_FRAMES 이상 지속되면 실제 track-loss로 보고 reset한다.
    - 양쪽 카메라가 모두 DETECTED인데도 유효 3D를 연속해서 얻지 못하는 경우에만
      MAX_STEREO_NO_ACCEPT_FRAMES를 적용한다.
    - 후보가 거부되면 임시 prediction을 폐기한다.
    - 후보가 수용된 경우에만 update 결과를 committed state로 저장한다.
    - track confirmation은 영상 frame 기준이 아니라 실제 stereo candidate 시도 기준이다.
      최근 CONFIRM_N개의 stereo candidate 평가 중 gate 통과가 CONFIRM_M개 이상이면
      track을 CONFIRMED로 전환한다.
    - TOP_MISSING / FRONT_MISSING / BOTH_MISSING은 stereo 평가 기회가 없었던 것이므로
      confirmation history에 False로 기록하지 않는다.
    - Meas에는 CONFIRMED 이후 gate를 통과한 raw candidate만 기록한다.
    """

    def __init__(self, jump_limit_cm_per_frame, jump_limit_abs_cm):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.kf.R = np.diag(settings.SIG_GATE_MEAS ** 2)
        self.jump_limit_cm_per_frame = float(jump_limit_cm_per_frame)
        self.jump_limit_abs_cm = float(jump_limit_abs_cm)
        self.accepted_hist = deque(maxlen=settings.CONFIRM_N)
        self.last_step_frame = None
        self._reset_track(clear_history=True)

    @staticmethod
    def _transition(dt_frames):
        dt = float(dt_frames)
        return np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)],
        ])

    def _reset_filter_state(self):
        self.kf.x = np.zeros(6, dtype=float)
        self.kf.P = np.diag([1000., 1000., 1000., 100., 100., 100.])
        self.kf.F = self._transition(1.0)
        self.kf.Q = build_Q(settings.SIGMA_A_ONLINE, dt=1.0)
        self.initialized = False
        self.last_accepted_frame = None

    def _reset_track(self, clear_history=True):
        self._reset_filter_state()
        self.state = "LOST"
        self.both_lost_counter = 0.0
        self.stereo_no_accept_counter = 0.0
        self.reject_streak = 0
        if clear_history:
            self.accepted_hist.clear()

    def _init_kf(self, pos, frame_idx):
        self.kf.x = np.array([pos[0], pos[1], pos[2], 0., 0., 0.], dtype=float)
        self.kf.P = np.diag([1000., 1000., 1000., 100., 100., 100.])
        self.initialized = True
        self.last_accepted_frame = float(frame_idx)
        self.state = "TENTATIVE"
        self.reject_streak = 0

    def _register_confirmation_result(self, accepted):
        """실제 stereo candidate 평가가 있었던 경우에만 호출한다."""
        self.accepted_hist.append(bool(accepted))
        if self.state != "CONFIRMED" and sum(self.accepted_hist) >= settings.CONFIRM_M:
            self.state = "CONFIRMED"

    def _register_stereo_no_accept(self, step_elapsed):
        """양안 관측 기회가 있었지만 유효 3D를 얻지 못한 시간을 누적한다."""
        self.stereo_no_accept_counter += float(step_elapsed)

    def _clear_stereo_no_accept(self):
        self.stereo_no_accept_counter = 0.0

    def _temporary_prediction(self, frame_idx):
        if not self.initialized or self.last_accepted_frame is None:
            raise RuntimeError('Tracker must be initialized before prediction.')
        elapsed = float(frame_idx) - float(self.last_accepted_frame)
        if elapsed <= 1e-9:
            raise ValueError(
                f'Time indices must increase: current={frame_idx}, '
                f'last_accepted={self.last_accepted_frame}'
            )

        rounded = round(elapsed)
        if abs(elapsed - rounded) < 1e-7:
            elapsed = float(rounded)

        x_pred = self.kf.x.copy()
        P_pred = self.kf.P.copy()
        whole_steps = int(np.floor(elapsed))
        fractional_step = float(elapsed - whole_steps)

        F_step = self._transition(1.0)
        Q_step = build_Q(settings.SIGMA_A_ONLINE, dt=1.0)
        for _ in range(whole_steps):
            x_pred = F_step @ x_pred
            P_pred = F_step @ P_pred @ F_step.T + Q_step

        if fractional_step > 1e-9:
            F_frac = self._transition(fractional_step)
            Q_frac = build_Q(settings.SIGMA_A_ONLINE, dt=fractional_step)
            x_pred = F_frac @ x_pred
            P_pred = F_frac @ P_pred @ F_frac.T + Q_frac

        return elapsed, x_pred, P_pred

    def step(
        self, frame_idx, top_detected, front_detected,
        candidate_3d, observation_reason,
    ):
        """한 출력 시각을 처리하고 raw measurement 및 진단값을 반환한다."""
        frame_idx = float(frame_idx)
        meas = np.full(3, np.nan)

        if self.last_step_frame is None:
            step_elapsed = 1.0
        else:
            step_elapsed = max(frame_idx - float(self.last_step_frame), 0.0)
        self.last_step_frame = frame_idx

        # --------------------------------------------------------------
        # Track-loss 판단
        # --------------------------------------------------------------
        # 한쪽 카메라가 살아 있으면 동일 객체를 계속 추적 중인 것으로 보고
        # 3D measurement 공백만으로 CONFIRMED track을 reset하지 않는다.
        both_missing = (not bool(top_detected)) and (not bool(front_detected))
        both_detected = bool(top_detected) and bool(front_detected)

        self.both_lost_counter = (
            self.both_lost_counter + step_elapsed if both_missing else 0.0
        )
        if self.both_lost_counter >= settings.MAX_BOTH_LOST_FRAMES:
            self._reset_track(clear_history=True)
            return self._out(
                meas=meas,
                status="TRACK_BOTH_LOST_RESET",
                reason=observation_reason,
                updated=False,
            )

        # single-view 상태는 stereo 실패가 아니라 stereo 평가 기회 자체가 없는 상태다.
        # 따라서 confirmation history에 False를 넣지 않고,
        # stereo-no-accept counter도 누적하지 않는다.
        if not both_detected:
            self._clear_stereo_no_accept()

        if candidate_3d is None:
            if both_detected:
                # 양쪽 카메라는 보이는데 triangulation/geometry가 실패한 경우:
                # 실제 stereo 시도 실패로 confirmation history에 기록한다.
                self._register_confirmation_result(False)
                self._register_stereo_no_accept(step_elapsed)

                if self.stereo_no_accept_counter >= settings.MAX_STEREO_NO_ACCEPT_FRAMES:
                    self._reset_track(clear_history=True)
                    return self._out(
                        meas=meas,
                        status="TRACK_STEREO_NO_ACCEPT_RESET",
                        reason=observation_reason,
                        updated=False,
                    )

            # TOP_MISSING / FRONT_MISSING / BOTH_MISSING은 history를 건드리지 않는다.
            self.reject_streak = 0
            return self._out(
                meas=meas,
                status="NO_CANDIDATE",
                reason=observation_reason,
                updated=False,
            )

        cand = np.asarray(candidate_3d, dtype=float).reshape(3)
        if not np.all(np.isfinite(cand)):
            # candidate object는 있었지만 유효하지 않음 = 실제 stereo 시도 실패.
            self._register_confirmation_result(False)
            self._register_stereo_no_accept(step_elapsed)
            if self.stereo_no_accept_counter >= settings.MAX_STEREO_NO_ACCEPT_FRAMES:
                self._reset_track(clear_history=True)
                return self._out(
                    meas=meas,
                    status="TRACK_STEREO_NO_ACCEPT_RESET",
                    reason="TRIANGULATION_FAILED",
                    updated=False,
                )
            return self._out(
                meas=meas,
                status="NO_CANDIDATE",
                reason="TRIANGULATION_FAILED",
                updated=False,
            )

        if not self.initialized:
            self._init_kf(cand, frame_idx)
            self._clear_stereo_no_accept()
            self._register_confirmation_result(True)
            if self.state == "CONFIRMED":
                return self._out(cand.copy(), "TRIANGULATED", "NONE", updated=True)
            return self._out(meas, "TENTATIVE_HOLD", "TENTATIVE_HOLD", updated=True)

        elapsed, x_pred, P_pred = self._temporary_prediction(frame_idx)

        # 물리 gate는 마지막 수용 위치에서 현재 후보까지의 실제 변위를 검사한다.
        displacement = float(np.linalg.norm(cand - self.kf.x[:3]))
        # REV-06: 프레임 간격에 비례한 허용 변위가 장시간 공백에서 무한정 커지지 않도록
        # 기존 2.0 cm/frame 기준에 30 cm 절대 상한을 함께 적용한다.
        allowed_displacement = min(
            self.jump_limit_cm_per_frame * float(elapsed),
            self.jump_limit_abs_cm,
        )
        if displacement > allowed_displacement:
            self._register_confirmation_result(False)
            self._register_stereo_no_accept(step_elapsed)
            self.reject_streak += 1
            if self.reject_streak >= settings.MAX_REJECT_STREAK:
                self._reset_track(clear_history=True)
                return self._out(
                    meas, "REJECT_STREAK_RESET", "REJECT_STREAK_RESET", updated=False
                )
            if self.stereo_no_accept_counter >= settings.MAX_STEREO_NO_ACCEPT_FRAMES:
                self._reset_track(clear_history=True)
                return self._out(
                    meas, "TRACK_STEREO_NO_ACCEPT_RESET",
                    "TRACK_STEREO_NO_ACCEPT_RESET", updated=False
                )
            return self._out(meas, "REJECTED_JUMP", "REJECTED_JUMP", updated=False)

        innovation = cand - x_pred[:3]
        S = self.kf.H @ P_pred @ self.kf.H.T + self.kf.R
        d2 = float(innovation.T @ inv(S) @ innovation)
        if d2 > settings.GATE_CHI2:
            self._register_confirmation_result(False)
            self._register_stereo_no_accept(step_elapsed)
            self.reject_streak += 1
            if self.reject_streak >= settings.MAX_REJECT_STREAK:
                self._reset_track(clear_history=True)
                return self._out(
                    meas, "REJECT_STREAK_RESET", "REJECT_STREAK_RESET", updated=False
                )
            if self.stereo_no_accept_counter >= settings.MAX_STEREO_NO_ACCEPT_FRAMES:
                self._reset_track(clear_history=True)
                return self._out(
                    meas, "TRACK_STEREO_NO_ACCEPT_RESET",
                    "TRACK_STEREO_NO_ACCEPT_RESET", updated=False
                )
            return self._out(
                meas, "REJECTED_MAHALANOBIS", "REJECTED_MAHALANOBIS", updated=False
            )

        # 통과한 경우에만 임시 prediction과 update 결과를 committed state로 저장한다.
        self.kf.F = self._transition(1.0)
        self.kf.Q = build_Q(settings.SIGMA_A_ONLINE, dt=1.0)
        self.kf.x = x_pred
        self.kf.P = P_pred
        self.kf.update(cand)
        self.last_accepted_frame = frame_idx
        self.reject_streak = 0
        self._clear_stereo_no_accept()
        self._register_confirmation_result(True)

        if self.state == "CONFIRMED":
            return self._out(cand.copy(), "TRIANGULATED", "NONE", updated=True)
        return self._out(meas, "TENTATIVE_HOLD", "TENTATIVE_HOLD", updated=True)

    def _out(self, meas, status, reason, updated):
        # Online_*/pos_std는 실제 update가 성공한 frame에만 값을 낸다.
        if updated:
            online = self.kf.x[:3].copy()
            pos_std = np.sqrt(np.maximum(np.diag(self.kf.P)[:3], 0.0))
        else:
            online = np.full(3, np.nan)
            pos_std = np.full(3, np.nan)
        return meas, online, pos_std, status, self.state, reason, bool(updated)


def ray_gap_gate(selected_tri):
    """④ 첫 단계: 광선 간 거리 게이트.

    두 광선 최근접 거리(Ray_Gap_cm)가 RAY_GAP_MAX_CM을 넘으면 후보를 거부한다.
    반환: (candidate_3d 또는 None, observation_reason)
    """
    if selected_tri is not None and np.all(
        np.isfinite(selected_tri.point_world)
    ):
        if selected_tri.ray_gap_cm > settings.RAY_GAP_MAX_CM:
            return None, 'RAY_GAP_EXCEEDED'
        return np.asarray(selected_tri.point_world, dtype=float), 'NONE'
    return None, 'TRIANGULATION_FAILED'
