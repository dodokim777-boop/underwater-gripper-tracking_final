"""Step ④: candidate gating (ray gap -> jump -> Mahalanobis) and 7-of-10 track confirmation."""

from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import inv

from .. import settings


def build_Q(sigma_a, dt=1.0):
    q = float(sigma_a) ** 2
    a, b, c = q * dt**4 / 4.0, q * dt**3 / 2.0, q * dt**2
    I3 = np.eye(3)
    return np.block([[a * I3, b * I3],
                     [b * I3, c * I3]])


class OnlineTracker:
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
        self.accepted_hist.append(bool(accepted))
        if self.state != "CONFIRMED" and sum(self.accepted_hist) >= settings.CONFIRM_M:
            self.state = "CONFIRMED"

    def _register_stereo_no_accept(self, step_elapsed):
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
        frame_idx = float(frame_idx)
        meas = np.full(3, np.nan)

        if self.last_step_frame is None:
            step_elapsed = 1.0
        else:
            step_elapsed = max(frame_idx - float(self.last_step_frame), 0.0)
        self.last_step_frame = frame_idx

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

        if not both_detected:
            self._clear_stereo_no_accept()

        if candidate_3d is None:
            if both_detected:
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

            self.reject_streak = 0
            return self._out(
                meas=meas,
                status="NO_CANDIDATE",
                reason=observation_reason,
                updated=False,
            )

        cand = np.asarray(candidate_3d, dtype=float).reshape(3)
        if not np.all(np.isfinite(cand)):
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

        displacement = float(np.linalg.norm(cand - self.kf.x[:3]))
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
        if updated:
            online = self.kf.x[:3].copy()
            pos_std = np.sqrt(np.maximum(np.diag(self.kf.P)[:3], 0.0))
        else:
            online = np.full(3, np.nan)
            pos_std = np.full(3, np.nan)
        return meas, online, pos_std, status, self.state, reason, bool(updated)


def ray_gap_gate(selected_tri):
    if selected_tri is not None and np.all(
        np.isfinite(selected_tri.point_world)
    ):
        if selected_tri.ray_gap_cm > settings.RAY_GAP_MAX_CM:
            return None, 'RAY_GAP_EXCEEDED'
        return np.asarray(selected_tri.point_world, dtype=float), 'NONE'
    return None, 'TRIANGULATION_FAILED'
