"""Input: open the top and front videos and pair frames on a common timeline."""

from dataclasses import dataclass
import warnings

import cv2
import numpy as np

from .. import settings


@dataclass(frozen=True)
class SampledVideoFrame:
    source_frame: int
    video_time_sec: float
    sync_time_sec: float
    frame: np.ndarray
    reused: bool


class SequentialNearestSampler:
    def __init__(self, cap, fps, frame_total, time_offset_sec=0.0):
        self.cap = cap
        self.fps = float(fps)
        self.frame_total = int(frame_total)
        self.time_offset_sec = float(time_offset_sec)
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f'Invalid FPS: {self.fps}')
        if self.frame_total <= 0:
            raise ValueError(f'Invalid frame count: {self.frame_total}')
        self.current_index = -1
        self.current_frame = None

    def _read_to(self, target_index):
        target_index = int(target_index)
        if target_index < self.current_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            self.current_index = target_index - 1
            self.current_frame = None

        while self.current_index < target_index:
            ret, frame = self.cap.read()
            if not ret:
                return None
            self.current_index += 1
            self.current_frame = frame
        return self.current_frame

    def nearest(self, target_sync_time_sec):
        target_sync_time_sec = float(target_sync_time_sec)
        target_video_time_sec = target_sync_time_sec - self.time_offset_sec
        if target_video_time_sec < -1e-9:
            return None

        target_index = int(np.floor(target_video_time_sec * self.fps + 0.5))
        if target_index < 0 or target_index >= self.frame_total:
            return None

        reused = target_index == self.current_index and self.current_frame is not None
        frame = self._read_to(target_index)
        if frame is None:
            return None

        video_time_sec = target_index / self.fps
        return SampledVideoFrame(
            source_frame=target_index + 1,
            video_time_sec=float(video_time_sec),
            sync_time_sec=float(video_time_sec + self.time_offset_sec),
            frame=frame,
            reused=bool(reused),
        )


def _validate_video_format(role, width, height, fps, frame_total):
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f'{role}: cannot read FPS (fps={fps})')
    if int(frame_total) <= 0:
        raise ValueError(f'{role}: cannot read frame count ({frame_total})')

    if (width, height) != settings.EXPECTED_RESOLUTION:
        msg = (
            f'{role}: resolution mismatch {width}x{height} '
            f'(expected {settings.EXPECTED_RESOLUTION[0]}x{settings.EXPECTED_RESOLUTION[1]})'
        )
        if settings.STRICT_VIDEO_FORMAT:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)

    if abs(fps - settings.EXPECTED_FPS) > settings.FPS_TOLERANCE:
        warnings.warn(
            f'{role}: FPS {fps:.4f} differs from {settings.EXPECTED_FPS:.2f}; '
            'continuing with nearest-time frame pairing.',
            RuntimeWarning,
        )


def _video_duration_sec(frame_total, fps):
    return max(int(frame_total) - 1, 0) / float(fps)


def open_videos():
    cap_top = cv2.VideoCapture(settings.TOP_VIDEO_PATH)
    cap_front = cv2.VideoCapture(settings.FRONT_VIDEO_PATH)
    if not cap_top.isOpened() or not cap_front.isOpened():
        raise FileNotFoundError(
            f"Cannot open videos: top={settings.TOP_VIDEO_PATH}, front={settings.FRONT_VIDEO_PATH}"
        )

    fps_top = float(cap_top.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_front = float(cap_front.get(cv2.CAP_PROP_FPS) or 0.0)
    width_top = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height_top = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    width_front = int(cap_front.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height_front = int(cap_front.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_total_top = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_total_front = int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    _validate_video_format('top', width_top, height_top, fps_top, frame_total_top)
    _validate_video_format('front', width_front, height_front, fps_front, frame_total_front)

    duration_top_sec = _video_duration_sec(frame_total_top, fps_top)
    duration_front_sec = _video_duration_sec(frame_total_front, fps_front)
    top_sync_start_sec = float(settings.TOP_TIME_OFFSET_SEC)
    front_sync_start_sec = float(settings.FRONT_TIME_OFFSET_SEC)
    top_sync_end_sec = top_sync_start_sec + duration_top_sec
    front_sync_end_sec = front_sync_start_sec + duration_front_sec
    overlap_start_sync_sec = max(top_sync_start_sec, front_sync_start_sec)
    overlap_end_sync_sec = min(top_sync_end_sec, front_sync_end_sec)
    if overlap_end_sync_sec < overlap_start_sync_sec:
        raise ValueError(
            'The two videos do not overlap after applying TOP/FRONT_TIME_OFFSET_SEC.'
        )

    sync_output_fps = (
        max(fps_top, fps_front)
        if settings.SYNC_OUTPUT_FPS is None else float(settings.SYNC_OUTPUT_FPS)
    )
    if not np.isfinite(sync_output_fps) or sync_output_fps <= 0.0:
        raise ValueError(f'Invalid SYNC_OUTPUT_FPS: {sync_output_fps}')

    if settings.SYNC_MAX_ERROR_SEC is None:
        sync_max_error_sec = (
            settings.SYNC_MAX_ERROR_FRAME_FRACTION / min(fps_top, fps_front)
        )
    else:
        sync_max_error_sec = float(settings.SYNC_MAX_ERROR_SEC)
    if not np.isfinite(sync_max_error_sec) or sync_max_error_sec <= 0.0:
        raise ValueError(f'Invalid sync tolerance: {sync_max_error_sec}')

    common_duration_sec = overlap_end_sync_sec - overlap_start_sync_sec
    sync_output_frames = int(np.floor(common_duration_sec * sync_output_fps + 1e-9)) + 1

    print(
        f"Input: top {width_top}x{height_top}@{fps_top:.4f} fps "
        f"({frame_total_top} frames) | "
        f"front {width_front}x{height_front}@{fps_front:.4f} fps "
        f"({frame_total_front} frames)"
    )
    print(
        f"Timeline: output={sync_output_fps:.4f} fps, "
        f"common={common_duration_sec:.3f}s/{sync_output_frames} frames, "
        f"tolerance={sync_max_error_sec*1000:.2f} ms"
    )
    if abs(fps_top - fps_front) > settings.FPS_TOLERANCE:
        print('   FPS differs: frames of the slower video are reused.')
    if frame_total_top != frame_total_front:
        print('   Frame counts differ: only the common time range is processed.')

    return dict(
        cap_top=cap_top, cap_front=cap_front,
        fps_top=fps_top, fps_front=fps_front,
        frame_total_top=frame_total_top, frame_total_front=frame_total_front,
        duration_top_sec=duration_top_sec, duration_front_sec=duration_front_sec,
        overlap_start_sync_sec=overlap_start_sync_sec,
        overlap_end_sync_sec=overlap_end_sync_sec,
        sync_output_fps=sync_output_fps,
        sync_max_error_sec=sync_max_error_sec,
        common_duration_sec=common_duration_sec,
        sync_output_frames=sync_output_frames,
    )
