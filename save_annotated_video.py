"""Optional annotated video (display only, not used for coordinates)."""

import cv2
import numpy as np

from .. import settings


def draw_overlay(frame, calib, is_front=False):
    if calib is None: return frame
    cm, dc, rvec, tvec = calib['cam_matrix'], calib['dist_coeffs'], calib['rvec'], calib['tvec']
    axis_len = 15.0
    cv2.drawFrameAxes(frame, cm, dc, rvec, tvec, axis_len, 3)
    axis_3d = np.array([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]], dtype=np.float32)
    axis_px, _ = cv2.projectPoints(axis_3d, rvec, tvec, cm, dc)
    pts = np.int32(axis_px.reshape(-1, 2))
    cv2.putText(frame, "X", (pts[0][0]+5, pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.putText(frame, "Y", (pts[1][0]+5, pts[1][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, "Z", (pts[2][0]+5, pts[2][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    mx, my, mz = settings.MOTOR_CENTER_IN_ID0_CM
    if not is_front:
        hw, hh = settings.TOP_ZONE_W_CM / 2.0, settings.TOP_ZONE_H_CM / 2.0
        zone_3d = np.array([[mx-hw, my-hh, mz], [mx+hw, my-hh, mz],
                            [mx+hw, my+hh, mz], [mx-hw, my+hh, mz]], dtype=np.float32)
    else:
        hw, z_top, z_bot = settings.FRONT_ZONE_W_CM / 2.0, mz + settings.FRONT_ZONE_ABOVE_CM, mz - settings.FRONT_ZONE_BELOW_CM
        zone_3d = np.array([[mx-hw, my, z_bot], [mx+hw, my, z_bot],
                            [mx+hw, my, z_top], [mx-hw, my, z_top]], dtype=np.float32)
    zone_px, _ = cv2.projectPoints(zone_3d, rvec, tvec, cm, dc)
    pts = np.int32(zone_px.reshape(-1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
    opx = (int(calib['motor_origin_px'][0]), int(calib['motor_origin_px'][1]))
    cv2.circle(frame, opx, 6, (255, 255, 255), -1)
    return frame


def annotate_pair(
    res_top, res_front, online_w, calib_top, calib_front,
    dataset_is_wet, enable_refraction, sync_output_fps,
    frame_count, top_source_frame, top_time_sec, top_reused,
    front_source_frame, front_time_sec, front_reused,
    sync_status, sync_error_ms,
):
    ann_top, ann_front = res_top[0].plot(), res_front[0].plot()
    if np.all(np.isfinite(online_w)):
        p3d = np.array([online_w], dtype=np.float32)
        pt_t, _ = cv2.projectPoints(
            p3d, calib_top['rvec'], calib_top['tvec'],
            calib_top['cam_matrix'], calib_top['dist_coeffs']
        )
        pt_f, _ = cv2.projectPoints(
            p3d, calib_front['rvec'], calib_front['tvec'],
            calib_front['cam_matrix'], calib_front['dist_coeffs']
        )
        for image, point in (
            (ann_top, pt_t.reshape(2)), (ann_front, pt_f.reshape(2))
        ):
            if np.all(np.isfinite(point)) and np.all(np.abs(point) < 1e6):
                cv2.circle(
                    image, tuple(point.astype(int)), 8, (0, 255, 255), -1
                )

    ann_top = draw_overlay(ann_top, calib_top)
    ann_front = draw_overlay(ann_front, calib_front, is_front=True)
    mode_txt = (
        f"{'WET' if dataset_is_wet else 'DRY'} / "
        f"REFRACTION {'ON' if enable_refraction else 'OFF'} / "
        f'SYNC {sync_output_fps:.2f} fps'
    )
    mode_color = (0, 200, 255) if enable_refraction else (170, 170, 170)
    for image in (ann_top, ann_front):
        cv2.putText(
            image, mode_txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.1, (0, 0, 0), 6
        )
        cv2.putText(
            image, mode_txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.1, mode_color, 2
        )

    top_txt = (
        f'out {frame_count} | top src {top_source_frame} | '
        f't={top_time_sec:.3f}s | reused={top_reused}'
    )
    front_txt = (
        f'front src {front_source_frame} | t={front_time_sec:.3f}s | '
        f'reused={front_reused}'
    )
    sync_txt = f'{sync_status} | dt(front-top)={sync_error_ms:+.1f}ms'
    for image, line in ((ann_top, top_txt), (ann_front, front_txt)):
        cv2.putText(
            image, line, (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
            0.82, (0, 0, 0), 5
        )
        cv2.putText(
            image, line, (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
            0.82, (0, 255, 0), 2
        )
        cv2.putText(
            image, sync_txt, (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (0, 0, 0), 5
        )
        cv2.putText(
            image, sync_txt, (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (0, 255, 255), 2
        )
    return ann_top, ann_front


class AnnotatedVideoWriter:
    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.video_writer = None

    def write(self, ann_top, ann_front):
        h = min(ann_top.shape[0], ann_front.shape[0])

        def _fit(image):
            image2 = image[:h] if image.shape[0] >= h else cv2.copyMakeBorder(
                image, 0, h - image.shape[0], 0, 0, cv2.BORDER_CONSTANT
            )
            return cv2.resize(
                image2, (0, 0),
                fx=settings.ANNOTATED_VIDEO_SCALE, fy=settings.ANNOTATED_VIDEO_SCALE
            )

        combined = cv2.hconcat([_fit(ann_top), _fit(ann_front)])
        if self.video_writer is None:
            writer_fps = settings.ANNOTATED_VIDEO_FPS or self.fps
            self.video_writer = cv2.VideoWriter(
                self.path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                writer_fps,
                (combined.shape[1], combined.shape[0]),
            )
            print(
                f'Annotated video: {self.path} '
                f'({combined.shape[1]}x{combined.shape[0]}@{writer_fps:.2f})'
            )
        self.video_writer.write(combined)

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
