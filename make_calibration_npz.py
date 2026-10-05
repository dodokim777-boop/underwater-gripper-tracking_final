"""Calibration NPZ: detect the ArUco marker automatically and save camera extrinsics (R, t)."""

import os
import time

import cv2
import numpy as np

from main_code import settings


def marker_object_points():
    half = settings.MARKER_SIZE_CM / 2.0
    return np.array([[-half,  half, 0],
                     [ half,  half, 0],
                     [ half, -half, 0],
                     [-half, -half, 0]], dtype=np.float32)


def grab_frame(video_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'Cannot read frame {frame_index}: {video_path}')
    return frame


def detect_marker(frame, pick_rightmost=False):
    aruco_dict = cv2.aruco.getPredefinedDictionary(settings.ARUCO_DICT_TYPE)
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners_list, ids, _ = detector.detectMarkers(frame)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        corners_list, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)
    if ids is None or len(ids) == 0:
        return None

    candidates = []
    for c, mid in zip(corners_list, ids.flatten()):
        pts = c.reshape(4, 2).astype(np.float32)
        candidates.append((pts[:, 0].mean(), pts, int(mid)))
    if pick_rightmost:
        candidates.sort(key=lambda x: x[0])
        _, pts, mid = candidates[-1]
    else:
        _, pts, mid = candidates[0]
    return pts, mid


def solve_marker_pose(corners, cam_matrix, dist_coeffs):
    obj_pts = marker_object_points()
    img_pts = np.asarray(corners, dtype=np.float32)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_matrix, dist_coeffs)
    if not ok:
        raise RuntimeError('solvePnP failed.')
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1)


def marker_to_world(R, t, use_wall_transform):
    if not use_wall_transform:
        return R, t
    return R @ settings.R_FW, R @ settings.t_FW + t


def draw_check_image(frame, corners, cam_matrix, dist_coeffs, R_world, t_world):
    vis = frame.copy()
    pts = np.asarray(corners, dtype=np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
    for i, (px, py) in enumerate(pts):
        cv2.putText(vis, str(i + 1), (int(px) + 8, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    rvec, _ = cv2.Rodrigues(R_world)
    cv2.drawFrameAxes(vis, cam_matrix, dist_coeffs, rvec, t_world, settings.MARKER_SIZE_CM * 1.5, 4)
    return vis


def make_calibration_npz(role, overwrite=False):
    if role == 'top':
        K, D = settings.K_MATRIX_TOP, settings.DIST_COEFFS_TOP
        video = settings.CALIB_TOP_VIDEO_PATH or settings.TOP_VIDEO_PATH
        save_path = settings.MANUAL_CALIB_TOP_PATH
        use_wall, pick_rightmost = False, False
    elif role == 'front':
        K, D = settings.K_MATRIX_FRONT, settings.DIST_COEFFS_FRONT
        video = settings.CALIB_FRONT_VIDEO_PATH or settings.FRONT_VIDEO_PATH
        save_path = settings.MANUAL_CALIB_FRONT_PATH
        use_wall = bool(settings.CALIB_FRONT_USE_WALL_TRANSFORM)
        pick_rightmost = bool(settings.CALIB_FRONT_PICK_RIGHTMOST)
    else:
        raise ValueError("role must be 'top' or 'front'")

    if os.path.exists(save_path) and not overwrite:
        raise FileExistsError(f'{save_path} already exists. Set overwrite=True to replace it.')

    frame_index = int(settings.CALIB_FRAME_INDEX)
    frame = grab_frame(video, frame_index)
    found = detect_marker(frame, pick_rightmost=pick_rightmost)
    if found is None:
        raise RuntimeError(
            f'[{role}] No ArUco marker detected in frame {frame_index} of {video}. '
            'Set calibration.frame_index to a frame where the marker is clearly visible.')
    corners, marker_id = found

    R, t = solve_marker_pose(corners, K, D)
    R_world, t_world = marker_to_world(R, t, use_wall)

    meta = dict(
        corners_px=corners,
        order=np.array([0, 1, 2, 3], dtype=int),
        marker_size_cm=settings.MARKER_SIZE_CM,
        source_video=video,
        frame_index=frame_index,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
    )
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if role == 'top':
        np.savez(save_path, R=R, t=t.reshape(3), **meta)
    else:
        np.savez(save_path, R=R, t=t.reshape(3), use_wall_transform=use_wall, **meta)

    check_path = os.path.join(settings.OUTPUT_DIR or os.path.dirname(save_path), f'calibration_check_{role}.png')
    cv2.imwrite(check_path, draw_check_image(frame, corners, K, D, R_world, t_world))

    print(f'[{role}] marker ID {marker_id} detected -> saved {save_path}')
    print(f'[{role}] camera position in world [cm]: {(-R_world.T @ t_world).ravel().round(2)}')
    return {'npz': save_path, 'check_image': check_path, 'R_world': R_world, 't_world': t_world}
