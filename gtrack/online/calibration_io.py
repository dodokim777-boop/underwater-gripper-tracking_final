# -*- coding: utf-8 -*-
"""
입력: 보정값 (NPZ의 R, t + settings의 K, D)

- NPZ는 notebooks/01_calibration_npz.ipynb 에서 만든다 (마커 4코너 → solvePnP).
- 상단 NPZ : 마커(=world) → 카메라 R, t
- 정면 NPZ : 클릭한 마커 → 카메라 R, t, use_wall_transform
             use_wall_transform=True이면 R_FW/t_FW로 벽 마커 → world 변환을 붙인다.
- K, D는 체커보드로 측정한 고정값이며 config.yaml의 cameras 항목에서 관리한다.

calib dict 키: P, rvec, tvec, cam_matrix, dist_coeffs, motor_origin_px (+ metadata)
"""

import cv2
import numpy as np

from .. import settings
from ..io_utils import _npz_value


# ArUco 검출기 (OpenCV 신·구 API 모두 지원)
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(settings.ARUCO_DICT_TYPE)
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    _aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    def detect_aruco(gray): return _aruco_detector.detectMarkers(gray)
except AttributeError:
    ARUCO_DICT = cv2.aruco.Dictionary_get(settings.ARUCO_DICT_TYPE)
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
    def detect_aruco(gray): return cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)


def detect_marker_pose(frame, cam_matrix, dist_coeffs, select='first'):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect_aruco(gray)
    if ids is None or len(corners) == 0:
        return None
    cx = [c[0][:, 0].mean() for c in corners]
    if select == 'right': k = int(np.argmax(cx))
    elif select == 'left': k = int(np.argmin(cx))
    else: k = 0
    c = corners[k][0].astype(np.float32)

    half = settings.MARKER_SIZE_CM / 2.0
    obj_pts = np.array([[-half, half, 0], [half, half, 0],
                        [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, c, cam_matrix, dist_coeffs,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, c, cam_matrix, dist_coeffs)
    if not ok: return None
    R_mat, _ = cv2.Rodrigues(rvec)
    return R_mat, tvec.reshape(3, 1)


def build_world_calib(
    R_cm, t_cm, cam_matrix, dist_coeffs, marker_to_world=None, metadata=None
):
    R_cm = np.asarray(R_cm, dtype=float).reshape(3, 3)
    t_cm = np.asarray(t_cm, dtype=float).reshape(3, 1)
    cam_matrix = np.asarray(cam_matrix, dtype=float).reshape(3, 3)
    dist_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(1, -1)
    if marker_to_world is None:
        R_w, t_w = R_cm, t_cm
    else:
        R_fw, t_fw = marker_to_world
        R_w = R_cm @ R_fw
        t_w = R_cm @ t_fw + t_cm
    rvec, _ = cv2.Rodrigues(R_w)
    tvec = t_w.reshape(3, 1)
    P = cam_matrix @ np.hstack((R_w, tvec))

    motor_3d = np.array([settings.MOTOR_CENTER_IN_ID0_CM], dtype=np.float32)
    motor_px, _ = cv2.projectPoints(motor_3d, rvec, tvec, cam_matrix, dist_coeffs)
    calib = {
        'P': P,
        'rvec': rvec,
        'tvec': tvec,
        'cam_matrix': cam_matrix,
        'dist_coeffs': dist_coeffs,
        'motor_origin_px': motor_px.reshape(-1, 2)[0],
    }
    calib.update(metadata or {})
    return calib


def verify_unification(calib_front, frame_front):
    half = settings.MARKER_SIZE_CM / 2.0
    corners_w = np.array([[-half, half, 0], [half, half, 0],
                          [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    px, _ = cv2.projectPoints(corners_w, calib_front['rvec'], calib_front['tvec'],
                              calib_front['cam_matrix'], calib_front['dist_coeffs'])
    vis = frame_front.copy()
    for p in px.reshape(-1, 2).astype(int):
        cv2.circle(vis, tuple(p), 8, (0, 0, 255), -1)
    if settings.HEADLESS:
        try:
            cv2.imwrite(settings.VERIFY_IMAGE_PATH, vis)
            print(f"   └─ (headless) 검증 이미지 저장: {settings.VERIFY_IMAGE_PATH}")
        except Exception as e:
            print(f"   └─ (headless) 검증 이미지 저장 실패: {e}")
        return
    cv2.imshow("VERIFY", cv2.resize(vis, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(5000); cv2.destroyWindow("VERIFY")


def _load_top_calib(path):
    """상단 extrinsic NPZ에서 R,t를 읽고 하드코딩된 top intrinsic을 붙인다."""
    data = np.load(path, allow_pickle=False)
    if 'R' not in data.files or 't' not in data.files:
        raise KeyError(f"{path}: R과 t가 필요합니다.")
    return build_world_calib(
        data['R'], np.asarray(data['t']).reshape(3, 1),
        settings.K_MATRIX_TOP, settings.DIST_COEFFS_TOP,
        metadata={
            'camera_role': 'top',
            'camera_model': settings.TOP_CAMERA_MODEL,
            'intrinsic_source': 'hardcoded',
        },
    )


def _load_front_calib(path):
    """정면 extrinsic NPZ에서 R,t(+use_wall_transform)를 읽고 하드코딩된 front intrinsic을 붙인다."""
    data = np.load(path, allow_pickle=False)
    if 'R' not in data.files or 't' not in data.files:
        raise KeyError(f"{path}: R과 t가 필요합니다.")
    use_wall = bool(_npz_value(data, ('use_wall_transform',), False))
    marker_to_world = (settings.R_FW, settings.t_FW) if use_wall else None
    calib = build_world_calib(
        data['R'], np.asarray(data['t']).reshape(3, 1),
        settings.K_MATRIX_FRONT, settings.DIST_COEFFS_FRONT,
        marker_to_world=marker_to_world,
        metadata={
            'camera_role': 'front',
            'camera_model': settings.FRONT_CAMERA_MODEL,
            'intrinsic_source': 'hardcoded',
        },
    )
    return calib, use_wall


def _calib_summary(name, calib):
    R_w, _ = cv2.Rodrigues(calib['rvec'])
    t_w = calib['tvec'].reshape(3)
    C = -R_w.T @ t_w
    euler_deg = cv2.RQDecomp3x3(R_w)[0]
    print(
        f"   {name}: model={calib.get('camera_model', 'unknown')}, "
        f"mode={calib.get('projection_mode', 'unknown')}, "
        f"C_ID0=({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}) cm, "
        f"Euler≈({euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}) deg"
    )


def load_calibrations(dataset_is_wet):
    """상단·정면 NPZ를 읽어 calib dict를 만든다.

    wet 데이터는 사전 dry-calibration NPZ 없이는 실행하지 않는다.
    반환: (calib_top, calib_front, front_use_wall_transform)
    """
    calib_top = None
    calib_front = None
    front_use_wall_transform = None
    calib_errors = []
    try:
        calib_top = _load_top_calib(settings.MANUAL_CALIB_TOP_PATH)
        print(f"✅ 상단 NPZ 로드: {settings.MANUAL_CALIB_TOP_PATH}")
    except Exception as exc:
        calib_errors.append(f"top: {exc}")

    try:
        calib_front, front_use_wall_transform = _load_front_calib(settings.MANUAL_CALIB_FRONT_PATH)
        print(
            f"✅ 정면 NPZ 로드: {settings.MANUAL_CALIB_FRONT_PATH} | "
            f"use_wall_transform={front_use_wall_transform}"
        )
    except Exception as exc:
        calib_errors.append(f"front: {exc}")

    # 핵심 수정: wet에서는 사전 dry-calibration NPZ 없이는 실행하지 않는다.
    if dataset_is_wet and (calib_top is None or calib_front is None):
        raise RuntimeError(
            "Wet mode requires valid dry-calibration NPZ files. "
            + " | ".join(calib_errors)
        )

    if calib_top is not None:
        _calib_summary('top', calib_top)
    if calib_front is not None:
        _calib_summary('front', calib_front)
    return calib_top, calib_front, front_use_wall_transform
