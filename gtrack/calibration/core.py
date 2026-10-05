# -*- coding: utf-8 -*-
"""
보정 NPZ 생성 (입력 블록 "보정값 NPZ: R, t")

원본 manual_calib_top.py / manual_calib_front.py의 계산 부분을 창(cv2.imshow) 없이 쓸 수 있게 옮겼다.
클릭은 notebooks/01에서 colab_click.click_points()로 받는다.

NPZ 내용 (원본과 같은 키)
  상단: R(3x3), t(3,)  마커(=world) → 카메라
        corners_px(4,2), order, marker_size_cm, source_video, frame_index, timestamp
        (선택) measured_px, measured_world, measured_dist_from_marker_cm
  정면: R(3x3), t(3,)  클릭한 마커 → 카메라 (world 변환 전 raw)
        use_wall_transform, corners_px, order, marker_size_cm, source_video, frame_index, timestamp

마커 코너 순서는 OpenCV ArUco 자동검출 순서와 같다: 1=TL(좌상단), 2=TR(우상단), 3=BR(우하단), 4=BL(좌하단)
"""

import os
import time

import cv2
import numpy as np

from .. import settings

CORNER_LABELS = ["1:TL(좌상단)", "2:TR(우상단)", "3:BR(우하단)", "4:BL(좌하단)"]


def marker_obj_pts():
    """마커 로컬 좌표 (detect_marker_pose와 동일한 순서: TL, TR, BR, BL)."""
    _half = settings.MARKER_SIZE_CM / 2.0
    return np.array([[-_half,  _half, 0],
                     [ _half,  _half, 0],
                     [ _half, -_half, 0],
                     [-_half, -_half, 0]], dtype=np.float32)


def grab_frame(video_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"프레임을 읽을 수 없습니다: {video_path} (index={frame_index})")
    return frame


def auto_detect_marker(frame, pick_rightmost=False):
    """ArUco 자동검출. 여러 개 잡히면 pick_rightmost=True일 때 center-x가 가장 큰(오른쪽) 마커,
    False일 때 처음 검출된 마커를 고른다 (상단은 False, 정면은 CALIB_FRONT_PICK_RIGHTMOST).
    반환: (corners_px(4,2) float32, marker_id) 또는 검출 실패 시 None.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(settings.ARUCO_DICT_TYPE)
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners_list, ids, _ = detector.detectMarkers(frame)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        corners_list, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return None

    candidates = []
    for c, mid in zip(corners_list, ids.flatten()):
        pts = c.reshape(4, 2).astype(np.float32)   # TL,TR,BR,BL
        cx = pts[:, 0].mean()
        candidates.append((cx, pts, int(mid)))

    if pick_rightmost:
        candidates.sort(key=lambda x: x[0])   # center-x 오름차순
        _, pts, mid = candidates[-1]          # 가장 오른쪽
    else:
        _, pts, mid = candidates[0]
    return pts, mid


def solve_marker_pose(points, order, cam_matrix, dist_coeffs):
    """4점 + order로 solvePnP. 성공 시 (R, t(3,1), rvec), 실패 시 None."""
    if len(points) < 4:
        return None
    img_pts = np.array(points, dtype=np.float32)
    obj_pts = marker_obj_pts()[order]
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_matrix, dist_coeffs)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1), rvec


def marker_to_world(R_cm, t_cm, use_wall_transform):
    """클릭한 마커 pose(R_cm, t_cm: 마커->카메라)를 월드 pose로 변환.
       use_wall_transform=False -> 이 마커가 곧 월드 (그대로 반환)
       use_wall_transform=True  -> R_FW/t_FW 적용 (벽 마커 -> 월드)"""
    if not use_wall_transform:
        return R_cm, t_cm
    R_w = R_cm @ settings.R_FW
    t_w = R_cm @ settings.t_FW + t_cm
    return R_w, t_w


def world_ray_intersect(u, v, R, t, cam_matrix, dist_coeffs, target_z):
    """이미지 픽셀(u,v)을 solvePnP로 구한 (R,t) 기준 월드좌표(X,Y,target_z)로 변환.
    R,t: 마커(월드)->카메라 외부파라미터(t는 (3,1) 또는 (3,)).
    target_z: 교차시킬 평면의 월드 Z값(cm). 0 = 마커가 놓인 평면.
    굴절은 고려하지 않는다 (마른 상태 보정 확인용)."""
    pt = np.array([[[u, v]]], dtype=np.float32)
    undist = cv2.undistortPoints(pt, cam_matrix, dist_coeffs)  # 정규화 카메라 좌표
    x, y = undist[0, 0]
    d_cam = np.array([x, y, 1.0])
    d_cam /= np.linalg.norm(d_cam)

    R_wc = R.T                                      # 카메라->월드 회전
    d_world = R_wc @ d_cam                          # 월드좌표계에서의 광선 방향
    C_world = (-R_wc @ t.reshape(3, 1)).reshape(3)  # 카메라 원점의 월드좌표

    if abs(d_world[2]) < 1e-9:
        return None  # 광선이 평면과 거의 평행
    s = (target_z - C_world[2]) / d_world[2]
    if s <= 0:
        return None  # 광선이 평면 반대쪽을 향함(비정상)
    return C_world + s * d_world  # (X, Y, Z) cm, 마커 중심(원점) 기준


class MarkerCalibrator:
    """한 카메라(top 또는 front)의 보정 NPZ를 만드는 작업 단위.

    사용 순서 (notebooks/01 참고)
      cal = MarkerCalibrator('top')
      cal.load_frame()          # 보정용 영상의 지정 프레임 읽기
      cal.auto_detect()         # ArUco 자동인식 (실패하면 cal.set_points(클릭 결과))
      cal.preview()             # 좌표축 확인: X(빨강) 오른쪽, Y(초록) 수조 안쪽, Z(파랑) 위
      cal.rotate_order(1)       # 축 방향이 틀리면 코너 순서를 한 칸씩 돌린다 (원본 'z' 키)
      cal.save()                # NPZ 저장
    """

    def __init__(self, role):
        if role not in ('top', 'front'):
            raise ValueError("role은 'top' 또는 'front'")
        self.role = role
        if role == 'top':
            self.K, self.D = settings.K_MATRIX_TOP, settings.DIST_COEFFS_TOP
            self.video_path = settings.CALIB_TOP_VIDEO_PATH or settings.TOP_VIDEO_PATH
            self.save_path = settings.MANUAL_CALIB_TOP_PATH
            self.use_wall_transform = False
            self.pick_rightmost = False
        else:
            self.K, self.D = settings.K_MATRIX_FRONT, settings.DIST_COEFFS_FRONT
            self.video_path = settings.CALIB_FRONT_VIDEO_PATH or settings.FRONT_VIDEO_PATH
            self.save_path = settings.MANUAL_CALIB_FRONT_PATH
            self.use_wall_transform = bool(settings.CALIB_FRONT_USE_WALL_TRANSFORM)
            self.pick_rightmost = bool(settings.CALIB_FRONT_PICK_RIGHTMOST)
        self.frame_index = int(settings.CALIB_FRAME_INDEX)
        self.frame = None
        self.points = []
        self.order = [0, 1, 2, 3]
        self.source = None      # 'auto' 또는 'manual'
        self.marker_id = None
        self.measured = None    # (px, py, world)

    # ---------- 입력 ----------
    def load_frame(self):
        self.frame = grab_frame(self.video_path, self.frame_index)
        h, w = self.frame.shape[:2]
        print(f"[{self.role}] {self.video_path} | frame {self.frame_index} | {w}x{h}")
        return self.frame

    def auto_detect(self):
        result = auto_detect_marker(self.frame, pick_rightmost=self.pick_rightmost)
        if result is None:
            print(f"[{self.role}] ⚠️ 자동인식 실패 — 클릭으로 4개 코너를 지정하세요.")
            return False
        pts, mid = result
        self.points = [tuple(map(float, p)) for p in pts]
        self.order = [0, 1, 2, 3]
        self.source, self.marker_id = 'auto', mid
        extra = ' (오른쪽 마커 선택)' if self.pick_rightmost else ''
        print(f"[{self.role}] 🎯 자동인식 성공: ID={mid}{extra}")
        return True

    def set_points(self, points):
        """클릭 또는 직접 입력한 4개 코너 픽셀 [[u,v], ...] (TL, TR, BR, BL 순)."""
        points = [tuple(map(float, p)) for p in points]
        if len(points) != 4:
            raise ValueError(f'코너 4개가 필요합니다 (받은 개수: {len(points)})')
        self.points = points
        self.order = [0, 1, 2, 3]
        self.source, self.marker_id = 'manual', None
        print(f"[{self.role}] ✏️ 수동 코너 지정: {np.round(points, 1).tolist()}")

    def rotate_order(self, steps=1):
        """코너 순서를 steps번 순환 (원본 도구의 'z' 키와 같음)."""
        for _ in range(int(steps) % 4):
            self.order = self.order[1:] + self.order[:1]
        print(f"[{self.role}] 🔁 코너 순서 -> order={self.order}")

    # ---------- 계산 ----------
    def solve(self):
        """반환: (R_cm, t_cm, R_w, t_w, rvec_w). top은 R_cm=R_w."""
        solved = solve_marker_pose(self.points, self.order, self.K, self.D)
        if solved is None:
            return None
        R_cm, t_cm, _ = solved
        R_w, t_w = marker_to_world(R_cm, t_cm, self.use_wall_transform)
        rvec_w, _ = cv2.Rodrigues(R_w)
        return R_cm, t_cm, R_w, t_w, rvec_w

    def preview_image(self):
        """코너 점과 world 좌표축을 그린 BGR 이미지."""
        vis = self.frame.copy()
        for i, (px, py) in enumerate(self.points):
            cv2.circle(vis, (int(px), int(py)), 6, (0, 255, 255), -1)
            label = CORNER_LABELS[i].split(':')[0]
            cv2.putText(vis, label, (int(px) + 10, int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(vis, label, (int(px) + 10, int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if len(self.points) == 4:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 1)
        solved = self.solve()
        if solved is not None:
            _, _, _, t_w, rvec_w = solved
            axis_len = settings.MARKER_SIZE_CM * 1.5
            cv2.drawFrameAxes(vis, self.K, self.D, rvec_w, t_w, axis_len, 4)
        if self.measured is not None:
            px, py, world = self.measured
            cv2.circle(vis, (int(px), int(py)), 6, (255, 0, 0), -1)
        return vis, solved

    def preview(self, zoom_margin=150):
        """전체 화면 + 마커 주변 확대 화면을 notebook에 표시한다."""
        import matplotlib.pyplot as plt
        vis, solved = self.preview_image()
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [16, 9]})
        axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'{self.role}: full frame / source={self.source} order={self.order}')
        if self.points:
            pts = np.array(self.points)
            x0, y0 = np.maximum(pts.min(axis=0) - zoom_margin, 0).astype(int)
            x1, y1 = (pts.max(axis=0) + zoom_margin).astype(int)
            axes[1].imshow(cv2.cvtColor(vis[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            axes[1].set_title('marker zoom')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        if solved is None:
            print('⚠️ 아직 4점이 없거나 solvePnP 실패')
        else:
            print('확인: X(빨강) 오른쪽 / Y(초록) 수조 안쪽(카메라에서 멀어지는 방향) / Z(파랑) 위')
            print('      방향이 틀리면 rotate_order(1)을 실행하고 다시 preview()')
            if self.role == 'front':
                mode = '벽마커 변환(R_FW/t_FW 적용)' if self.use_wall_transform else '직접 모드(클릭한 마커=월드)'
                print(f'      정면 모드: {mode} (config.yaml calibration.front_use_wall_transform)')
        return solved

    # ---------- 저장 ----------
    def save(self, overwrite=False):
        solved = self.solve()
        if solved is None:
            raise RuntimeError('4점이 없거나 solvePnP 실패 — 저장하지 않았습니다.')
        if os.path.exists(self.save_path) and not overwrite:
            raise FileExistsError(
                f'{self.save_path} 가 이미 있습니다. 덮어쓰려면 save(overwrite=True)')
        os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
        R_cm, t_cm, R_w, t_w, _ = solved
        common = dict(
            corners_px=np.array(self.points, dtype=np.float32),
            order=np.array(self.order, dtype=int),
            marker_size_cm=settings.MARKER_SIZE_CM,
            source_video=self.video_path,
            frame_index=self.frame_index,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if self.role == 'top':
            np.savez(self.save_path, R=R_cm, t=t_cm.reshape(3), **common)
        else:
            np.savez(self.save_path,
                     R=R_cm, t=t_cm.reshape(3),               # raw 마커 pose (월드 변환 전)
                     use_wall_transform=self.use_wall_transform,
                     **common)
        print(f"✅ 저장 완료: {self.save_path}")
        if self.role == 'front':
            print(f"   use_wall_transform={self.use_wall_transform}")
            print(f"   R(raw marker)=\n{R_cm}\n   t(raw marker)={t_cm.reshape(3)}")
        print(f"   R(world)=\n{R_w}\n   t(world)={t_w.reshape(3)}")
        if self.measured is not None and self.role == 'top':
            self.save_measured_point()

    # ---------- 측정점 (상단, 선택) ----------
    def measure_point(self, px, py, target_z=0.0):
        """한 점의 마커 중심 기준 좌표/거리를 계산한다 (굴절 미적용, 마른 상태 확인용).
        target_z: 그 점이 있는 평면의 world Z(cm). 기본 0 = 마커 평면."""
        solved = self.solve()
        if solved is None:
            raise RuntimeError('먼저 4개 코너를 지정하세요.')
        R_cm, t_cm, R_w, t_w, _ = solved
        world = world_ray_intersect(px, py, R_w, t_w, self.K, self.D, target_z)
        if world is None:
            print("  ⚠️ 계산 실패 (광선이 평면과 거의 평행하거나 방향이 반대)")
            return None
        self.measured = (float(px), float(py), world)
        dist_3d = float(np.linalg.norm(world))
        dist_2d = float(np.hypot(world[0], world[1]))
        print(f"  📍 픽셀({px:.1f},{py:.1f}) -> 마커 중심 기준 X={world[0]:.2f}cm, "
              f"Y={world[1]:.2f}cm, Z={world[2]:.2f}cm")
        print(f"     -> 마커 중심으로부터 거리: 3D={dist_3d:.2f}cm, 평면(XY)={dist_2d:.2f}cm")
        return world

    def save_measured_point(self):
        """measure_point()로 구한 점을 이미 저장한 NPZ에 추가한다."""
        if self.measured is None:
            print("  ⚠️ 저장할 점이 없습니다.")
            return
        px, py, world = self.measured
        data = dict(np.load(self.save_path))
        data["measured_px"] = np.array([px, py], dtype=np.float32)
        data["measured_world"] = np.array(world, dtype=np.float32)
        data["measured_dist_from_marker_cm"] = np.float32(np.linalg.norm(world))
        np.savez(self.save_path, **data)
        print(f"  💾 측정점을 {self.save_path} 에 추가 저장 "
              f"(measured_px, measured_world, measured_dist_from_marker_cm)")
