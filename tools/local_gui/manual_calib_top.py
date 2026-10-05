# -*- coding: utf-8 -*-
"""
[로컬 PC 전용] 상단 카메라 수동 클릭 캘리브레이션 도구
원본 manual_calib_top.py과 동작이 같고, 경로·K·D 등 설정값만 config.yaml에서 읽는다.

Colab에서는 창을 띄울 수 없으므로 notebooks/01_calibration_npz.ipynb를 사용한다.
이 파일은 로컬 PC(창 표시 가능)에서 같은 NPZ를 만들 때만 쓴다.

실행 (저장소 최상위 폴더에서)
  pip install -r requirements.txt
  python tools/local_gui/manual_calib_top.py path/to/config.yaml

config.yaml에서 읽는 값: calibration.top_video(비어 있으면 inputs.top_video), calibration.frame_index,
  inputs.calib_top_npz(저장 경로), cameras.top.K/dist, setup.marker_size_cm

----- 원본 설명 (경로·상수는 위 config.yaml 항목으로 대체됨) -----
🖱️ 상단 카메라 수동 클릭 캘리브레이션 도구
====================================================
분석영상(top8.mp4)을 추가로 촬영할 수 없을 때, 이미 가지고 있는
분석영상의 첫 프레임에서 마커 4개 코너를 직접 클릭해서
월드 원점(calib_top)을 임시로 잡기 위한 도구입니다.

사용법:
  1) 아래 TOP_VIDEO_PATH 등 경로가 본 스크립트와 같은지 확인
  2) python manual_calib_top.py 실행
  3) 뜨는 창에서 마커의 4개 코너를 순서대로 클릭
       1번 = 마커 좌상단(TL)
       2번 = 마커 우상단(TR)
       3번 = 마커 우하단(BR)
       4번 = 마커 좌하단(BL)
     (OpenCV ArUco 자동검출이 리턴하는 코너 순서와 동일하게 맞춘 것 -
      기존 detect_marker_pose()의 obj_pts 순서와 호환됩니다)
  4) 4점을 찍으면 화면에 좌표축(X=빨강, Y=초록, Z=파랑)이 바로 투영됩니다.
       - X(빨강)이 오른쪽, Y(초록)이 수조 안쪽(뒤), Z(파랑)이 위를 향해야 정상입니다.
       - 방향이 이상하면 다시 클릭하지 말고 'z' 키를 눌러 코너 순서를 순환시켜보세요.
  5) 방향이 맞으면 ENTER(또는 's')로 저장. 'u'=마지막 점 취소, 'c'=전체 초기화, 'q'=중단
  6) 저장 후 이어서 측정 모드가 열립니다. 마커 중심으로부터 거리를 알고 싶은 점을
     딱 1번 클릭하면 마커 중심(월드 원점) 기준 X,Y,Z와 3D 거리가 콘솔에 출력되고
     화면에 파란 점으로 표시됩니다. 기본은 마커가 놓인 평면(Z=0) 기준이며,
     그 점이 다른 높이에 있다면 '+'/'-'로 기준 평면 Z를 맞춘 뒤 다시 찍으세요.
     'q'로 종료하면 그 점이 npz에 자동 저장됩니다.

저장 결과: manual_calib_top.npz
  - R (3x3), t (3,)                  : 마커(=월드) -> 카메라 외부파라미터
  - corners_px (4,2)                 : 클릭한 마커 코너 픽셀 좌표(감사/재현용)
  - measured_px (2,)                 : 측정 모드에서 클릭한 픽셀 좌표
  - measured_world (3,)              : 그 픽셀의 마커 중심 기준 월드좌표(X,Y,Z cm)
  - measured_dist_from_marker_cm     : 마커 중심으로부터의 3D 거리(cm)
본 스크립트가 만든 npz는 메인 스크립트(triaruco...)가 있으면 자동으로 읽어
상단 카메라 캘리브레이션을 대체합니다.
"""

import cv2
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gtrack import settings as _settings
_settings.configure(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml', make_output_dir=False)
import json
import time

# ==========================================
# 🎯 자동 인식 설정
# ==========================================
ARUCO_DICT_TYPE = _settings.ARUCO_DICT_TYPE
AUTO_DETECT = _settings.CALIB_AUTO_DETECT
# top은 마커가 하나뿐이라 여러개 중 고르는 로직 불필요 (front의 PICK_RIGHTMOST 해당없음)

# ==========================================
# 🌟 메인 스크립트와 반드시 동일하게 맞춰야 하는 값들
# ==========================================
MARKER_SIZE_CM = _settings.MARKER_SIZE_CM

K_MATRIX_TOP = _settings.K_MATRIX_TOP


DIST_COEFFS_TOP = _settings.DIST_COEFFS_TOP



# 클릭 기준 프레임: 상단 분석영상(top8.mp4) 첫 프레임
TOP_VIDEO_PATH = _settings.CALIB_TOP_VIDEO_PATH or _settings.TOP_VIDEO_PATH
FRAME_INDEX = _settings.CALIB_FRAME_INDEX

SAVE_PATH = _settings.MANUAL_CALIB_TOP_PATH

# 마커 로컬 좌표 (기존 detect_marker_pose와 동일한 순서: TL, TR, BR, BL)
_half = MARKER_SIZE_CM / 2.0
OBJ_PTS = np.array([[-_half,  _half, 0],
                     [ _half,  _half, 0],
                     [ _half, -_half, 0],
                     [-_half, -_half, 0]], dtype=np.float32)

CORNER_LABELS = ["1:TL(\uc88c\uc0c1\ub2e8)", "2:TR(\uc6b0\uc0c1\ub2e8)", "3:BR(\uc6b0\ud558\ub2e8)", "4:BL(\uc88c\ud558\ub2e8)"]


def auto_detect_marker(frame):
    """ArUco 자동검출. top은 마커가 하나뿐이므로 처음 검출된 것을 그대로 사용.
       반환: (corners_px(4,2) float32, marker_id) 또는 검출 실패 시 None.
       코너 순서는 OpenCV ArUco 기본순서 TL,TR,BR,BL 이므로 OBJ_PTS와 그대로 호환됨."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners_list, ids, _ = detector.detectMarkers(frame)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        corners_list, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return None

    pts = corners_list[0].reshape(4, 2).astype(np.float32)   # TL,TR,BR,BL
    mid = int(ids.flatten()[0])
    return pts, mid

# ==========================================
# 🔍 확대경(magnifier) 설정
# ==========================================
MAG_SIZE   = 220   # 확대경 표시 크기(px, 정사각형)
MAG_ZOOM   = 6      # 확대 배율
MAG_SRC    = MAG_SIZE // MAG_ZOOM  # 원본에서 잘라올 영역 크기

# ==========================================
# 🖥️ 창 표시 크기 설정 (원본 프레임이 화면보다 클 때 축소 표시)
# ==========================================
DISPLAY_MAX_W = 1280   # 창에 보일 최대 가로(px)
DISPLAY_MAX_H = 800    # 창에 보일 최대 세로(px)
# 클릭 좌표는 항상 원본 해상도 기준으로 자동 환산되므로 정밀도 손실 없음.
# (정밀 클릭이 더 필요하면 확대경을 활용하세요)


def _world_ray_intersect(u, v, R, t, cam_matrix, dist_coeffs, target_z):
    """이미지 픽셀(u,v)을 solvePnP로 구한 (R,t) 기준 월드좌표(X,Y,target_z)로 변환.
    R,t: 마커(월드)->카메라 외부파라미터(캘리브레이션 결과 그대로, t는 (3,1) 또는 (3,)).
    target_z: 교차시킬 평면의 월드 Z값(cm). 기본 0 = 마커가 놓인 평면(=마커 중심 높이)."""
    pt = np.array([[[u, v]]], dtype=np.float32)
    undist = cv2.undistortPoints(pt, cam_matrix, dist_coeffs)  # 정규화 카메라 좌표
    x, y = undist[0, 0]
    d_cam = np.array([x, y, 1.0])
    d_cam /= np.linalg.norm(d_cam)

    R_wc = R.T                                      # 카메라->월드 회전
    d_world = R_wc @ d_cam                          # 월드좌표계에서의 광선 방향
    C_world = (-R_wc @ t.reshape(3, 1)).reshape(3)   # 카메라 원점의 월드좌표

    if abs(d_world[2]) < 1e-9:
        return None  # 광선이 평면과 거의 평행
    s = (target_z - C_world[2]) / d_world[2]
    if s <= 0:
        return None  # 광선이 평면 반대쪽을 향함(비정상)
    return C_world + s * d_world  # (X, Y, Z) cm, 마커 중심(원점) 기준


class ManualCornerPicker:
    def __init__(self, frame, cam_matrix, dist_coeffs, auto_points=None, auto_id=None):
        self.base = frame.copy()
        self.H, self.W = frame.shape[:2]
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.mouse_pos = (0, 0)   # 항상 원본 해상도 기준으로 저장
        self.order = [0, 1, 2, 3]  # OBJ_PTS 인덱스 순환용
        self.window = "Top Calibration (auto-detect or click 4 marker corners)"

        if auto_points is not None:
            self.points = [tuple(p) for p in auto_points]
            self.is_auto = True
            self.auto_id = auto_id
        else:
            self.points = []
            self.is_auto = False
            self.auto_id = None

        # 원본이 화면보다 크면 표시용 축소 비율 계산 (클릭 좌표는 항상 원본 기준으로 환산)
        self.scale = min(1.0, DISPLAY_MAX_W / self.W, DISPLAY_MAX_H / self.H)
        print(f"  🖥️ 원본 해상도 {self.W}x{self.H} -> 표시 배율 {self.scale:.3f}배로 축소 표시")
        if self.is_auto:
            print(f"  🎯 자동인식 성공 (ID={self.auto_id}). 's'로 그대로 저장하거나, 'm'으로 직접 재클릭하세요.")
        else:
            print("  ⚠️ 자동인식 실패 또는 비활성 - 수동 클릭으로 진행하세요.")

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        # 콜백으로 들어오는 x,y는 '표시된(축소된) 창' 기준 -> 원본 해상도로 환산
        ox, oy = x / self.scale, y / self.scale
        self.mouse_pos = (ox, oy)
        if self.is_auto:
            return  # 자동인식 결과 표시 중엔 클릭 무시. 'm'을 눌러야 수동모드로 전환됨.
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((ox, oy))
            print(f"  📍 {CORNER_LABELS[len(self.points)-1]} 클릭(원본좌표): ({ox:.1f}, {oy:.1f})")

    def _solve(self):
        """현재 4점 + 현재 order로 solvePnP. 성공 시 (R, t) 반환, 실패 시 None."""
        if len(self.points) < 4:
            return None
        img_pts = np.array(self.points, dtype=np.float32)
        obj_pts = OBJ_PTS[self.order]
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.cam_matrix, self.dist_coeffs,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.cam_matrix, self.dist_coeffs)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3, 1), rvec

    def _draw(self):
        vis = self.base.copy()

        # 이미 찍은 점 + 라벨
        for i, (px, py) in enumerate(self.points):
            cv2.circle(vis, (int(px), int(py)), 6, (0, 255, 255), -1)
            cv2.putText(vis, CORNER_LABELS[i].split(':')[0], (int(px)+10, int(py)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vis, CORNER_LABELS[i].split(':')[0], (int(px)+10, int(py)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(vis, tuple(map(int, self.points[i])), tuple(map(int, self.points[i+1])), (0, 255, 255), 1)
        if len(self.points) == 4:
            cv2.line(vis, tuple(map(int, self.points[3])), tuple(map(int, self.points[0])), (0, 255, 255), 1)

        # 4점 다 찍혔으면 좌표축 오버레이로 방향 확인
        solved = self._solve()
        status_txt, status_color = "", (255, 255, 255)
        if solved is not None:
            R, t, rvec = solved
            axis_len = MARKER_SIZE_CM * 1.5
            try:
                cv2.drawFrameAxes(vis, self.cam_matrix, self.dist_coeffs, rvec, t, axis_len, 4)
            except AttributeError:
                cv2.aruco.drawAxis(vis, self.cam_matrix, self.dist_coeffs, rvec, t, axis_len)
            status_txt = "4\uc810 \uc644\ub8cc - \ucd95 \ubc29\ud5a5 \ud655\uc778: X(\ube68\uac04)\uc6b0\uce35 / Y(\ucd08\ub85d)\uc548\ucabd / Z(\ud30c\ub791)\uc704.  's' \uc800\uc7a5"
            status_txt += " / 'm' \uc218\ub3d9\uc7ac\ud074\ub9ad" if self.is_auto else " / 'z' \uc21c\uc11c\ubcc0\uacbd"
            src_txt = f" [\uc790\ub3d9\uc778\uc2dd ID={self.auto_id}]" if self.is_auto else " [\uc218\ub3d9\ud074\ub9ad]"
            status_txt += src_txt
            status_color = (0, 220, 0)
        else:
            status_txt = f"{len(self.points)}/4\uc810 \ud074\ub9ad\ud558\uc138\uc694 (\ub2e4\uc74c: {CORNER_LABELS[len(self.points)] if len(self.points)<4 else ''})"
            status_color = (0, 200, 255)

        cv2.rectangle(vis, (0, 0), (self.W, 46), (0, 0, 0), -1)
        cv2.putText(vis, status_txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        # 하단 안내
        help_txt = "m=manual re-click | click=corner select | u=undo | c=clear | z=rotate order | s/ENTER=save | q=abort"
        cv2.putText(vis, help_txt, (12, self.H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return vis, solved

    def _magnifier_patch(self):
        """마우스 주변 원본 해상도 확대 패치 (MAG_SIZE x MAG_SIZE, BGR)."""
        mx, my = self.mouse_pos
        x0 = int(np.clip(mx - MAG_SRC // 2, 0, self.W - MAG_SRC))
        y0 = int(np.clip(my - MAG_SRC // 2, 0, self.H - MAG_SRC))
        patch = self.base[y0:y0+MAG_SRC, x0:x0+MAG_SRC]
        if patch.size == 0:
            return None
        mag = cv2.resize(patch, (MAG_SIZE, MAG_SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.line(mag, (MAG_SIZE//2, 0), (MAG_SIZE//2, MAG_SIZE), (0, 255, 0), 1)
        cv2.line(mag, (0, MAG_SIZE//2), (MAG_SIZE, MAG_SIZE//2), (0, 255, 0), 1)
        cv2.rectangle(mag, (0, 0), (MAG_SIZE-1, MAG_SIZE-1), (255, 255, 255), 2)
        return mag

    def run(self):
        print("🖱️  4개 코너를 순서대로 클릭하세요: TL -> TR -> BR -> BL")
        disp_w, disp_h = int(self.W * self.scale), int(self.H * self.scale)
        while True:
            vis, solved = self._draw()

            # 표시용으로 축소 (클릭좌표 계산에는 영향 없음, _on_mouse에서 별도 환산)
            vis_disp = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA) if self.scale < 1.0 else vis

            # 확대경은 축소된 화면 위에 '고정 크기'로 올려서 선명하게 유지
            mag = self._magnifier_patch()
            if mag is not None:
                ox, oy = disp_w - MAG_SIZE - 10, 56
                if oy + MAG_SIZE <= disp_h and ox >= 0:
                    vis_disp[oy:oy+MAG_SIZE, ox:ox+MAG_SIZE] = mag
                    cv2.putText(vis_disp, f"zoom x{MAG_ZOOM}", (ox, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(self.window, vis_disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('m'):
                self.points = []
                self.order = [0, 1, 2, 3]
                self.is_auto = False
                print("  ✏️  수동클릭 모드로 전환 - 4개 코너를 순서대로 클릭하세요 (TL->TR->BR->BL)")
            elif key == ord('u') and self.points and not self.is_auto:
                removed = self.points.pop()
                print(f"  ↩️  마지막 점 취소: {removed}")
            elif key == ord('c') and not self.is_auto:
                self.points = []
                self.order = [0, 1, 2, 3]
                print("  🧹 전체 초기화")
            elif key == ord('z') and len(self.points) == 4 and not self.is_auto:
                self.order = self.order[1:] + self.order[:1]
                print(f"  🔁 코너 순서 회전 -> order={self.order}")
            elif key in (ord('s'), 13):  # 's' or ENTER
                if solved is None:
                    print("  ⚠️ 아직 4점을 다 안 찍었거나 solvePnP 실패")
                    continue
                cv2.destroyWindow(self.window)
                return solved
            elif key == ord('q'):
                cv2.destroyWindow(self.window)
                return None


class SinglePointMeasurer:
    """캘리브레이션 완료 후, 딱 한 점만 클릭해서 마커 중심 기준 좌표/거리를 계산."""
    def __init__(self, frame, cam_matrix, dist_coeffs, R, t, scale):
        self.base = frame.copy()
        self.H, self.W = frame.shape[:2]
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.R = R
        self.t = t
        self.scale = scale
        self.target_z = 0.0  # 기본: 마커 평면(Z=0). 다른 높이의 점이면 +/-로 조정
        self.point = None    # (px, py, world_XYZ)
        self.window = "Point Measurement (click ONE point, +/-=change Z plane, c=reset, q=quit&save)"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = x / self.scale, y / self.scale
            world = _world_ray_intersect(ox, oy, self.R, self.t,
                                          self.cam_matrix, self.dist_coeffs, self.target_z)
            if world is None:
                print("  ⚠️ 계산 실패 (광선이 평면과 거의 평행하거나 방향이 반대)")
                return
            self.point = (ox, oy, world)
            dist_3d = float(np.linalg.norm(world))          # 마커 중심(원점)까지 3D 거리
            dist_2d = float(np.hypot(world[0], world[1]))    # 같은 평면 기준 XY 평면거리
            print(f"  📍 픽셀({ox:.1f},{oy:.1f}) -> 마커 중심 기준 X={world[0]:.2f}cm, "
                  f"Y={world[1]:.2f}cm, Z={world[2]:.2f}cm")
            print(f"     -> 마커 중심으로부터 거리: 3D={dist_3d:.2f}cm, 평면(XY)={dist_2d:.2f}cm")

    def run(self):
        print("🖱️  마커 중심 기준 좌표를 알고 싶은 점을 딱 1번 클릭하세요.")
        print("     '+'/'-' : 그 점이 있는 평면의 Z 높이 변경 (기본 0=마커 평면) | 'c' 다시 찍기 | 'q' 종료(자동 저장)")
        disp_w, disp_h = int(self.W * self.scale), int(self.H * self.scale)
        while True:
            vis = self.base.copy()
            if self.point is not None:
                px, py, world = self.point
                cv2.circle(vis, (int(px), int(py)), 6, (255, 0, 0), -1)   # 파란 점 (BGR)
                cv2.circle(vis, (int(px), int(py)), 8, (255, 255, 255), 1)
                label = f"({world[0]:.1f},{world[1]:.1f},{world[2]:.1f}) d={np.linalg.norm(world):.1f}cm"
                cv2.putText(vis, label, (int(px)+10, int(py)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(vis, label, (int(px)+10, int(py)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 120), 1)
            cv2.rectangle(vis, (0, 0), (self.W, 40), (0, 0, 0), -1)
            cv2.putText(vis, f"target Z = {self.target_z:.1f} cm", (12, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            vis_disp = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA) if self.scale < 1.0 else vis
            cv2.imshow(self.window, vis_disp)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.point = None
                print("  🧹 다시 찍으세요")
            elif key in (ord('+'), ord('=')):
                self.target_z += 1.0
                print(f"  ↕️ target Z = {self.target_z:.1f} cm")
            elif key == ord('-'):
                self.target_z -= 1.0
                print(f"  ↕️ target Z = {self.target_z:.1f} cm")
        cv2.destroyWindow(self.window)

    def save_point(self, path):
        """찍은 점(픽셀+월드좌표+거리)을 기존 npz(R,t,corners_px 등)에 추가 저장."""
        if self.point is None:
            print("  ⚠️ 저장할 점이 없습니다.")
            return
        px, py, world = self.point
        data = dict(np.load(path))
        data["measured_px"] = np.array([px, py], dtype=np.float32)
        data["measured_world"] = np.array(world, dtype=np.float32)
        data["measured_dist_from_marker_cm"] = np.float32(np.linalg.norm(world))
        np.savez(path, **data)
        print(f"  💾 측정점을 {path} 에 추가 저장 완료 "
              f"(measured_px, measured_world, measured_dist_from_marker_cm)")


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


def main():
    frame = grab_frame(TOP_VIDEO_PATH, FRAME_INDEX)

    auto_points, auto_id = None, None
    if AUTO_DETECT:
        result = auto_detect_marker(frame)
        if result is not None:
            auto_points, auto_id = result
            print(f"  🎯 자동인식 성공: ID={auto_id}")
        else:
            print("  ⚠️ 자동인식 실패 - 수동 클릭으로 시작합니다.")

    picker = ManualCornerPicker(frame, K_MATRIX_TOP, DIST_COEFFS_TOP,
                                 auto_points=auto_points, auto_id=auto_id)
    solved = picker.run()
    cv2.destroyAllWindows()

    if solved is None:
        print("❌ 저장하지 않고 종료했습니다.")
        return

    R, t, rvec = solved
    np.savez(SAVE_PATH,
             R=R, t=t.reshape(3),
             corners_px=np.array(picker.points, dtype=np.float32),
             order=np.array(picker.order, dtype=int),
             marker_size_cm=MARKER_SIZE_CM,
             source_video=TOP_VIDEO_PATH,
             frame_index=FRAME_INDEX,
             timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\n✅ 저장 완료: {SAVE_PATH}")
    print(f"   R=\n{R}")
    print(f"   t={t.reshape(3)}")
    print("\n👉 메인 스크립트를 다시 실행하면 이 파일을 자동으로 읽어 상단 카메라 캘리브레이션에 사용합니다.")

    # --- 추가: 점 하나 클릭 -> 마커 중심 기준 좌표/거리 확인 + npz 저장 ---
    print("\n🖱️  마커 중심으로부터의 거리를 알고 싶은 점을 클릭하세요.")
    measurer = SinglePointMeasurer(frame, K_MATRIX_TOP, DIST_COEFFS_TOP, R, t, picker.scale)
    measurer.run()
    measurer.save_point(SAVE_PATH)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()