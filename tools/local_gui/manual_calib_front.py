# -*- coding: utf-8 -*-
"""
[로컬 PC 전용] 정면 카메라 수동 클릭 캘리브레이션 도구
원본 manual_calib_front.py과 동작이 같고, 경로·K·D 등 설정값만 config.yaml에서 읽는다.

Colab에서는 창을 띄울 수 없으므로 notebooks/01_calibration_npz.ipynb를 사용한다.
이 파일은 로컬 PC(창 표시 가능)에서 같은 NPZ를 만들 때만 쓴다.

실행 (저장소 최상위 폴더에서)
  pip install -r requirements.txt
  python tools/local_gui/manual_calib_front.py path/to/config.yaml

config.yaml에서 읽는 값: calibration.front_video(비어 있으면 inputs.front_video), calibration.frame_index,
  inputs.calib_front_npz(저장 경로), cameras.front.K/dist, setup.marker_size_cm,
  calibration.front_use_wall_transform / front_pick_rightmost, setup.front_wall_marker_R/t

----- 원본 설명 (경로·상수는 위 config.yaml 항목으로 대체됨) -----
🖱️ 정면 카메라 수동 클릭 캘리브레이션 도구
====================================================
manual_calib_top.py의 정면 카메라 버전. 분석영상(front8.mp4)을 추가로
촬영할 수 없을 때, 이미 가지고 있는 분석영상의 첫 프레임에서 마커 4개
코너를 직접 클릭해서 정면 카메라 캘리브레이션(calib_front)을 임시로
잡기 위한 도구입니다.

★ 정면은 상단과 달리 '어떤 마커를 클릭하느냐'에 따라 처리 방식이 다릅니다.
  아래 USE_WALL_TRANSFORM 스위치로 선택하세요.

  USE_WALL_TRANSFORM = False (기본값, 현재 메인 스크립트와 동일한 동작)
    → 클릭한 마커가 곧 월드 원점입니다. (메인 스크립트의 select='right'로
      고르던 것과 '같은' 마커, 즉 수조 내부 바닥 마커를 정면 각도에서
      클릭하는 경우)

  USE_WALL_TRANSFORM = True
    → 클릭한 마커가 수조 밖 벽 마커인 경우. R_FW/t_FW(마커→월드 변환)를
      적용해서 월드 좌표로 변환합니다. (물속 굴절로 내부 마커가 잘 안
      보일 때, 수조 밖에 붙인 별도 마커를 쓰는 경우)

사용법:
  1) 아래 FRONT_VIDEO_PATH 등 경로가 실제와 같은지 확인
  2) USE_WALL_TRANSFORM 을 클릭할 마커에 맞게 설정
  3) python manual_calib_front.py 실행
  4) 뜨는 창에서 마커의 4개 코너를 순서대로 클릭
       1번 = 마커 좌상단(TL), 2번 = 우상단(TR), 3번 = 우하단(BR), 4번 = 좌하단(BL)
     (OpenCV ArUco 자동검출 코너 순서와 동일 - detect_marker_pose()의 obj_pts 순서와 호환)
  5) 4점을 찍으면 화면에 '월드' 좌표축(X=빨강, Y=초록, Z=파랑)이 투영됩니다.
       - X(빨강)이 오른쪽, Y(초록)이 수조 안쪽(카메라에서 멀어지는 방향), Z(파랑)이 위.
       - 방향이 이상하면 재클릭 없이 'z' 키로 코너 순서를 순환시켜 보세요.
       - USE_WALL_TRANSFORM 설정 자체가 틀렸을 수도 있습니다 (마커를 잘못 골랐을 때).
  6) 방향이 맞으면 ENTER(또는 's')로 저장. 'u'=마지막 점 취소, 'c'=전체 초기화, 'q'=중단

저장 결과: manual_calib_front.npz
  - R (3x3), t (3,) : 클릭한 '마커' 기준 -> 카메라 외부파라미터 (월드 변환 전, raw)
  - use_wall_transform (bool) : 로드할 때 R_FW/t_FW를 적용할지 여부
  - corners_px (4,2) : 클릭한 픽셀 좌표(감사/재현용)
본 스크립트가 만든 npz는 메인 스크립트(triaruco...)가 있으면 자동으로 읽어
정면 카메라 캘리브레이션을 대체합니다.
"""

import cv2
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gtrack import settings as _settings
_settings.configure(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml', make_output_dir=False)
import time

# ==========================================
# 🎯 자동 인식 설정
# ==========================================
ARUCO_DICT_TYPE = _settings.ARUCO_DICT_TYPE
AUTO_DETECT = _settings.CALIB_AUTO_DETECT
# ★ 같은 ID 마커가 화면에 두 개 잡힐 때, 오른쪽(픽셀 center-x가 더 큰 쪽) 마커만 사용
PICK_RIGHTMOST = _settings.CALIB_FRONT_PICK_RIGHTMOST

# ==========================================
# 🌟 메인 스크립트와 반드시 동일하게 맞춰야 하는 값들
# ==========================================
MARKER_SIZE_CM = _settings.MARKER_SIZE_CM

K_MATRIX_FRONT = _settings.K_MATRIX_FRONT


DIST_COEFFS_FRONT = _settings.DIST_COEFFS_FRONT



# 클릭 기준 프레임: 정면 분석영상(front8.mp4) 첫 프레임
FRONT_VIDEO_PATH = _settings.CALIB_FRONT_VIDEO_PATH or _settings.FRONT_VIDEO_PATH
FRAME_INDEX = _settings.CALIB_FRAME_INDEX

SAVE_PATH = _settings.MANUAL_CALIB_FRONT_PATH

# ★ 클릭할 마커 종류에 맞게 선택 (위 설명 참고)
USE_WALL_TRANSFORM = _settings.CALIB_FRONT_USE_WALL_TRANSFORM

# 메인 스크립트의 R_FW/t_FW 값과 반드시 동일하게 유지
R_FW = _settings.R_FW
t_FW = _settings.t_FW

# 마커 로컬 좌표 (기존 detect_marker_pose와 동일한 순서: TL, TR, BR, BL)
_half = MARKER_SIZE_CM / 2.0
OBJ_PTS = np.array([[-_half,  _half, 0],
                     [ _half,  _half, 0],
                     [ _half, -_half, 0],
                     [-_half, -_half, 0]], dtype=np.float32)

CORNER_LABELS = ["1:TL(\uc88c\uc0c1\ub2e8)", "2:TR(\uc6b0\uc0c1\ub2e8)", "3:BR(\uc6b0\ud558\ub2e8)", "4:BL(\uc88c\ud558\ub2e8)"]


def auto_detect_marker(frame):
    """ArUco 자동검출. 여러 개 잡히면(같은 ID 두 개 등) PICK_RIGHTMOST에 따라
       center-x가 가장 큰(오른쪽) 마커 하나만 골라 반환한다.
       반환: (corners_px(4,2) float32, marker_id) 또는 검출 실패 시 None.
       코너 순서는 OpenCV ArUco 기본순서 TL,TR,BR,BL 이므로 OBJ_PTS와 그대로 호환됨."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    try:
        # 최신 OpenCV API
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners_list, ids, _ = detector.detectMarkers(frame)
    except AttributeError:
        # 구버전 OpenCV API
        params = cv2.aruco.DetectorParameters_create()
        corners_list, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return None

    candidates = []
    for c, mid in zip(corners_list, ids.flatten()):
        pts = c.reshape(4, 2).astype(np.float32)   # TL,TR,BR,BL
        cx = pts[:, 0].mean()
        candidates.append((cx, pts, int(mid)))

    if PICK_RIGHTMOST:
        candidates.sort(key=lambda x: x[0])   # center-x 오름차순
        _, pts, mid = candidates[-1]          # 가장 오른쪽
    else:
        _, pts, mid = candidates[0]

    return pts, mid

# ==========================================
# 🔍 확대경(magnifier) 설정
# ==========================================
MAG_SIZE   = 220
MAG_ZOOM   = 6
MAG_SRC    = MAG_SIZE // MAG_ZOOM

# ==========================================
# 🖥️ 창 표시 크기 설정
# ==========================================
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 800


def marker_to_world(R_cm, t_cm):
    """클릭한 마커 pose(R_cm, t_cm: 마커->카메라)를 월드 pose로 변환.
       USE_WALL_TRANSFORM=False -> 이 마커가 곧 월드 (그대로 반환)
       USE_WALL_TRANSFORM=True  -> R_FW/t_FW 적용 (벽 마커 -> 월드)"""
    if not USE_WALL_TRANSFORM:
        return R_cm, t_cm
    R_w = R_cm @ R_FW
    t_w = R_cm @ t_FW + t_cm
    return R_w, t_w


class ManualCornerPicker:
    def __init__(self, frame, cam_matrix, dist_coeffs, auto_points=None, auto_id=None):
        self.base = frame.copy()
        self.H, self.W = frame.shape[:2]
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.mouse_pos = (0, 0)
        self.order = [0, 1, 2, 3]
        self.window = "Front Calibration (auto-detect or click 4 marker corners)"

        # 자동인식 결과가 있으면 그걸로 시작 (아직 사용자가 손대지 않은 상태 = is_auto True)
        if auto_points is not None:
            self.points = [tuple(p) for p in auto_points]
            self.is_auto = True
            self.auto_id = auto_id
        else:
            self.points = []
            self.is_auto = False
            self.auto_id = None

        self.scale = min(1.0, DISPLAY_MAX_W / self.W, DISPLAY_MAX_H / self.H)
        print(f"  🖥️ 원본 해상도 {self.W}x{self.H} -> 표시 배율 {self.scale:.3f}배로 축소 표시")
        mode_txt = "벽마커 변환(R_FW/t_FW 적용)" if USE_WALL_TRANSFORM else "직접 모드(클릭한 마커=월드)"
        print(f"  🧭 모드: {mode_txt}  (바꾸려면 파일 상단 USE_WALL_TRANSFORM 수정)")
        if self.is_auto:
            print(f"  🎯 자동인식 성공 (ID={self.auto_id}). 's'로 그대로 저장하거나, 'm'으로 직접 재클릭하세요.")
        else:
            print("  ⚠️ 자동인식 실패 또는 비활성 - 수동 클릭으로 진행하세요.")

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        ox, oy = x / self.scale, y / self.scale
        self.mouse_pos = (ox, oy)
        if self.is_auto:
            return  # 자동인식 결과 표시 중엔 클릭 무시. 'm'을 눌러야 수동모드로 전환됨.
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((ox, oy))
            print(f"  📍 {CORNER_LABELS[len(self.points)-1]} 클릭(원본좌표): ({ox:.1f}, {oy:.1f})")

    def _solve(self):
        """현재 4점 + 현재 order로 solvePnP 후 월드로 변환.
           성공 시 (R_cm, t_cm, R_w, t_w, rvec_w) 반환, 실패 시 None."""
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
        R_cm, _ = cv2.Rodrigues(rvec)
        t_cm = tvec.reshape(3, 1)
        R_w, t_w = marker_to_world(R_cm, t_cm)
        rvec_w, _ = cv2.Rodrigues(R_w)
        return R_cm, t_cm, R_w, t_w, rvec_w

    def _draw(self):
        vis = self.base.copy()

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

        solved = self._solve()
        status_txt, status_color = "", (255, 255, 255)
        if solved is not None:
            _, _, R_w, t_w, rvec_w = solved
            axis_len = MARKER_SIZE_CM * 1.5
            try:
                cv2.drawFrameAxes(vis, self.cam_matrix, self.dist_coeffs, rvec_w, t_w, axis_len, 4)
            except AttributeError:
                cv2.aruco.drawAxis(vis, self.cam_matrix, self.dist_coeffs, rvec_w, t_w, axis_len)
            mode_short = "\uc9c1\uc811" if not USE_WALL_TRANSFORM else "\ubca3\ub9c8\ucee4\ubcc0\ud658"
            src_txt = f"\uc790\ub3d9\uc778\uc2dd(ID={self.auto_id})" if self.is_auto else "\uc218\ub3d9\ud074\ub9ad"
            status_txt = f"[{mode_short}/{src_txt}] 4\uc810 \uc644\ub8cc - \uc6d4\ub4dc\ucd95: X(\ube68\uac04)\uc6b0\uce35 / Y(\ucd08\ub85d)\uc548\ucabd / Z(\ud30c\ub791)\uc704.  's' \uc800\uc7a5"
            status_txt += " / 'm' \uc218\ub3d9\uc7ac\ud074\ub9ad" if self.is_auto else " / 'z' \uc21c\uc11c\ubcc0\uacbd"
            status_color = (0, 220, 0)
        else:
            status_txt = f"{len(self.points)}/4\uc810 \ud074\ub9ad\ud558\uc138\uc694 (\ub2e4\uc74c: {CORNER_LABELS[len(self.points)] if len(self.points)<4 else ''})"
            status_color = (0, 200, 255)

        cv2.rectangle(vis, (0, 0), (self.W, 46), (0, 0, 0), -1)
        cv2.putText(vis, status_txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        help_txt = "m=manual re-click | click=corner select | u=undo | c=clear | z=rotate order | s/ENTER=save | q=abort"
        cv2.putText(vis, help_txt, (12, self.H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return vis, solved

    def _magnifier_patch(self):
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
            vis_disp = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA) if self.scale < 1.0 else vis

            mag = self._magnifier_patch()
            if mag is not None:
                ox, oy = disp_w - MAG_SIZE - 10, 56
                if oy + MAG_SIZE <= disp_h and ox >= 0:
                    vis_disp[oy:oy+MAG_SIZE, ox:ox+MAG_SIZE] = mag
                    cv2.putText(vis_disp, f"zoom x{MAG_ZOOM}", (ox, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(self.window, vis_disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('m'):
                # 자동인식 결과 버리고 수동클릭 모드로 전환
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
            elif key in (ord('s'), 13):
                if solved is None:
                    print("  ⚠️ 아직 4점을 다 안 찍었거나 solvePnP 실패")
                    continue
                cv2.destroyWindow(self.window)
                return solved
            elif key == ord('q'):
                cv2.destroyWindow(self.window)
                return None


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
    frame = grab_frame(FRONT_VIDEO_PATH, FRAME_INDEX)

    auto_points, auto_id = None, None
    if AUTO_DETECT:
        result = auto_detect_marker(frame)
        if result is not None:
            auto_points, auto_id = result
            print(f"  🎯 자동인식 성공: ID={auto_id} (오른쪽 마커 선택됨)" if PICK_RIGHTMOST
                  else f"  🎯 자동인식 성공: ID={auto_id}")
        else:
            print("  ⚠️ 자동인식 실패 - 수동 클릭으로 시작합니다.")

    picker = ManualCornerPicker(frame, K_MATRIX_FRONT, DIST_COEFFS_FRONT,
                                 auto_points=auto_points, auto_id=auto_id)
    solved = picker.run()
    cv2.destroyAllWindows()

    if solved is None:
        print("❌ 저장하지 않고 종료했습니다.")
        return

    R_cm, t_cm, R_w, t_w, rvec_w = solved
    np.savez(SAVE_PATH,
             R=R_cm, t=t_cm.reshape(3),               # raw 마커 pose (월드 변환 전)
             use_wall_transform=USE_WALL_TRANSFORM,
             corners_px=np.array(picker.points, dtype=np.float32),
             order=np.array(picker.order, dtype=int),
             marker_size_cm=MARKER_SIZE_CM,
             source_video=FRONT_VIDEO_PATH,
             frame_index=FRAME_INDEX,
             timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\n✅ 저장 완료: {SAVE_PATH}")
    print(f"   use_wall_transform={USE_WALL_TRANSFORM}")
    print(f"   R(raw marker)=\n{R_cm}")
    print(f"   t(raw marker)={t_cm.reshape(3)}")
    print(f"   R(world)=\n{R_w}")
    print(f"   t(world)={t_w.reshape(3)}")
    print("\n👉 메인 스크립트를 다시 실행하면 이 파일을 자동으로 읽어 정면 카메라 캘리브레이션에 사용합니다.")


if __name__ == "__main__":
    main()
