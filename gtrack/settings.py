# -*- coding: utf-8 -*-
"""
전체 상수 모음 (전처리 · 후처리 · 보정 NPZ 생성 공통)

- 이 파일의 값은 원본 코드(online_preprocess_final_20260930.py,
  postprocess_final_20261001.py, manual_calib_*.py)의 값과 동일하다.
- 실험마다 바꾸는 값과 장비가 바뀔 때만 바꾸는 값은 config.yaml에서 덮어쓴다.
  config.yaml에 없는 값은 이 파일의 기본값을 그대로 쓴다.
- 다른 모듈은 `from .. import settings` 후 `settings.이름`으로 값을 읽는다.
  configure()를 다시 호출하면 모든 모듈에 새 값이 바로 반영된다.

구역
  [A] 실험마다 바뀌는 값      : config.yaml이 채운다 (경로, 물 높이, 영상 시간 오프셋)
  [B] 장비가 바뀔 때만 바꾸는 값 : config.yaml에서 덮어쓴다 (카메라 K/D, 좌표계, 마커 등)
  [C] 알고리즘 상수 (수정 금지)   : 바꾸면 결과가 달라진다. 필요하면 config.yaml의 advanced에서만 덮어쓴다.
"""

import copy
import os

import cv2
import numpy as np

# ============================================================
# [A] 실험마다 바뀌는 값 (config.yaml → configure()가 채움)
# ============================================================
EXPERIMENT_NAME = None
DATA_ROOT = None
OUTPUT_DIR = None

# 입력 파일
TOP_VIDEO_PATH = None
FRONT_VIDEO_PATH = None
TOP_MODEL_PATH = None
FRONT_MODEL_PATH = None
MANUAL_CALIB_TOP_PATH = None
MANUAL_CALIB_FRONT_PATH = None

# 산출물 (OUTPUT_DIR 아래에 자동 지정)
RAW_EXCEL = None             # 전처리 산출물 3D_raw_online.xlsx (후처리 입력)
FINAL_EXCEL = None           # 후처리 산출물 3D_final.xlsx
ANNOTATED_VIDEO_PATH = None  # 주석 영상 (선택)
VERIFY_IMAGE_PATH = None     # 정면 좌표계 통합 검증 이미지

# 물 높이: 좌표값이 아니라 바닥(ID0 마커 평면)으로부터 실제 물 높이(cm)
WATER_SURFACE_Z_CM = 67.0

# 두 영상을 박수/편집으로 이미 맞췄다면 0.0을 유지한다.
# 값은 각 영상 원본 시각에 더해지는 정렬 오프셋이다.
TOP_TIME_OFFSET_SEC = 0.0
FRONT_TIME_OFFSET_SEC = 0.0

# 보정 NPZ 생성(notebooks/01)에만 사용
CALIB_TOP_VIDEO_PATH = None     # None이면 TOP_VIDEO_PATH 사용
CALIB_FRONT_VIDEO_PATH = None   # None이면 FRONT_VIDEO_PATH 사용
CALIB_FRAME_INDEX = 0           # 몇 번째 프레임에서 마커를 찍을지 (기본 첫 프레임)
# ★ 클릭할 정면 마커 종류. False = 수조 내부 바닥 마커(클릭한 마커=월드),
#   True = 수조 밖 벽 마커(R_FW/t_FW 적용)
CALIB_FRONT_USE_WALL_TRANSFORM = False
# ★ 같은 ID 마커가 정면 화면에 두 개 잡힐 때, 오른쪽(픽셀 center-x가 더 큰 쪽) 마커만 사용
CALIB_FRONT_PICK_RIGHTMOST = True
CALIB_AUTO_DETECT = True        # True: 자동인식 먼저 시도. False: 항상 수동클릭

# ============================================================
# [B] 장비가 바뀔 때만 바꾸는 값
# ============================================================
# ---- 영상 형식 ----
EXPECTED_RESOLUTION = (1920, 1080)
# 게이트/후처리 frame 상수의 기준 시간축.
EXPECTED_FPS = 59.94

# ---- ArUco 및 카메라 calibration ----
MARKER_SIZE_CM = 15.0
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# 카메라 렌즈 intrinsic(K, dist)은 체커보드로 측정한 고정값을 직접 사용한다.
# 실행 시 활성화되는 카메라는 top 1대와 front 1대뿐이다.
TOP_CAMERA_MODEL = "GoPro HERO13 Black"
FRONT_CAMERA_MODEL = "Insta360 Ace Pro 2"
K_MATRIX_TOP = np.array([
    [1.04157146e+03, 0.0, 9.49734448e+02],
    [0.0, 1.04797285e+03, 5.51602202e+02],
    [0.0, 0.0, 1.0],
], dtype=float)
DIST_COEFFS_TOP = np.array([
    [0.03618997, -0.04817693, -0.00320361, -0.00075223, 0.02632374]
], dtype=float)
K_MATRIX_FRONT = np.array([
    [734.42834248, 0.0, 958.12885964],
    [0.0, 738.92389777, 536.16766245],
    [0.0, 0.0, 1.0],
], dtype=float)
DIST_COEFFS_FRONT = np.array([
    [1.51047198e-02, -8.55712758e-03, -1.38100761e-03,
     1.74587731e-05, 5.72933769e-03]
], dtype=float)

# ---- 좌표계 통합: 전면 벽 marker 좌표계 → ID0 world ----
R_FW = np.array([[1, 0, 0],
                 [0, 0, 1],
                 [0, -1, 0]], dtype=float)
t_FW = np.array([52.0, -15.5, -10]).reshape(3, 1)

# ---- 굴절 경계 ----
N_AIR = 1.0
N_WATER = 1.333
FRONT_WALL_Y_CM = -11.6
INTERFACE_FRONT = None   # configure()/_rebuild_derived()가 FRONT_WALL_Y_CM으로 만든다
INTERFACE_TOP = None     # configure()/_rebuild_derived()가 WATER_SURFACE_Z_CM으로 만든다

# ---- 출력 좌표계: ID0 축 방향을 유지하고 원점만 모터 중심으로 이동 ----
MOTOR_CENTER_IN_ID0_CM = np.array([-26.5, 28.0, 17.1], dtype=float)

# ---- YOLO 대상 클래스 ----
# 대상 클래스는 번호가 아니라 이름으로 찾는다. 모델 로드 직후 번호로 변환한다.
# 공백·밑줄·대소문자는 무시하고 비교한다('white cube' == 'whitecube').
TARGET_CLASS_NAMES = ('whitecube',)

# ============================================================
# [C] 알고리즘 상수 — 수정 금지 (변경 시 결과가 달라짐)
# ============================================================
# ---------------- 전처리 실행 모드 ----------------
HEADLESS = True
DATASET_IS_WET = True
# 실제 궤적 생성에 사용할 geometry. ON/OFF 후보는 항상 같은 픽셀에서 동시에 저장한다.
ENABLE_REFRACTION = True
FPS_TOLERANCE = 0.10
# True여도 해상도만 강제한다. FPS 차이는 아래 시간기준 동기화로 허용한다.
STRICT_VIDEO_FORMAT = True
VERIFY_UNIFICATION = True

# ---------------- 시간기준 동기화 (보조 기능) ----------------
# None이면 두 입력 중 더 높은 FPS를 공통 출력 시간축으로 사용한다.
SYNC_OUTPUT_FPS = None
SYNC_METHOD = 'nearest_timestamp'
# None이면 느린 영상 한 프레임 간격의 60%를 자동 허용한다.
SYNC_MAX_ERROR_SEC = None
SYNC_MAX_ERROR_FRAME_FRACTION = 0.60

# ---------------- 주석 영상 ----------------
SAVE_ANNOTATED_VIDEO = True
ANNOTATED_VIDEO_SCALE = 0.5
# None이면 공통 출력 시간축 FPS로 저장한다.
ANNOTATED_VIDEO_FPS = None

# 존 치수 (draw_overlay 시각화용)
TOP_ZONE_W_CM = 50.0
TOP_ZONE_H_CM = 50.0
FRONT_ZONE_W_CM = 50.0
FRONT_ZONE_ABOVE_CM = 25.0
FRONT_ZONE_BELOW_CM = 5.0

# ---------------- ① 물체 검출 ----------------
CONF_CUT = 0.4                              # YOLO conf 하한 (top·front 동일)
YOLO_IMGSZ = 640                            # 추론 입력 크기(긴 변 기준)

# ---------------- ④ 후보 수용 판정 및 궤적 확정 ----------------
# 온라인 measurement gate 전용 측정 표준편차.
# 후처리 RTS의 R과 역할은 같아 보이지만 별도 상수로 유지한다.
SIG_GATE_MEAS = np.array([0.12, 0.11, 0.27], dtype=float)  # cm
SIGMA_A_ONLINE = 0.10                                     # cm/frame²
GATE_CHI2 = 11.34                                         # χ²(df=3, 99%)

# jump gate: 마지막 수용 위치 기준 frame당 최대 변위(cm/frame).
JUMP_LIMIT_CM_PER_FRAME = 2.0
JUMP_LIMIT_ABS_CM = 30.0  # 장시간 공백 후 허용 변위가 과도하게 커지는 것을 막는 절대 상한

# ray gap gate: stereo 후보의 두 광선 최근접 거리 상한(cm).
# 한쪽 카메라가 다른 물체·반사를 잡아 잘못 짝지어진 후보를 거른다.
# 경과 시간과 무관하게 적용되므로, 공백 뒤 재획득 시점의 오검출도 걸러진다.
# 12개 시도 raw 기준(final_2 제외): 정상 최대 5.78 cm(final_1), 오검출 11.0~11.1 cm.
RAY_GAP_MAX_CM = 7.0

CONFIRM_N, CONFIRM_M = 10, 7  # 최근 10개의 실제 stereo candidate 시도 중 7회 gate 통과
MAX_BOTH_LOST_FRAMES = 30
# 양쪽 카메라가 모두 DETECTED인데도 유효 3D를 연속해서 못 얻는 경우에만 사용.
# 한쪽 카메라 소실(single-view) 동안에는 이 counter를 누적하지 않는다.
MAX_STEREO_NO_ACCEPT_FRAMES = 150
MAX_REJECT_STREAK = 10

# ---------------- ⑥ Hampel 이상치 제거 ----------------
HAMPEL_K = 15
HAMPEL_NSIGMA = 3.5
# Hampel residual threshold의 축별 절대 하한.
HAMPEL_ABS_FLOOR = np.array([0.50, 0.50, 0.30], dtype=float)  # cm

# ---------------- ⑧ 과정 잡음 ----------------
SIGMA_A_FALLBACK = np.array([0.030, 0.030, 0.025], dtype=float)

# ---------------- ⑩ 단일시점 구간 좌표 복원 ----------------
# hidden-axis 초기 context 검색. 우선 30 frame, 초기 속도 prior에 필요한 실제 raw가
# 부족하면 60 frame까지 한 번 확장한다.
ANCHOR_SEARCH_INITIAL = 30
ANCHOR_SEARCH_EXTENDED = 60
ANCHOR_MODEL_MIN_POINTS = 3

# 같은 recovery mode run 사이의 짧은 raw/TENTATIVE/reject island를 하나의 연속 episode로
# 묶는 최대 간격. GRAB_MIN과 역할이 다른 독립 상수다.
HIDDEN_EPISODE_LINK_MAX = 60
# 중간 raw island에 실제 stereo가 이 길이 이상 연속되면 안정적 재획득으로 보고
# episode를 분리한다. 2+3처럼 끊어진 짧은 island는 새 velocity prior로 재시작하지 않는다.
HIDDEN_REACQUIRE_MIN_CONSECUTIVE_RAW = 5

# 1D hidden-axis CV model 설정.
HIDDEN_AXIS_MOTION_MODEL = 'CLEAN_LINEAR__INTERMITTENT_1D_KALMAN_RTS'
HIDDEN_HUBER_K = 1.345  # Huber 95% efficiency 표준값
HIDDEN_Q_SCALE_Z = 0.3
HIDDEN_Q_SCALE_Y = 0.03

# posterior uncertainty stop. 예비 실험과 기존 데이터 검토에서 정상 궤적을
# 과도하게 제한하지 않도록 보수적으로 정한 최종 운영값이다.
HIDDEN_STD_STOP_Z_CM = 6.0
HIDDEN_STD_STOP_Y_CM = 2.0
HIDDEN_STD_STOP_THRESHOLDS_FINALIZED = True

# Z는 낙하 운동을 포함할 수 있으므로 초기 velocity prior의 절대 sanity bound를 유지한다.
V_MAX_Z = 0.50  # cm/frame
# Front-only 복원에서 시간적으로 추정하는 Y는 수평방향이므로 더 보수적으로 제한한다.
V_MAX_Y = 0.10  # cm/frame
# Front 광선으로 얻은 X가 anchor 기반 범위를 크게 벗어나면 clipping하지 않고 LOST로 둔다.
V_MAX_X_FRONT_RAY = 0.10  # cm/frame
FRONT_X_SANITY_MARGIN = 2.0  # cm

# [REV-09 | 2026-09-13] 작업영역 내부 TOP_RAY 전방 말단도 무제한 연장하지 않고 300 frame hard cap을 둔다.
# 일반 단측 예측은 150 frame, 조건을 만족하는 TOP_RAY 전방 말단은 300 frame까지 허용한다.
# 두 경우 모두 posterior standard deviation 종료 기준은 별도로 유지한다.
MAX_EXTRAP_LEN = 150
TOP_ZONE_MAX_EXTRAP_LEN = 300

# ---------------- ⑪ 작업영역 판정 · GRAB ----------------
ZONE_X_HALF = 25.0
ZONE_Y_HALF = 25.0
# [2026-09-30] 작업영역(In_Zone) 판정 하한. 그리퍼 안 안착 위치(Z 약 2~3 cm)를 포함하도록 0 cm.
# (REV-11의 3 cm 통일에서 변경. 복원 Z 하한은 아래 ZONE_TRACK_CENTER_Z_FLOOR로 분리)
ZONE_Z_LOW = 0.0
ZONE_Z_HIGH = 25.0

# TOP_RAY 복원 중 Top 검출 광선이 gripper zone을 통과하는 frame에 적용하는
# 물리적 Z 하한(복원값 clamp). 작업영역 판정 하한과 별개로 3 cm를 유지한다.
ZONE_TRACK_CENTER_Z_FLOOR = 3.0

GRAB_MIN = 60  # 약 1.001 s @ 59.94 fps

# ---------------- ⑫ 표시용 작업영역 연결 ----------------
# 실제 좌표를 만들지 않고 zone 음영만 한 번 연결하는 시각화 전용 bridge.
# anchor 검색/복원/GRAB/RTS에는 절대 사용하지 않는다.
# 1) 짧은 공백: 7 frame 이하이면 기존처럼 양옆 원본 pre-RTS in-zone만으로 연결.
# 2) 긴 공백: 길이 제한 없이, 공백 동안 매 frame 적어도 한 카메라가 DETECTED이고
#    공백 양옆에서 가장 가까운 pre-RTS 유효 좌표 3점씩이 모두 zone 내부일 때만 연결.
# 두 규칙 모두 immutable pre-RTS snapshot을 한 번만 읽으며 bridge 결과를 재사용하지 않는다.
DISPLAY_ZONE_BRIDGE_MAX = 7
DISPLAY_ZONE_STABLE_SEARCH = 30
DISPLAY_ZONE_STABLE_MIN_POINTS_PER_SIDE = 3

# ---------------- ⑬ 표시용 3차원 RTS ----------------
# RTS measurement noise. Top-ray와 Front-ray의 오차 구조가 다르므로 분리한다.
SIG_RTS_RAW = np.array([0.12, 0.11, 0.27], dtype=float)
# 아래 두 값은 예비 실험과 기존 데이터 분석을 바탕으로 정한 최종 운영값이다.
SIG_RTS_RECON_TOP = np.array([0.10, 0.10, 0.30], dtype=float)
SIG_RTS_RECON_FRONT = np.array([0.10, 0.10, 0.30], dtype=float)
SIG_RTS_RECON_TOP_FINALIZED = True
SIG_RTS_RECON_FRONT_FINALIZED = True

# ---------------- ⑭ 프레임별 속도 ----------------
VEL_HALF_WINDOW = 5                  # 창 반폭 k: t-5 ~ t+5, 11프레임
VEL_MAX_NOISE_RATIO = 2.0            # 11프레임 전부 사용 대비 잡음 배수 상한
VEL_MAX_CENTER_OFFSET_FRAMES = 2.0   # 사용 프레임 평균 위치와 t의 차이 상한
VEL_FALLBACK_FPS = 59.94             # raw에 Time_sec이 없는 구버전 파일에서만 사용
VEL_SOURCE_LABELS = ['OBSERVED', 'RAY_CONSTRAINED', 'MODEL_INTERP', 'MODEL_EXTRAP']
VEL_SLOW_PHASE_FLUCTUATION_NOTE = (
    '느린 구간 속력 흔들림 3D 약 0.55–1.24 cm/s '
    '(움직이는 영상 6개에서 추정한 상한, 정지점 영상 측정 후 갱신 예정)'
)

# ---------------- 코드 내부 열 이름 · 라벨 ----------------
MEAS_COLS = ['Meas_X', 'Meas_Y', 'Meas_Z']
RAW_MEAS_COLS = ['Raw_Meas_X', 'Raw_Meas_Y', 'Raw_Meas_Z']

RECOVERY_NONE = 'NONE'
RECOVERY_TOP_RAY = 'TOP_RAY'      # FRONT_MISSING: Top ray + estimated Z
RECOVERY_FRONT_RAY = 'FRONT_RAY'  # TOP_MISSING: Front ray + estimated Y

RESULT_STATE_MAP = {
    'DETECTED': 'DETECTED',
    'DETECTED_ZONE': 'DETECTED',
    'SUB': 'RECOVERED',
    'SUB_ZONE': 'RECOVERED',
    'LOST': 'LOST',
    'GRAB': 'GRAB',
}

# raw 엑셀 열 이름표: 코드 내부 이름 -> 엑셀 이름.
# 전처리(저장)와 후처리(읽기)가 이 표 하나를 같이 쓴다.
RAW_COLUMN_RENAME = {
    'Time_sec': 'Time_s',
    'Status_Top': 'Top_Status',
    'Status_Front': 'Front_Status',
    'Top_conf': 'Top_Conf',
    'Front_conf': 'Front_Conf',
    'Top_px_u': 'Top_U_px',
    'Top_px_v': 'Top_V_px',
    'Front_px_u': 'Front_U_px',
    'Front_px_v': 'Front_V_px',
    'Meas_X': 'Meas_X_cm',
    'Meas_Y': 'Meas_Y_cm',
    'Meas_Z': 'Meas_Z_cm',
    'Candidate_X': 'Candidate_X_cm',
    'Candidate_Y': 'Candidate_Y_cm',
    'Candidate_Z': 'Candidate_Z_cm',
    'Candidate_ON_X': 'Candidate_ON_X_cm',
    'Candidate_ON_Y': 'Candidate_ON_Y_cm',
    'Candidate_ON_Z': 'Candidate_ON_Z_cm',
    'Candidate_OFF_X': 'Candidate_OFF_X_cm',
    'Candidate_OFF_Y': 'Candidate_OFF_Y_cm',
    'Candidate_OFF_Z': 'Candidate_OFF_Z_cm',
    'Ray_s_Top_cm': 'Top_Ray_Dist_cm',
    'Ray_s_Front_cm': 'Front_Ray_Dist_cm',
    'Online_X': 'KF_X_cm',
    'Online_Y': 'KF_Y_cm',
    'Online_Z': 'KF_Z_cm',
    'KF_PosStd_X': 'KF_Std_X_cm',
    'KF_PosStd_Y': 'KF_Std_Y_cm',
    'KF_PosStd_Z': 'KF_Std_Z_cm',
}
RAW_COLUMN_RENAME_INVERSE = None   # _rebuild_derived()가 만든다

# raw 시트별 열 (코드 내부 이름 기준). 목록에 없는 새 열은 Trajectory 끝에 붙는다.
RAW_SHEET_COLUMNS = {
    'Trajectory': [
        'Frame', 'Time_sec', 'Status_Top', 'Status_Front', 'Top_conf', 'Front_conf',
        'Top_px_u', 'Top_px_v', 'Front_px_u', 'Front_px_v',
        'Meas_X', 'Meas_Y', 'Meas_Z', 'Meas_Status', 'Missing_Reason', 'Ray_Gap_cm',
    ],
    'Triangulation': [
        'Frame', 'Candidate_X', 'Candidate_Y', 'Candidate_Z',
        'Candidate_ON_X', 'Candidate_ON_Y', 'Candidate_ON_Z',
        'Candidate_OFF_X', 'Candidate_OFF_Y', 'Candidate_OFF_Z',
        'Geometry_Status_ON', 'Geometry_Status_OFF',
        'Ray_s_Top_cm', 'Ray_s_Front_cm', 'Ray_Gap_ON_cm', 'Ray_Gap_OFF_cm',
    ],
    'Online_Filter': [
        'Frame', 'Track_State', 'KF_Updated', 'Online_X', 'Online_Y', 'Online_Z',
        'KF_PosStd_X', 'KF_PosStd_Y', 'KF_PosStd_Z',
    ],
}
# 엑셀에 쓰지 않는 열: Sig_*는 상수(SIG_GATE_MEAS, Configuration에 기록), Tracker_Ref_Frame은
# Time_sec으로 계산되는 필터 내부값, 동기화 열은 Summary 통계와 Missing_Reason으로 대체한다.
RAW_EXCLUDED_COLUMNS = [
    'Sig_X', 'Sig_Y', 'Sig_Z', 'Tracker_Ref_Frame',
    'Top_Source_Frame', 'Front_Source_Frame', 'Top_Time_sec', 'Front_Time_sec',
    'Top_Sync_Time_sec', 'Front_Sync_Time_sec', 'Sync_Error_ms', 'Sync_Abs_Error_ms',
    'Sync_Status', 'Top_Frame_Reused', 'Front_Frame_Reused',
]

# 산출물 파일 이름 (OUTPUT_DIR 아래)
RAW_EXCEL_NAME = '3D_raw_online.xlsx'
FINAL_EXCEL_NAME = '3D_final.xlsx'
ANNOTATED_VIDEO_NAME = 'analyzed_output.mp4'
VERIFY_IMAGE_NAME = 'verify_unification.png'
CONFIG_SNAPSHOT_NAME = 'config_used.yaml'


# ============================================================
# 파생값 계산
# ============================================================
def _rebuild_derived():
    """다른 상수로부터 계산되는 값을 다시 만든다. 값을 바꾼 뒤 반드시 호출한다."""
    g = globals()
    g['INTERFACE_FRONT'] = {
        'axis': 1, 'value': FRONT_WALL_Y_CM,
        'normal_to_air': np.array([0.0, -1.0, 0.0]),
    }
    g['INTERFACE_TOP'] = {
        'axis': 2, 'value': WATER_SURFACE_Z_CM,
        'normal_to_air': np.array([0.0, 0.0, 1.0]),
    }
    g['RAW_COLUMN_RENAME_INVERSE'] = {v: k for k, v in RAW_COLUMN_RENAME.items()}
    if OUTPUT_DIR:
        g['RAW_EXCEL'] = os.path.join(OUTPUT_DIR, RAW_EXCEL_NAME)
        g['FINAL_EXCEL'] = os.path.join(OUTPUT_DIR, FINAL_EXCEL_NAME)
        g['ANNOTATED_VIDEO_PATH'] = os.path.join(OUTPUT_DIR, ANNOTATED_VIDEO_NAME)
        g['VERIFY_IMAGE_PATH'] = os.path.join(OUTPUT_DIR, VERIFY_IMAGE_NAME)


_rebuild_derived()

# 기본값 스냅숏: configure()가 매번 기본값에서 다시 시작하도록 보관한다.
_DEFAULTS = {
    k: copy.deepcopy(v) for k, v in globals().items()
    if (k.isupper() or k == 't_FW') and not k.startswith('_')
}


# ============================================================
# config.yaml 적용
# ============================================================
# config.yaml 항목 -> settings 이름
_CONFIG_MAP = {
    ('experiment', 'name'): 'EXPERIMENT_NAME',
    ('experiment', 'data_root'): 'DATA_ROOT',
    ('inputs', 'top_video'): 'TOP_VIDEO_PATH',
    ('inputs', 'front_video'): 'FRONT_VIDEO_PATH',
    ('inputs', 'top_weights'): 'TOP_MODEL_PATH',
    ('inputs', 'front_weights'): 'FRONT_MODEL_PATH',
    ('inputs', 'calib_top_npz'): 'MANUAL_CALIB_TOP_PATH',
    ('inputs', 'calib_front_npz'): 'MANUAL_CALIB_FRONT_PATH',
    ('water', 'surface_z_cm'): 'WATER_SURFACE_Z_CM',
    ('sync', 'top_time_offset_sec'): 'TOP_TIME_OFFSET_SEC',
    ('sync', 'front_time_offset_sec'): 'FRONT_TIME_OFFSET_SEC',
    ('calibration', 'top_video'): 'CALIB_TOP_VIDEO_PATH',
    ('calibration', 'front_video'): 'CALIB_FRONT_VIDEO_PATH',
    ('calibration', 'frame_index'): 'CALIB_FRAME_INDEX',
    ('calibration', 'front_use_wall_transform'): 'CALIB_FRONT_USE_WALL_TRANSFORM',
    ('calibration', 'front_pick_rightmost'): 'CALIB_FRONT_PICK_RIGHTMOST',
    ('calibration', 'auto_detect'): 'CALIB_AUTO_DETECT',
    ('video', 'expected_resolution'): 'EXPECTED_RESOLUTION',
    ('video', 'expected_fps'): 'EXPECTED_FPS',
    ('video', 'save_annotated_video'): 'SAVE_ANNOTATED_VIDEO',
    ('models', 'target_class_names'): 'TARGET_CLASS_NAMES',
    ('cameras', 'top', 'model'): 'TOP_CAMERA_MODEL',
    ('cameras', 'top', 'K'): 'K_MATRIX_TOP',
    ('cameras', 'top', 'dist'): 'DIST_COEFFS_TOP',
    ('cameras', 'front', 'model'): 'FRONT_CAMERA_MODEL',
    ('cameras', 'front', 'K'): 'K_MATRIX_FRONT',
    ('cameras', 'front', 'dist'): 'DIST_COEFFS_FRONT',
    ('setup', 'marker_size_cm'): 'MARKER_SIZE_CM',
    ('setup', 'motor_center_in_id0_cm'): 'MOTOR_CENTER_IN_ID0_CM',
    ('setup', 'front_wall_y_cm'): 'FRONT_WALL_Y_CM',
    ('setup', 'front_wall_marker_R'): 'R_FW',
    ('setup', 'front_wall_marker_t'): 't_FW',
}

# 상대경로를 DATA_ROOT 기준으로 바꿀 항목
_PATH_KEYS = (
    'TOP_VIDEO_PATH', 'FRONT_VIDEO_PATH', 'TOP_MODEL_PATH', 'FRONT_MODEL_PATH',
    'MANUAL_CALIB_TOP_PATH', 'MANUAL_CALIB_FRONT_PATH',
    'CALIB_TOP_VIDEO_PATH', 'CALIB_FRONT_VIDEO_PATH',
)


def _coerce_like(name, value, default):
    """config 값을 기본값과 같은 형태로 바꾼다 (배열 모양까지 맞춤)."""
    if value is None:
        return None
    if isinstance(default, np.ndarray):
        arr = np.asarray(value, dtype=float)
        if arr.size != default.size:
            raise ValueError(f'{name}: 원소 {default.size}개가 필요한데 {arr.size}개가 들어왔습니다.')
        return arr.reshape(default.shape)
    if isinstance(default, tuple):
        return tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    return value


def _get_nested(cfg, keys):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return False, None
        node = node[k]
    return True, node


def _resolve_path(path):
    if path is None or str(path).strip() == '':
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path) or DATA_ROOT is None:
        return path
    return os.path.join(DATA_ROOT, path)


def reset_defaults():
    """모든 값을 이 파일의 기본값으로 되돌린다."""
    globals().update(copy.deepcopy(_DEFAULTS))
    _rebuild_derived()


def apply_config(cfg):
    """dict 형태 config를 적용한다. 항상 기본값에서 다시 시작한다."""
    reset_defaults()
    g = globals()
    for keys, name in _CONFIG_MAP.items():
        found, value = _get_nested(cfg, keys)
        if found and value is not None:
            g[name] = _coerce_like(name, value, _DEFAULTS.get(name))

    for name, value in (cfg.get('advanced') or {}).items():
        if name not in _DEFAULTS:
            raise KeyError(f'advanced: settings.py에 없는 이름입니다: {name}')
        g[name] = _coerce_like(name, value, _DEFAULTS[name])

    for name in _PATH_KEYS:
        g[name] = _resolve_path(g[name])

    output_root = (cfg.get('outputs') or {}).get('output_root', 'results')
    if EXPERIMENT_NAME:
        g['OUTPUT_DIR'] = os.path.join(_resolve_path(output_root) or output_root, str(EXPERIMENT_NAME))
    _rebuild_derived()


def load_config_file(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def configure(config_path, make_output_dir=True):
    """config.yaml을 읽어 적용하고, 결과 폴더를 만든 뒤 적용 값을 저장한다."""
    cfg = load_config_file(config_path)
    apply_config(cfg)
    if make_output_dir and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_snapshot(os.path.join(OUTPUT_DIR, CONFIG_SNAPSHOT_NAME), source=config_path)
    return cfg


def snapshot():
    """현재 적용된 값 전체를 dict로 반환한다 (배열은 list로 변환)."""
    out = {}
    for k in _DEFAULTS:
        v = globals()[k]
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, dict):
            v = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
        out[k] = v
    return out


def save_snapshot(path, source=None):
    """재현용으로 실제 사용한 값 전체를 YAML로 저장한다."""
    import yaml
    data = {'source_config': source, 'values': snapshot()}
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


REQUIRED_KEYS_BY_STAGE = {
    'calibration': ['EXPERIMENT_NAME', 'DATA_ROOT'],
    'online': ['EXPERIMENT_NAME', 'DATA_ROOT', 'TOP_VIDEO_PATH', 'FRONT_VIDEO_PATH',
               'TOP_MODEL_PATH', 'FRONT_MODEL_PATH',
               'MANUAL_CALIB_TOP_PATH', 'MANUAL_CALIB_FRONT_PATH'],
    'postprocess': ['EXPERIMENT_NAME', 'DATA_ROOT',
                    'MANUAL_CALIB_TOP_PATH', 'MANUAL_CALIB_FRONT_PATH'],
}

FILES_BY_STAGE = {
    'calibration': [],
    'online': ['TOP_VIDEO_PATH', 'FRONT_VIDEO_PATH', 'TOP_MODEL_PATH', 'FRONT_MODEL_PATH',
               'MANUAL_CALIB_TOP_PATH', 'MANUAL_CALIB_FRONT_PATH'],
    'postprocess': ['MANUAL_CALIB_TOP_PATH', 'MANUAL_CALIB_FRONT_PATH', 'RAW_EXCEL'],
}


def check(stage):
    """실행 전 점검. 문제가 있으면 어떤 항목을 고쳐야 하는지 알려주고 멈춘다."""
    problems = []
    g = globals()
    for name in REQUIRED_KEYS_BY_STAGE[stage]:
        if g.get(name) in (None, ''):
            problems.append(f'{name} 값이 비어 있습니다 (config.yaml 확인).')
    for name in FILES_BY_STAGE[stage]:
        path = g.get(name)
        if path and not os.path.exists(path):
            hint = ''
            if name == 'RAW_EXCEL':
                hint = ' — 전처리(notebooks/02)를 먼저 실행했는지 확인'
            elif name.startswith('MANUAL_CALIB'):
                hint = ' — 보정 NPZ 생성(notebooks/01)을 먼저 실행했는지 확인'
            problems.append(f'{name} 경로가 존재하지 않습니다: {path}{hint}')
    if not (20.0 <= float(WATER_SURFACE_Z_CM) <= 150.0):
        problems.append(f'WATER_SURFACE_Z_CM={WATER_SURFACE_Z_CM}: 바닥 기준 물 높이(cm)가 맞는지 확인')
    if K_MATRIX_TOP.shape != (3, 3) or K_MATRIX_FRONT.shape != (3, 3):
        problems.append('카메라 K 행렬은 3x3이어야 합니다.')
    if problems:
        msg = '\n'.join(f'  - {p}' for p in problems)
        raise RuntimeError(f'[{stage}] 설정 점검 실패:\n{msg}')
    print(f'[{stage}] 설정 점검 통과')


def print_summary(stage=None):
    """실험마다 확인할 값을 한눈에 출력한다."""
    rows = [
        ('실험 이름', EXPERIMENT_NAME),
        ('결과 폴더', OUTPUT_DIR),
        ('상단 영상', TOP_VIDEO_PATH),
        ('정면 영상', FRONT_VIDEO_PATH),
        ('상단 YOLO 가중치', TOP_MODEL_PATH),
        ('정면 YOLO 가중치', FRONT_MODEL_PATH),
        ('상단 보정 NPZ', MANUAL_CALIB_TOP_PATH),
        ('정면 보정 NPZ', MANUAL_CALIB_FRONT_PATH),
        ('물 높이 WATER_SURFACE_Z_CM', WATER_SURFACE_Z_CM),
        ('영상 오프셋 top/front [s]', f'{TOP_TIME_OFFSET_SEC} / {FRONT_TIME_OFFSET_SEC}'),
        ('모터 중심 MOTOR_CENTER_IN_ID0_CM', MOTOR_CENTER_IN_ID0_CM.tolist()),
    ]
    if stage == 'calibration':
        rows += [
            ('보정용 상단 영상', CALIB_TOP_VIDEO_PATH or TOP_VIDEO_PATH),
            ('보정용 정면 영상', CALIB_FRONT_VIDEO_PATH or FRONT_VIDEO_PATH),
            ('보정 프레임 번호', CALIB_FRAME_INDEX),
            ('정면 벽 마커 변환 사용', CALIB_FRONT_USE_WALL_TRANSFORM),
        ]
    width = max(len(r[0]) for r in rows)
    for k, v in rows:
        print(f'{k:<{width}} : {v}')
