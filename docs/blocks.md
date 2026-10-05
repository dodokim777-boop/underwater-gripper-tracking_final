# 블록별 코드 · 변수 정리

다이어그램(`docs/pipeline_diagram.drawio`)의 블록 번호와 코드 파일을 1:1로 맞췄다.
표의 "코드 내부 이름"은 프로그램 안에서 쓰는 열 이름, "엑셀 이름"은 결과 파일에 저장되는 열 이름이다.
두 이름이 같으면 엑셀 이름 칸에 `=`로 표시했다. 상수는 모두 `gtrack/settings.py`에 있다.

> 코드 내부 이름과 엑셀 이름이 다른 열이 있다. 특히 **`X_cm`은 코드 내부와 Result 시트에서 뜻이 다르다** (아래 "이름 주의" 참고).

---

## 입력

| 다이어그램 | 코드 | 설정 (config.yaml → settings) |
|---|---|---|
| 상단 · 정면 영상 (프레임 정렬됨) | `online/video_sync.py` `open_videos()` | `inputs.top_video`, `inputs.front_video` → `TOP_VIDEO_PATH`, `FRONT_VIDEO_PATH` |
| YOLO 가중치 top / front .pt | `online/b01_detection.py` `load_models()` | `inputs.top_weights`, `inputs.front_weights`, `models.target_class_names` |
| 보정값 NPZ: R, t | 생성 `calibration/core.py`, 읽기 `online/calibration_io.py`, `postprocess/inputs.py` | `inputs.calib_top_npz`, `inputs.calib_front_npz` |
| 보정값 소스 고정값: K, D | `settings.K_MATRIX_*`, `DIST_COEFFS_*` | `cameras.top/front.K`, `.dist` |

시간기준 최근접 프레임 동기화(`SequentialNearestSampler`)는 서로 다른 FPS 영상을 복구하려고 넣은 보조 기능이며 다이어그램에는 없다.
정렬된 영상에서는 같은 번호 프레임끼리 짝지어진다.

---

## 온라인 전처리 → `3D_raw_online.xlsx`

### ① 물체 검출 (YOLO) — `online/b01_detection.py`

| 함수 | 역할 |
|---|---|
| `resolve_class_id` | 모델 클래스 이름에서 대상 클래스 번호를 찾음 (`TARGET_CLASS_NAMES`) |
| `pick_best_target` | 대상 클래스 중 conf 최대 상자의 중심 픽셀 · conf |
| `CachedDetector.detect` | 같은 원본 프레임이 재사용되면 YOLO 결과 재사용 |
| `pairing_reason` | 다이어그램 마름모 "두 시점 검출 성립?" |

| 코드 내부 이름 | 엑셀 이름 (Trajectory 시트) | 내용 |
|---|---|---|
| `Status_Top`, `Status_Front` | `Top_Status`, `Front_Status` | DETECTED / LOST |
| `Top_px_u`, `Top_px_v` | `Top_U_px`, `Top_V_px` | 상단 상자 중심 픽셀 |
| `Front_px_u`, `Front_px_v` | `Front_U_px`, `Front_V_px` | 정면 상자 중심 픽셀 |
| `Top_conf`, `Front_conf` | `Top_Conf`, `Front_Conf` | YOLO conf |

상수: `CONF_CUT=0.4`, `YOLO_IMGSZ=640`, `TARGET_CLASS_NAMES=('whitecube',)`

### ② 광선 생성 · 평면 경계 굴절 — `online/b02_rays.py` → `geometry.py`

| 함수 | 역할 |
|---|---|
| `_camera_ray_world(pixel, calib)` | 픽셀 → 카메라 중심 C, world 광선 방향 d (왜곡 보정 포함) |
| `_refract_at_plane(C, d, interface)` | 평면 경계에서 벡터형 Snell 굴절 → (경계 교점, 굴절 방향) |

상수: `N_AIR=1.0`, `N_WATER=1.333`, `WATER_SURFACE_Z_CM`(config `water.surface_z_cm`), `FRONT_WALL_Y_CM=-11.6`
→ `INTERFACE_TOP`(수면, Z 평면), `INTERFACE_FRONT`(정면 벽, Y 평면). 후처리 ⑨⑩도 같은 함수를 쓴다.

### ③ 두 광선 최근접점 삼각측량 — `online/b03_triangulation.py`

| 함수 | 역할 |
|---|---|
| `_closest_points_on_forward_rays` | 두 광선의 최근접 두 점 → 중점(3D 후보), 두 점 거리(Ray gap) |
| `triangulate_pair(..., apply_refraction)` | 굴절 ON/OFF 한 가지 계산 |
| `triangulate_on_off` | 같은 픽셀로 ON(궤적용) · OFF(진단용) 동시 계산 |

| 코드 내부 이름 | 엑셀 이름 | 시트 | 내용 |
|---|---|---|---|
| `Candidate_ON_X/Y/Z` | `Candidate_ON_X/Y/Z_cm` | Triangulation | 굴절 적용 후보 (모터 중심 기준) |
| `Candidate_OFF_X/Y/Z` | `Candidate_OFF_X/Y/Z_cm` | Triangulation | 굴절 미적용 후보 (진단용) |
| `Candidate_X/Y/Z` | `Candidate_X/Y/Z_cm` | Triangulation | ④-1 거리 게이트를 통과한 후보 |
| `Geometry_Status_ON/OFF` | = | Triangulation | VALID / FAILED / NOT_PAIRED |
| `Ray_Gap_cm` | = | Trajectory | 사용한(ON) 후보의 두 광선 최근접 거리 |
| `Ray_Gap_ON_cm`, `Ray_Gap_OFF_cm` | = | Triangulation | ON/OFF 각각의 광선 거리 |
| `Ray_s_Top_cm`, `Ray_s_Front_cm` | `Top_Ray_Dist_cm`, `Front_Ray_Dist_cm` | Triangulation | 경계 교점에서 최근접점까지 광선 길이 |

### ④ 후보 수용 판정 및 궤적 확정 — `online/b04_tracker.py`

순서: `ray_gap_gate`(광선 간 거리) → `OnlineTracker.step`(변위 게이트 → Mahalanobis 게이트 → 7-of-10 확정)

| 코드 내부 이름 | 엑셀 이름 | 시트 | 내용 |
|---|---|---|---|
| `Meas_Status` | = | Trajectory | TRIANGULATED / NO_CANDIDATE / REJECTED_JUMP / REJECTED_MAHALANOBIS / TENTATIVE_HOLD / *_RESET |
| `Missing_Reason` | = | Trajectory | NONE / TOP_MISSING / FRONT_MISSING / BOTH_MISSING / RAY_GAP_EXCEEDED / TRIANGULATION_FAILED / REJECTED_* / TENTATIVE_HOLD / SYNC_OUT_OF_TOLERANCE |
| `Track_State` | = | Online_Filter | LOST / TENTATIVE / CONFIRMED |
| `KF_Updated` | = | Online_Filter | 이 프레임에 필터 update 성공 여부 |
| `Online_X/Y/Z` | `KF_X/Y/Z_cm` | Online_Filter | 게이트용 KF 상태 (update 성공 프레임만) |
| `KF_PosStd_X/Y/Z` | `KF_Std_X/Y/Z_cm` | Online_Filter | KF 위치 표준편차 |

상수: `RAY_GAP_MAX_CM=7.0`, `JUMP_LIMIT_CM_PER_FRAME=2.0`, `JUMP_LIMIT_ABS_CM=30.0`, `GATE_CHI2=11.34`,
`SIG_GATE_MEAS=[0.12,0.11,0.27]`, `SIGMA_A_ONLINE=0.10`, `CONFIRM_N=10`, `CONFIRM_M=7`,
`MAX_BOTH_LOST_FRAMES=30`, `MAX_STEREO_NO_ACCEPT_FRAMES=150`, `MAX_REJECT_STREAK=10`

### ⑤ 그리퍼 중심 좌표 변환 — `online/b05_motor_frame.py`

| 코드 내부 이름 | 엑셀 이름 | 내용 |
|---|---|---|
| `Meas_X/Y/Z` | `Meas_X/Y/Z_cm` (Trajectory) | 게이트를 통과한 3D 좌표, 원점 = 모터 중심 |

상수: `MOTOR_CENTER_IN_ID0_CM`(config `setup.motor_center_in_id0_cm`). Candidate_*, KF_* 열도 같은 변환을 거친다.

### 기타 raw 열

| 코드 내부 이름 | 엑셀 이름 | 비고 |
|---|---|---|
| `Frame` | = | 동기화 타임라인 프레임 번호 (1부터) |
| `Time_sec` | `Time_s` | |
| `Sig_*`, `Tracker_Ref_Frame`, `*_Source_Frame`, `*_Sync_*`, `Sync_*`, `*_Frame_Reused` | (저장 안 함) | `RAW_EXCLUDED_COLUMNS`. 동기화 통계는 Summary 시트에 있음 |

---

## 오프라인 후처리 → `3D_final.xlsx`

후처리는 raw 엑셀을 읽자마자 엑셀 이름을 코드 내부 이름으로 되돌린다 (`settings.RAW_COLUMN_RENAME` 하나를 전처리와 같이 씀).

### 입력 읽기 — `postprocess/inputs.py`, `pipeline.step_load`

| 코드 내부 이름 | 내용 |
|---|---|
| `Raw_Meas_X/Y/Z` | raw의 `Meas_*`를 Hampel 전에 복사해 둔 값 (Debug 시트에서도 제외) |

### ⑥ Hampel 이상치 제거 — `b06_hampel.py` `hampel_filter`

| 코드 내부 이름 | 엑셀 | 내용 |
|---|---|---|
| `Meas_X/Y/Z` | — | 이상치 프레임을 NaN으로 바꿈 |
| `Outlier_Flag` | Debug | 이상치 여부 |
| `Missing_Reason` | Debug | 이상치 프레임은 HAMPEL_OUTLIER |

상수: `HAMPEL_K=15`, `HAMPEL_NSIGMA=3.5`, `HAMPEL_ABS_FLOOR=[0.5,0.5,0.3]`

### ⑦ 복원 기준 관측 고정 — `b07_anchor.py` `freeze_anchor_raw_snapshot`

| 코드 내부 이름 | 엑셀 | 내용 |
|---|---|---|
| `Anchor_Raw_X/Y/Z` | Debug | Hampel 이후 두 시점 좌표. 이후 갱신되지 않음 |
| `Anchor_Raw_Valid` | Debug | 기준 관측 유효 여부 |

### ⑧ 과정 잡음 σa 추정 — `b08_process_noise.py` `estimate_process_scale`

| 값 | 엑셀 | 내용 |
|---|---|---|
| `sigma_a` (3축) | Summary `Sigma_a_*_cm_per_frame2` | 연속 두 시점 구간(4프레임 이상) 2차 차분의 1.4826·MAD |

상수: `SIGMA_A_FALLBACK=[0.030,0.030,0.025]`. `build_cv_process_Q`는 ⑬에서 사용.

### ⑨ 복원 모드 분류 · Z 하한 적용 상자 통과 판정 — `b09_recovery_mode.py`

| 코드 내부 이름 | 엑셀 | 내용 |
|---|---|---|
| `Recovery_Mode` | Debug | NONE / TOP_RAY(FRONT_MISSING) / FRONT_RAY(TOP_MISSING) |
| `Recovery_Run_ID` | Plot_Aux = | 같은 모드 연속 구간 번호 |
| `State2` | Debug | DETECTED / SUB / LOST (⑩에서 갱신) |
| `Top_Zone_Supported` | Debug | 상단 굴절 광선이 작업영역 상자(Z 3~25 cm)를 통과 |

상수: `ZONE_X_HALF=25`, `ZONE_Y_HALF=25`, `ZONE_TRACK_CENTER_Z_FLOOR=3.0`, `ZONE_Z_HIGH=25`

### ⑩ 단일시점 구간 좌표 복원 — `b10_reconstruction.py`, `b10_hidden_axis_model.py`

| 코드 내부 이름 | 엑셀 이름 | 시트 | 내용 |
|---|---|---|---|
| `Meas_X/Y/Z` | `X_cm/Y_cm/Z_cm` | **Result** | 복원 프레임에 복원 좌표를 채움 (평활화 전 최종 좌표) |
| `Filled_Type` | → `Coord_Type` | Result | LINEAR_INTERP_* → *_LINEAR, KALMAN_RTS_EPISODE_* → *_KALMAN, KALMAN_*_TAIL_* → *_EXTRAP (변환: ⑭ `_coord_type_labels`) |
| `Hidden_Axis_Posterior_Std` | `Hidden_Axis_Std_cm` | Result | 비관측 축 표준편차 (STEREO/NONE 프레임은 빈칸) |
| `Z_Floor_Applied` | = | Result | 복원 Z 하한(3 cm) 적용 프레임 |
| `Hidden_Stop_Reason` | = | Debug | LOST 사유: NO_RAW_ANCHOR / STATE_INIT_FAILED / POSTERIOR_STD_LIMIT / MAX_EXTRAP_LEN / RAY_GEOMETRY_FAILURE / FRONT_X_SANITY / RAY_OR_SANITY_FAILURE |
| `Recovery_Applied`, `Hidden_Axis_*`, `Anchor_Left/Right_*`, `Hidden_Episode_ID` 등 | = | Debug | 복원 진단값 |

상수: `ANCHOR_SEARCH_INITIAL=30`, `ANCHOR_SEARCH_EXTENDED=60`, `ANCHOR_MODEL_MIN_POINTS=3`, `HIDDEN_EPISODE_LINK_MAX=60`,
`HIDDEN_REACQUIRE_MIN_CONSECUTIVE_RAW=5`, `HIDDEN_HUBER_K=1.345`, `HIDDEN_Q_SCALE_Z=0.3`, `HIDDEN_Q_SCALE_Y=0.03`,
`HIDDEN_STD_STOP_Z_CM=6.0`, `HIDDEN_STD_STOP_Y_CM=2.0`, `V_MAX_Z=0.5`, `V_MAX_Y=0.1`, `V_MAX_X_FRONT_RAY=0.1`,
`FRONT_X_SANITY_MARGIN=2.0`, `MAX_EXTRAP_LEN=150`, `TOP_ZONE_MAX_EXTRAP_LEN=300`, `GRAB_MIN=60`(긴 양안 LOST는 기준 관측 검색 장벽)

### ⑪ 작업영역 판정 · GRAB 표시 — `b11_zone_grab.py`

| 코드 내부 이름 | 엑셀 이름 | 시트 | 내용 |
|---|---|---|---|
| `Coord_State` | → `State` | Result | DETECTED(_ZONE) → DETECTED, SUB(_ZONE) → RECOVERED, LOST, GRAB (`RESULT_STATE_MAP`) |
| `In_Zone` | (Result `In_Zone`의 근거) | Debug | 평활화 전 좌표의 작업영역 판정 |
| `Grab_*` | = | Debug | GRAB 판정 근거 프레임 |

상수: `ZONE_Z_LOW=0.0`, `ZONE_Z_HIGH=25.0`, `GRAB_MIN=60`

### ⑫ 표시용 작업영역 연결 — `b12_display_zone.py`

| 코드 내부 이름 | 엑셀 이름 | 시트 | 내용 |
|---|---|---|---|
| `In_Zone_PreRTS` | → `In_Zone` | Result | ⑪ `In_Zone`의 복사본 (연결 전) |
| `Display_In_Zone` | = | Plot_Aux | 그래프 음영용 (짧은 공백 연결) |
| `Display_Zone_*` | = | Debug | 연결 진단값 |

상수: `DISPLAY_ZONE_BRIDGE_MAX=7`, `DISPLAY_ZONE_STABLE_SEARCH=30`, `DISPLAY_ZONE_STABLE_MIN_POINTS_PER_SIDE=3`

### ⑬ 표시용 3차원 RTS 평활화 — `b13_rts.py`

| 코드 내부 이름 | 엑셀 이름 | 시트 | 내용 |
|---|---|---|---|
| `X_rts/Y_rts/Z_rts` | = | Debug | RTS 결과 |
| **`X_cm/Y_cm/Z_cm`** | **`X_vis_cm/Y_vis_cm/Z_vis_cm`** | Plot_Aux | `assemble()` 결과. 관측 구간 RTS, 복원 구간 복원값 |
| `RTS_Std_X/Y/Z` | `Vis_Std_X/Y/Z_cm` | Plot_Aux | |
| `In_Zone_Final` | = | Debug | RTS 좌표 기준 작업영역 판정 |
| `RTS_*`, `Assigned_Sigma_*` | = | Debug | |

상수: `SIG_RTS_RAW=[0.12,0.11,0.27]`, `SIG_RTS_RECON_TOP/FRONT=[0.10,0.10,0.30]`

### ⑭ 프레임별 속도 산출 — `b14_velocity.py`

| 코드 내부 이름 | 엑셀 이름 | 시트 |
|---|---|---|
| `Vx/Vy/Vz_cm_s`, `Speed_3D_cm_s` | = | Result |
| `Vx/Vy/Vz_Source`, `Speed_3D_Source` | = | Result |
| `Vx/Vy/Vz_obs_cm_s`, `Speed_3D_obs_cm_s` | = | Result |
| `V_Noise_Ratio`, `V_obs_Noise_Ratio` | = | Result |
| `Result_State`, `Result_Coord_Type` | `State`, `Coord_Type` | Result |
| `V_N_Used`, `V_obs_N_Used` | = | Debug |

상수: `VEL_HALF_WINDOW=5`, `VEL_MAX_NOISE_RATIO=2.0`, `VEL_MAX_CENTER_OFFSET_FRAMES=2.0`

---

## 이름 주의 (다음 작업: 코드 내부 이름 ↔ 엑셀 이름 통일)

현재 코드는 원본 결과와 완전히 같게 유지하려고 내부 이름을 바꾸지 않았다. 혼동하기 쉬운 항목은 다음과 같다.

| 코드 내부 | 엑셀 | 문제 |
|---|---|---|
| `Meas_X` (후처리 ⑩ 이후) | Result `X_cm` | 같은 이름 `Meas_X`가 단계마다 뜻이 바뀜: raw 두 시점 좌표 → Hampel 후 → 복원값 포함 |
| `X_cm` (⑬ `assemble`) | Plot_Aux `X_vis_cm` | **코드의 `X_cm`은 RTS 표시용, 엑셀 Result의 `X_cm`은 평활화 전 좌표** |
| `In_Zone`, `In_Zone_PreRTS`, `In_Zone_Final`, `Display_In_Zone` | Result `In_Zone` = `In_Zone_PreRTS` | 작업영역 판정 열이 4개 |
| `State2` → `Coord_State` → `Result_State` | Result `State` | 상태 열이 단계별로 3개 |
| `Filled_Type` → `Result_Coord_Type` | Result `Coord_Type` | |
| `Hidden_Axis_Posterior_Std` | Result `Hidden_Axis_Std_cm` | |
| `Status_Top`, `Top_px_u`, `Top_conf`, `Time_sec` | `Top_Status`, `Top_U_px`, `Top_Conf`, `Time_s` | raw 읽기/쓰기에서만 변환 |
| `Online_X`, `KF_PosStd_X`, `Ray_s_Top_cm` | `KF_X_cm`, `KF_Std_X_cm`, `Top_Ray_Dist_cm` | |

통일 작업 시 Result · Plot_Aux · Summary · Configuration · Legend 시트는 그대로 유지되어야 하므로
`tests/test_postprocess_regression.py`로 확인할 수 있다. Debug 시트는 열 이름이 바뀌는 것이 정상이다.
