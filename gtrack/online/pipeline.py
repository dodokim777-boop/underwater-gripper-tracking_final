# -*- coding: utf-8 -*-
"""
온라인 전처리 실행 (다이어그램 왼쪽 영역 ①~⑤ → 3D_raw_online.xlsx)

프레임마다 다음 순서로 처리한다.
  ① 물체 검출 (b01_detection)            → 상단·정면 중심 픽셀
  ◇ 두 시점 검출 성립? (b01.pairing_reason) → 아니면 TOP/FRONT/BOTH_MISSING으로 ④에 전달
  ② 광선 생성 · 평면 경계 굴절 (b02_rays)
  ③ 두 광선 최근접점 삼각측량 (b03_triangulation) → 굴절 ON/OFF 후보
  ④ 후보 수용 판정 및 궤적 확정 (b04_tracker)   → 광선 간 거리 → 변위 → Mahalanobis → 7-of-10
  ⑤ 그리퍼 중심 좌표 변환 (b05_motor_frame)     → Meas_X/Y/Z
설계 원칙
  [전처리] 검출, 굴절/핀홀 삼각측량, 물리·통계 게이팅, 상태 기록만 수행한다.
  [후처리] 이상치 제거, 결측 복원, RTS 평활화는 gtrack/postprocess에서 수행한다.
  Meas 열에는 게이트를 통과한 원시 3D 후보만 기록하며 필터 예측값으로 결측을 채우지 않는다.
"""

import cv2
import numpy as np
import pandas as pd

from .. import settings
from . import video_sync, calibration_io, visualize, raw_excel
from .b01_detection import load_models, select_yolo_device, CachedDetector, pairing_reason
from .b03_triangulation import triangulate_on_off
from .b04_tracker import OnlineTracker, ray_gap_gate
from .b05_motor_frame import to_motor_origin_frame


def run_online(max_frames=None):
    """settings(config.yaml 적용 후)에 지정된 영상으로 전처리를 실행하고 raw 엑셀을 저장한다.

    max_frames: 처음 N 프레임만 처리 (설정 확인용 짧은 실행). None이면 전체.
    반환: 프레임별 결과 DataFrame (코드 내부 열 이름)
    """
    dataset_is_wet = settings.DATASET_IS_WET
    enable_refraction = settings.ENABLE_REFRACTION

    # ---------------- 입력: YOLO 가중치 ----------------
    model_top, model_front, cls_top, cls_front = load_models()
    yolo_device = select_yolo_device()

    # ---------------- 입력: 프레임 정렬된 두 영상 ----------------
    video = video_sync.open_videos()
    cap_top, cap_front = video['cap_top'], video['cap_front']
    sync_output_fps = video['sync_output_fps']
    sync_max_error_sec = video['sync_max_error_sec']

    print("🎬 3D 삼각측량 시작...")
    if not settings.HEADLESS:
        dataset_answer = input("데이터 환경 → [w] wet / [d] dry [기본 w]: ").strip().lower()
        dataset_is_wet = (dataset_answer != 'd')
        correction_answer = input("굴절 보정 → [o] ON / [x] OFF [기본 o]: ").strip().lower()
        enable_refraction = (correction_answer != 'x')
    print(
        f"   └─ dataset={'WET' if dataset_is_wet else 'DRY'}, "
        f"refraction={'ON' if enable_refraction else 'OFF'}"
    )

    # ---------------- 입력: 보정값 (NPZ R,t + K,D) ----------------
    calib_top, calib_front, front_use_wall_transform = calibration_io.load_calibrations(dataset_is_wet)

    print(
        f"jump gate = min({settings.JUMP_LIMIT_CM_PER_FRAME:.1f} cm/frame × Δk, "
        f"{settings.JUMP_LIMIT_ABS_CM:.1f} cm)"
    )
    tracker_3d = OnlineTracker(
        jump_limit_cm_per_frame=settings.JUMP_LIMIT_CM_PER_FRAME,
        jump_limit_abs_cm=settings.JUMP_LIMIT_ABS_CM,
    )
    data_records = []
    frame_count = 0
    verified_unification = False

    need_vis = (not settings.HEADLESS) or settings.SAVE_ANNOTATED_VIDEO
    video_writer = (
        visualize.AnnotatedVideoWriter(settings.ANNOTATED_VIDEO_PATH, sync_output_fps)
        if settings.SAVE_ANNOTATED_VIDEO else None
    )

    sampler_top = video_sync.SequentialNearestSampler(
        cap_top, video['fps_top'], video['frame_total_top'],
        time_offset_sec=settings.TOP_TIME_OFFSET_SEC
    )
    sampler_front = video_sync.SequentialNearestSampler(
        cap_front, video['fps_front'], video['frame_total_front'],
        time_offset_sec=settings.FRONT_TIME_OFFSET_SEC
    )

    predict_kwargs = dict(
        verbose=False, conf=settings.CONF_CUT, imgsz=settings.YOLO_IMGSZ, device=yolo_device
    )
    detector_top = CachedDetector(model_top, cls_top, predict_kwargs)
    detector_front = CachedDetector(model_front, cls_front, predict_kwargs)

    n_frames = video['sync_output_frames']
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))
    progress_every = max(int(round(sync_output_fps * 10)), 1)

    for output_index in range(n_frames):
        target_sync_time_sec = (
            video['overlap_start_sync_sec'] + output_index / sync_output_fps
        )
        sampled_top = sampler_top.nearest(target_sync_time_sec)
        sampled_front = sampler_front.nearest(target_sync_time_sec)
        if sampled_top is None or sampled_front is None:
            print(
                f'⚠️ 동기 프레임 읽기 종료: output={output_index + 1}, '
                f'top={sampled_top is not None}, front={sampled_front is not None}'
            )
            break

        frame_count = output_index + 1
        output_time_sec = output_index / sync_output_fps
        tracker_ref_frame = 1.0 + output_time_sec * settings.EXPECTED_FPS
        if frame_count % progress_every == 0:
            print(f'   ... {frame_count}/{n_frames} frames')

        frame_top = sampled_top.frame
        frame_front = sampled_front.frame
        top_source_frame = sampled_top.source_frame
        front_source_frame = sampled_front.source_frame
        top_time_sec = sampled_top.video_time_sec
        front_time_sec = sampled_front.video_time_sec
        top_sync_time_sec = sampled_top.sync_time_sec
        front_sync_time_sec = sampled_front.sync_time_sec
        sync_error_sec = float(front_sync_time_sec - top_sync_time_sec)
        sync_error_ms = sync_error_sec * 1000.0
        sync_ok = bool(abs(sync_error_sec) <= sync_max_error_sec)
        sync_status = 'SYNC_OK' if sync_ok else 'SYNC_OUT_OF_TOLERANCE'

        # dry 진단 모드에서만 영상 기반 fallback을 허용한다(하드코딩 intrinsic 사용).
        if not dataset_is_wet and calib_top is None:
            pose = calibration_io.detect_marker_pose(
                frame_top, settings.K_MATRIX_TOP, settings.DIST_COEFFS_TOP, select='first'
            )
            if pose is not None:
                calib_top = calibration_io.build_world_calib(
                    *pose, settings.K_MATRIX_TOP, settings.DIST_COEFFS_TOP
                )
        if not dataset_is_wet and calib_front is None:
            pose = calibration_io.detect_marker_pose(
                frame_front, settings.K_MATRIX_FRONT, settings.DIST_COEFFS_FRONT, select='right'
            )
            if pose is not None:
                calib_front = calibration_io.build_world_calib(
                    *pose, settings.K_MATRIX_FRONT, settings.DIST_COEFFS_FRONT
                )

        if calib_front is not None and settings.VERIFY_UNIFICATION and not verified_unification:
            calibration_io.verify_unification(calib_front, frame_front)
            verified_unification = True

        if calib_top is None or calib_front is None:
            continue

        # ============ ① 물체 검출 (YOLO) ============
        res_top, pt_top, conf_top = detector_top.detect(frame_top, top_source_frame)
        res_front, pt_front, conf_front = detector_front.detect(frame_front, front_source_frame)

        stat_top = 'DETECTED' if pt_top is not None else 'LOST'
        stat_front = 'DETECTED' if pt_front is not None else 'LOST'

        candidate_3d = None
        tri_on = None
        tri_off = None
        geometry_status_on = 'NOT_PAIRED'
        geometry_status_off = 'NOT_PAIRED'

        # ============ ◇ 두 시점 검출 성립? ============
        observation_reason = pairing_reason(sync_ok, pt_top, pt_front)
        if observation_reason is None:
            # ============ ② 광선·굴절 + ③ 최근접점 삼각측량 (굴절 ON/OFF 동시) ============
            tri_on, tri_off, geometry_status_on, geometry_status_off = triangulate_on_off(
                pt_top, pt_front, calib_top, calib_front
            )
            selected_tri = tri_on if enable_refraction else tri_off
            # ============ ④-1 광선 간 거리 게이트 ============
            candidate_3d, observation_reason = ray_gap_gate(selected_tri)

        selected_tri = tri_on if enable_refraction else tri_off
        ray_gap_cm = np.nan if selected_tri is None else selected_tri.ray_gap_cm
        ray_s_top_cm = np.nan if selected_tri is None else selected_tri.s_top_cm
        ray_s_front_cm = np.nan if selected_tri is None else selected_tri.s_front_cm

        # ============ ④-2 변위 → Mahalanobis → 7-of-10 확정 ============
        (
            meas_w,
            online_w,
            kf_std_w,
            meas_status,
            track_state,
            missing_reason,
            kf_updated,
        ) = tracker_3d.step(
            frame_idx=tracker_ref_frame,
            top_detected=(pt_top is not None),
            front_detected=(pt_front is not None),
            candidate_3d=candidate_3d,
            observation_reason=observation_reason,
        )

        # ============ ⑤ 그리퍼 중심 좌표 변환 ============
        meas_m = to_motor_origin_frame(meas_w)
        online_m = to_motor_origin_frame(online_w)
        candidate_m = to_motor_origin_frame(candidate_3d)
        candidate_on_m = to_motor_origin_frame(
            None if tri_on is None else tri_on.point_world
        )
        candidate_off_m = to_motor_origin_frame(
            None if tri_off is None else tri_off.point_world
        )
        sig_rec = (
            settings.SIG_GATE_MEAS
            if np.all(np.isfinite(meas_m))
            else np.full(3, np.nan)
        )

        data_records.append({
            'Frame': frame_count,
            'Time_sec': output_time_sec,
            'Tracker_Ref_Frame': tracker_ref_frame,
            'Top_Source_Frame': top_source_frame,
            'Front_Source_Frame': front_source_frame,
            'Top_Time_sec': top_time_sec,
            'Front_Time_sec': front_time_sec,
            'Top_Sync_Time_sec': top_sync_time_sec,
            'Front_Sync_Time_sec': front_sync_time_sec,
            'Sync_Error_ms': sync_error_ms,
            'Sync_Abs_Error_ms': abs(sync_error_ms),
            'Sync_Status': sync_status,
            'Top_Frame_Reused': sampled_top.reused,
            'Front_Frame_Reused': sampled_front.reused,
            'Status_Top': stat_top,
            'Status_Front': stat_front,
            'Top_conf': conf_top,
            'Front_conf': conf_front,
            'Top_px_u': pt_top[0] if pt_top is not None else np.nan,
            'Top_px_v': pt_top[1] if pt_top is not None else np.nan,
            'Front_px_u': pt_front[0] if pt_front is not None else np.nan,
            'Front_px_v': pt_front[1] if pt_front is not None else np.nan,
            'Candidate_X': candidate_m[0],
            'Candidate_Y': candidate_m[1],
            'Candidate_Z': candidate_m[2],
            'Candidate_ON_X': candidate_on_m[0],
            'Candidate_ON_Y': candidate_on_m[1],
            'Candidate_ON_Z': candidate_on_m[2],
            'Candidate_OFF_X': candidate_off_m[0],
            'Candidate_OFF_Y': candidate_off_m[1],
            'Candidate_OFF_Z': candidate_off_m[2],
            'Geometry_Status_ON': geometry_status_on,
            'Geometry_Status_OFF': geometry_status_off,
            'Meas_X': meas_m[0],
            'Meas_Y': meas_m[1],
            'Meas_Z': meas_m[2],
            'Sig_X': sig_rec[0],
            'Sig_Y': sig_rec[1],
            'Sig_Z': sig_rec[2],
            'Track_State': track_state,
            'Meas_Status': meas_status,
            'Missing_Reason': missing_reason,
            'KF_Updated': kf_updated,
            'Online_X': online_m[0],
            'Online_Y': online_m[1],
            'Online_Z': online_m[2],
            'KF_PosStd_X': kf_std_w[0],
            'KF_PosStd_Y': kf_std_w[1],
            'KF_PosStd_Z': kf_std_w[2],
            'Ray_Gap_cm': ray_gap_cm,
            'Ray_s_Top_cm': ray_s_top_cm,
            'Ray_s_Front_cm': ray_s_front_cm,
            'Ray_Gap_ON_cm': np.nan if tri_on is None else tri_on.ray_gap_cm,
            'Ray_Gap_OFF_cm': np.nan if tri_off is None else tri_off.ray_gap_cm,
        })

        # ============ 주석 영상 (선택, 시각화 전용) ============
        ann_top = ann_front = None
        if need_vis:
            ann_top, ann_front = visualize.annotate_pair(
                res_top, res_front, online_w, calib_top, calib_front,
                dataset_is_wet, enable_refraction, sync_output_fps,
                frame_count, top_source_frame, top_time_sec, sampled_top.reused,
                front_source_frame, front_time_sec, sampled_front.reused,
                sync_status, sync_error_ms,
            )

        if video_writer is not None and ann_top is not None and ann_front is not None:
            video_writer.write(ann_top, ann_front)

        if not settings.HEADLESS and ann_top is not None and ann_front is not None:
            cv2.imshow('Top', cv2.resize(ann_top, (0, 0), fx=0.5, fy=0.5))
            cv2.imshow('Front', cv2.resize(ann_front, (0, 0), fx=0.5, fy=0.5))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap_top.release()
    cap_front.release()
    if video_writer is not None:
        video_writer.release()
    if not settings.HEADLESS:
        cv2.destroyAllWindows()

    # ---------------- 출력: 3D_raw_online.xlsx ----------------
    print("영상 종료. 전처리 산출물 저장 중...")
    df = pd.DataFrame(data_records)
    summary_df = raw_excel.build_summary_df(df, sync_output_fps)
    config_df = raw_excel.build_config_df(
        dataset_is_wet, enable_refraction, video, tracker_3d,
        cls_top, cls_front, calib_top, calib_front,
    )
    raw_excel.write_raw_excel(df, summary_df, config_df, settings.RAW_EXCEL)

    print(f"저장: {settings.RAW_EXCEL} ({len(df)} frames)")
    if len(df):
        print(df['Missing_Reason'].value_counts(dropna=False).to_string())
    return df
