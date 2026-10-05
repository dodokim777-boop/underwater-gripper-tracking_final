# -*- coding: utf-8 -*-
"""
출력: 3D_raw_online.xlsx (Trajectory / Triangulation / Online_Filter / Summary / Configuration)

계산 결과(df)는 그대로 두고 엑셀에 쓸 때만 시트를 나누고 열 이름을 바꾼다.
열 이름표는 settings.RAW_COLUMN_RENAME 하나를 후처리와 같이 쓴다.
"""

import numpy as np
import pandas as pd

from .. import settings
from ..io_utils import format_workbook_sheets


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else np.nan


def write_raw_excel(df, summary_df, config_df, path):
    listed = {c for cols in settings.RAW_SHEET_COLUMNS.values() for c in cols}
    extra = [c for c in df.columns if c not in listed and c not in settings.RAW_EXCLUDED_COLUMNS]
    if extra:
        print(f'NOTE: 시트 목록에 없는 열을 Trajectory 끝에 붙임: {extra}')
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet, cols in settings.RAW_SHEET_COLUMNS.items():
            cols = [c for c in cols if c in df.columns]
            if sheet == 'Trajectory':
                cols = cols + extra
            df[cols].rename(columns=settings.RAW_COLUMN_RENAME).to_excel(writer, sheet_name=sheet, index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
        format_workbook_sheets(writer.book)


def build_summary_df(df, sync_output_fps):
    """Summary 시트: 검출 · 짝짓기 · 삼각측량 · 게이트 통과 비율, Missing_Reason 집계."""
    frames_n = len(df)
    top_detected_n = int(df['Status_Top'].eq('DETECTED').sum()) if frames_n else 0
    front_detected_n = int(df['Status_Front'].eq('DETECTED').sum()) if frames_n else 0
    paired_n = int(
        (df['Status_Top'].eq('DETECTED') & df['Status_Front'].eq('DETECTED')).sum()
    ) if frames_n else 0
    on_valid_n = int(df['Geometry_Status_ON'].eq('VALID').sum()) if frames_n else 0
    off_valid_n = int(df['Geometry_Status_OFF'].eq('VALID').sum()) if frames_n else 0
    common_valid_n = int(
        (df['Geometry_Status_ON'].eq('VALID') & df['Geometry_Status_OFF'].eq('VALID')).sum()
    ) if frames_n else 0
    raw_accepted_n = int(df['Meas_Status'].eq('TRIANGULATED').sum()) if frames_n else 0
    sync_ok_n = int(df['Sync_Status'].eq('SYNC_OK').sum()) if frames_n else 0
    top_reused_n = int(df['Top_Frame_Reused'].fillna(False).astype(bool).sum()) if frames_n else 0
    front_reused_n = int(df['Front_Frame_Reused'].fillna(False).astype(bool).sum()) if frames_n else 0
    sync_abs_error_ms = (
        pd.to_numeric(df['Sync_Abs_Error_ms'], errors='coerce').dropna().to_numpy(dtype=float)
        if frames_n else np.array([], dtype=float)
    )

    summary_rows = [
        ('Frames on synchronized timeline', frames_n),
        ('Output timeline FPS', sync_output_fps),
        ('Sync-valid frames', sync_ok_n),
        ('Sync-valid ratio', _safe_ratio(sync_ok_n, frames_n)),
        ('Rows reusing top source frame', top_reused_n),
        ('Rows reusing front source frame', front_reused_n),
        ('Top detected frames', top_detected_n),
        ('Front detected frames', front_detected_n),
        ('Paired detected frames', paired_n),
        ('Refraction ON geometry-valid frames', on_valid_n),
        ('Refraction OFF geometry-valid frames', off_valid_n),
        ('ON/OFF common-valid frames', common_valid_n),
        ('Raw triangulated measurements', raw_accepted_n),
        ('Paired detected ratio', _safe_ratio(paired_n, frames_n)),
        ('Refraction ON valid / paired', _safe_ratio(on_valid_n, paired_n)),
        ('Refraction OFF valid / paired', _safe_ratio(off_valid_n, paired_n)),
        ('ON/OFF common-valid / paired', _safe_ratio(common_valid_n, paired_n)),
        ('Gate-accepted raw / paired', _safe_ratio(raw_accepted_n, paired_n)),
    ]
    if frames_n:
        for status, count in df['Sync_Status'].value_counts(dropna=False).items():
            summary_rows.append((f'Sync_Status::{status}', int(count)))
        if sync_abs_error_ms.size:
            summary_rows.extend([
                ('Sync absolute error median [ms]', float(np.median(sync_abs_error_ms))),
                ('Sync absolute error P95 [ms]', float(np.percentile(sync_abs_error_ms, 95))),
                ('Sync absolute error max [ms]', float(np.max(sync_abs_error_ms))),
            ])
        for reason, count in df['Missing_Reason'].value_counts(dropna=False).items():
            summary_rows.append((f'Missing_Reason::{reason}', int(count)))
        accepted_gap = pd.to_numeric(
            df.loc[df['Meas_Status'].eq('TRIANGULATED'), 'Ray_Gap_cm'], errors='coerce'
        ).dropna().to_numpy(dtype=float)
        if accepted_gap.size:
            summary_rows.extend([
                ('Accepted ray gap median [cm]', float(np.median(accepted_gap))),
                ('Accepted ray gap P95 [cm]', float(np.percentile(accepted_gap, 95))),
                ('Accepted ray gap max [cm]', float(np.max(accepted_gap))),
            ])
    return pd.DataFrame(summary_rows, columns=['Metric', 'Value'])


def build_config_df(dataset_is_wet, enable_refraction, video, tracker_3d,
                    cls_top, cls_front, calib_top, calib_front):
    """Configuration 시트: 이번 실행에 쓴 설정값 기록."""
    return pd.DataFrame([
        ('Dataset is wet', dataset_is_wet),
        ('Tracking refraction enabled', enable_refraction),
        ('ON/OFF ablation logged simultaneously', True),
        ('Expected resolution', f'{settings.EXPECTED_RESOLUTION[0]}x{settings.EXPECTED_RESOLUTION[1]}'),
        ('Tracker reference FPS', settings.EXPECTED_FPS),
        ('Synchronization method', settings.SYNC_METHOD),
        ('Output timeline FPS', video['sync_output_fps']),
        ('Synchronization tolerance [ms]', video['sync_max_error_sec'] * 1000.0),
        ('Top time offset [s]', settings.TOP_TIME_OFFSET_SEC),
        ('Front time offset [s]', settings.FRONT_TIME_OFFSET_SEC),
        ('Common sync start [s]', video['overlap_start_sync_sec']),
        ('Common sync end [s]', video['overlap_end_sync_sec']),
        ('Common output frames planned', video['sync_output_frames']),
        ('Top input FPS', video['fps_top']),
        ('Front input FPS', video['fps_front']),
        ('Top metadata frame count', video['frame_total_top']),
        ('Front metadata frame count', video['frame_total_front']),
        ('Top nominal duration [s]', video['duration_top_sec']),
        ('Front nominal duration [s]', video['duration_front_sec']),
        ('Duplicate top inference cached', True),
        ('Duplicate front inference cached', True),
        ('Confidence threshold', settings.CONF_CUT),
        ('YOLO imgsz', settings.YOLO_IMGSZ),
        ('Top weights path', settings.TOP_MODEL_PATH),
        ('Front weights path', settings.FRONT_MODEL_PATH),
        ('SIG_GATE_MEAS_X_cm', settings.SIG_GATE_MEAS[0]),
        ('SIG_GATE_MEAS_Y_cm', settings.SIG_GATE_MEAS[1]),
        ('SIG_GATE_MEAS_Z_cm', settings.SIG_GATE_MEAS[2]),
        ('SIGMA_A_online_cm_per_frame2', settings.SIGMA_A_ONLINE),
        ('Jump limit [cm/frame]', tracker_3d.jump_limit_cm_per_frame),
        ('Jump absolute cap [cm]', tracker_3d.jump_limit_abs_cm),
        ('Mahalanobis chi2 threshold', settings.GATE_CHI2),
        ('Ray gap max [cm]', settings.RAY_GAP_MAX_CM),
        ('Target class names', ', '.join(settings.TARGET_CLASS_NAMES)),
        ('Top target class id', cls_top),
        ('Front target class id', cls_front),
        ('Confirm N stereo-candidate attempts', settings.CONFIRM_N),
        ('Confirm M accepted stereo candidates', settings.CONFIRM_M),
        ('Confirmation basis', 'candidate attempts only; camera-missing frames do not enter history'),
        ('Max both-camera-missing frames', settings.MAX_BOTH_LOST_FRAMES),
        ('Max stereo-observable no-accept frames', settings.MAX_STEREO_NO_ACCEPT_FRAMES),
        ('Single-view 3D-gap reset', 'disabled while at least one camera remains DETECTED'),
        ('Max reject streak', settings.MAX_REJECT_STREAK),
        ('Top camera model', calib_top.get('camera_model', settings.TOP_CAMERA_MODEL)),
        ('Top intrinsic source', calib_top.get('intrinsic_source', 'hardcoded')),
        ('Front camera model', calib_front.get('camera_model', settings.FRONT_CAMERA_MODEL)),
        ('Front intrinsic source', calib_front.get('intrinsic_source', 'hardcoded')),
    ], columns=['Parameter', 'Value'])
