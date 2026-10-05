# -*- coding: utf-8 -*-
"""
오프라인 후처리 실행 (다이어그램 오른쪽 영역 ⑥~⑭ → 3D_final.xlsx)

블록마다 step 함수 하나를 둔다. notebooks/03은 step을 한 셀에 하나씩 실행하고,
run_postprocess()는 같은 step을 순서대로 모두 실행한다 (두 경로의 계산은 같다).
각 단계 사이에서 불변식 검사(validate_*)를 수행한다.

  load   : 3D_raw_online.xlsx + 보정 NPZ 읽기
  ⑥ Hampel 이상치 제거
  ⑦ 복원 기준 관측 고정 (Anchor_Raw_*)
  ⑧ 과정 잡음 σa 추정
  ⑨ 복원 모드 분류 · Z 하한 적용 상자 통과 판정
  ⑩ 단일시점 구간 좌표 복원
  ⑪ 작업영역 판정 · GRAB 표시
  ⑫ 표시용 작업영역 연결 (Plot_Aux)
  ⑬ 표시용 3차원 RTS 평활화 (Plot_Aux)
  ⑭ 프레임별 속도 산출
  save   : 3D_final.xlsx 저장

핵심 원칙 (원본 postprocess_final 머리말 요약)
  - Hampel 직후 실제 stereo 3D만 Anchor_Raw_* 불변 스냅숏으로 고정한다. 복원 좌표, 표시용 연결,
    RTS 결과는 어떤 복원 모델의 관측/기준 관측으로도 재사용하지 않는다.
  - FRONT_MISSING은 Z를 1차원으로 복원한 뒤 상단 광선으로 X, Y를 계산한다 (TOP_RAY).
    TOP_MISSING은 Y를 1차원으로 복원한 뒤 정면 광선으로 X, Z를 계산한다 (FRONT_RAY).
  - REJECTED_*, TENTATIVE_HOLD, BOTH_MISSING은 좌표 출력 대상으로 임의 복원하지 않는다.
  - Result 좌표는 평활화 전 좌표이며, RTS 결과는 Plot_Aux(그래프용)에만 쓴다.
"""

from .. import settings
from . import inputs, final_excel
from .b06_hampel import hampel_filter
from .b07_anchor import freeze_anchor_raw_snapshot
from .b08_process_noise import estimate_process_scale
from .b09_recovery_mode import classify_states, classify_top_zone_support
from .b10_reconstruction import interpolate_missing, validate_reconstruction_invariants
from .b11_zone_grab import classify_zone_grab, validate_grab_invariants
from .b12_display_zone import build_display_zone_state, validate_display_zone_invariants
from .b13_rts import rts_smooth, assemble, validate_post_rts_invariants
from .b14_velocity import compute_velocity_columns


class PostContext(dict):
    """단계 사이에서 넘기는 값 묶음. ctx['df']가 프레임별 표(코드 내부 열 이름)다."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def step_load(raw_path=None, calib_top='auto', calib_front='auto'):
    if raw_path is None:
        raw_path = settings.RAW_EXCEL
    # [2026-09] 새/기존 raw 형식 모두 읽고 코드 내부 열 이름으로 되돌린다.
    df, raw_input_columns = inputs.read_raw_trajectory(raw_path)
    df = inputs.ensure_missing_reason(df)

    missing_required = [
        c for c in [
            'Frame', *settings.MEAS_COLS,
            'Status_Top', 'Status_Front',
            'Top_px_u', 'Top_px_v', 'Front_px_u', 'Front_px_v',
        ]
        if c not in df.columns
    ]
    if missing_required:
        raise KeyError(f'Missing required columns: {missing_required}')

    for raw_col, meas_col in zip(settings.RAW_MEAS_COLS, settings.MEAS_COLS):
        df[raw_col] = df[meas_col]

    if calib_top == 'auto':
        calib_top = inputs.load_calib_top()
    if calib_front == 'auto':
        calib_front = inputs.load_calib_front()
    if calib_top is None or calib_front is None:
        raise RuntimeError(
            'Both Top and Front dry-calibration are mandatory for bidirectional reconstruction.'
        )

    if not (
        settings.SIG_RTS_RECON_TOP_FINALIZED
        and settings.SIG_RTS_RECON_FRONT_FINALIZED
        and settings.HIDDEN_STD_STOP_THRESHOLDS_FINALIZED
    ):
        print(
            'WARNING: finalized flag mismatch. '
            'Check TOP_RAY/FRONT_RAY residual scales and hidden-axis std-stop thresholds '
            'before reporting final RTS uncertainty.'
        )
    return PostContext(df=df, raw_input_columns=raw_input_columns,
                       calib_top=calib_top, calib_front=calib_front)


def step06_hampel(ctx):
    ctx['df'], ctx['n_outliers'] = hampel_filter(ctx['df'])
    return ctx


def step07_anchor(ctx):
    # 복원 전에 Hampel-filtered raw stereo를 immutable anchor snapshot으로 고정한다.
    ctx['df'] = freeze_anchor_raw_snapshot(ctx['df'])
    return ctx


def step08_process_noise(ctx):
    sigma_a, n_q_runs, q_used_fallback = estimate_process_scale(ctx['df'])
    print(f'Q sigma_a={sigma_a}, valid runs={n_q_runs}, fallback={q_used_fallback}')
    ctx.update(sigma_a=sigma_a, n_q_runs=n_q_runs, q_used_fallback=q_used_fallback)
    return ctx


def step09_recovery_mode(ctx):
    df = classify_states(ctx['df'])
    ctx['df'] = classify_top_zone_support(df, ctx['calib_top'])
    return ctx


def step10_reconstruction(ctx):
    df, reconstruction_stats = interpolate_missing(
        ctx['df'], ctx['calib_top'], ctx['calib_front'], sigma_a=ctx['sigma_a']
    )
    ctx['df'] = validate_reconstruction_invariants(df)
    ctx['reconstruction_stats'] = reconstruction_stats
    return ctx


def step11_zone_grab(ctx):
    df, n_grab = classify_zone_grab(ctx['df'])
    ctx['df'] = validate_grab_invariants(df)
    ctx['n_grab'] = n_grab
    return ctx


def step12_display_zone(ctx):
    # 표시용 zone bridge는 pre-RTS 판정을 immutable snapshot으로 한 번만 훑는다.
    # 좌표/GRAB/RTS 입력에는 반영하지 않는다.
    df = build_display_zone_state(
        ctx['df'],
        max_gap_frames=settings.DISPLAY_ZONE_BRIDGE_MAX,
        stable_search_frames=settings.DISPLAY_ZONE_STABLE_SEARCH,
        stable_min_points=settings.DISPLAY_ZONE_STABLE_MIN_POINTS_PER_SIDE,
    )
    ctx['df'] = validate_display_zone_invariants(
        df,
        max_gap_frames=settings.DISPLAY_ZONE_BRIDGE_MAX,
        stable_search_frames=settings.DISPLAY_ZONE_STABLE_SEARCH,
        stable_min_points=settings.DISPLAY_ZONE_STABLE_MIN_POINTS_PER_SIDE,
    )
    return ctx


def step13_rts(ctx):
    df = rts_smooth(ctx['df'], sigma_a=ctx['sigma_a'])
    df = assemble(df)
    ctx['df'] = validate_post_rts_invariants(df)
    return ctx


def step14_velocity(ctx):
    # [2026-09] 평활화 전 좌표 기준 프레임별 속도 (계산 결과 열만 추가)
    ctx['df'] = compute_velocity_columns(ctx['df'])
    return ctx


def step_save(ctx, out_path=None):
    if out_path is None:
        out_path = settings.FINAL_EXCEL
    df = ctx['df']
    summary_df = final_excel.build_summary_df(
        df, ctx['n_outliers'], ctx['n_q_runs'], ctx['q_used_fallback'], ctx['sigma_a'],
        ctx['reconstruction_stats'], ctx['n_grab'],
    )
    config_df = final_excel.build_config_df(ctx['calib_front'])
    # [2026-09] 시트 구성: Result / Legend / Plot_Aux / Summary / Configuration / Debug
    sheets = final_excel.build_final_sheets(
        df, ctx['raw_input_columns'], final_excel.PREFERRED_DEBUG_COLUMNS, summary_df, config_df
    )
    final_excel.write_final_excel(sheets, out_path)
    print(f'Saved: {out_path}')
    ctx['sheets'] = sheets
    return ctx


STEPS = [
    step06_hampel, step07_anchor, step08_process_noise, step09_recovery_mode,
    step10_reconstruction, step11_zone_grab, step12_display_zone, step13_rts,
    step14_velocity,
]


def run_postprocess(raw_path=None, out_path=None, calib_top='auto', calib_front='auto'):
    """⑥~⑭를 순서대로 실행하고 3D_final.xlsx를 저장한다. 반환: 프레임별 df (코드 내부 열 이름)"""
    ctx = step_load(raw_path, calib_top, calib_front)
    for step in STEPS:
        ctx = step(ctx)
    ctx = step_save(ctx, out_path)
    return ctx['df']
