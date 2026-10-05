"""Run steps ⑥-⑭ (one function per step) and save 3D_final.xlsx."""

from .. import settings
from . import load_raw_excel, save_final_excel
from .step06_remove_outliers import hampel_filter
from .step07_freeze_anchors import freeze_anchor_raw_snapshot
from .step08_estimate_noise import estimate_process_scale
from .step09_classify_recovery import classify_states, classify_top_zone_support
from .step10_recover_missing import interpolate_missing, validate_reconstruction_invariants
from .step11_zone_and_grab import classify_zone_grab, validate_grab_invariants
from .step12_display_zone import build_display_zone_state, validate_display_zone_invariants
from .step13_smooth_rts import rts_smooth, assemble, validate_post_rts_invariants
from .step14_velocity import compute_velocity_columns


class PostContext(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def step_load(raw_path=None, calib_top='auto', calib_front='auto'):
    if raw_path is None:
        raw_path = settings.RAW_EXCEL
    df, raw_input_columns = load_raw_excel.read_raw_trajectory(raw_path)
    df = load_raw_excel.ensure_missing_reason(df)

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
        calib_top = load_raw_excel.load_calib_top()
    if calib_front == 'auto':
        calib_front = load_raw_excel.load_calib_front()
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


def step06_remove_outliers(ctx):
    ctx['df'], ctx['n_outliers'] = hampel_filter(ctx['df'])
    return ctx


def step07_freeze_anchors(ctx):
    ctx['df'] = freeze_anchor_raw_snapshot(ctx['df'])
    return ctx


def step08_estimate_noise(ctx):
    sigma_a, n_q_runs, q_used_fallback = estimate_process_scale(ctx['df'])
    print(f'Q sigma_a={sigma_a}, valid runs={n_q_runs}, fallback={q_used_fallback}')
    ctx.update(sigma_a=sigma_a, n_q_runs=n_q_runs, q_used_fallback=q_used_fallback)
    return ctx


def step09_classify_recovery(ctx):
    df = classify_states(ctx['df'])
    ctx['df'] = classify_top_zone_support(df, ctx['calib_top'])
    return ctx


def step10_recover_missing(ctx):
    df, reconstruction_stats = interpolate_missing(
        ctx['df'], ctx['calib_top'], ctx['calib_front'], sigma_a=ctx['sigma_a']
    )
    ctx['df'] = validate_reconstruction_invariants(df)
    ctx['reconstruction_stats'] = reconstruction_stats
    return ctx


def step11_zone_and_grab(ctx):
    df, n_grab = classify_zone_grab(ctx['df'])
    ctx['df'] = validate_grab_invariants(df)
    ctx['n_grab'] = n_grab
    return ctx


def step12_display_zone(ctx):
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


def step13_smooth_rts(ctx):
    df = rts_smooth(ctx['df'], sigma_a=ctx['sigma_a'])
    df = assemble(df)
    ctx['df'] = validate_post_rts_invariants(df)
    return ctx


def step14_velocity(ctx):
    ctx['df'] = compute_velocity_columns(ctx['df'])
    return ctx


def step_save(ctx, out_path=None):
    if out_path is None:
        out_path = settings.FINAL_EXCEL
    df = ctx['df']
    summary_df = save_final_excel.build_summary_df(
        df, ctx['n_outliers'], ctx['n_q_runs'], ctx['q_used_fallback'], ctx['sigma_a'],
        ctx['reconstruction_stats'], ctx['n_grab'],
    )
    config_df = save_final_excel.build_config_df(ctx['calib_front'])
    sheets = save_final_excel.build_final_sheets(
        df, ctx['raw_input_columns'], save_final_excel.PREFERRED_DEBUG_COLUMNS, summary_df, config_df
    )
    save_final_excel.write_final_excel(sheets, out_path)
    print(f'Saved: {out_path}')
    ctx['sheets'] = sheets
    return ctx


STEPS = [
    step06_remove_outliers, step07_freeze_anchors, step08_estimate_noise, step09_classify_recovery,
    step10_recover_missing, step11_zone_and_grab, step12_display_zone, step13_smooth_rts,
    step14_velocity,
]


def run_postprocessing(raw_path=None, out_path=None, calib_top='auto', calib_front='auto'):
    ctx = step_load(raw_path, calib_top, calib_front)
    for step in STEPS:
        ctx = step(ctx)
    ctx = step_save(ctx, out_path)
    return ctx['df']
