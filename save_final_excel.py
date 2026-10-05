"""Output: build and write 3D_final.xlsx sheets."""

import numpy as np
import pandas as pd

from .. import settings
from ..common.file_utils import format_workbook_sheets
from .step14_velocity import _time_seconds, velocity_summary_rows, velocity_config_df


def build_legend_df():
    rows = [
        ('Result', 'Frame', '-', 'Frame number on the synchronized timeline; same as in the raw file and the annotated video.'),
        ('Result', 'Time_s', 's', 'Frame time.'),
        ('Result', 'X_cm, Y_cm, Z_cm', 'cm',
         'Final position: pre-smoothing coordinates after outlier removal and gap recovery. '
         'Stereo frames are triangulated; recovered frames are reconstructed.'),
        ('Result', 'State', '-',
         'DETECTED: both cameras / RECOVERED: recovered from one camera / '
         'LOST: no position / GRAB: loss judged as capture (no position).'),
        ('Result', 'Coord_Type', '-',
         'STEREO: both cameras. TOP_*: top camera only (X, Y from the top ray, Z estimated). '
         'FRONT_*: front camera only (X, Z from the front ray, Y estimated). '
         'LINEAR: linear interpolation between observations, KALMAN: Kalman estimate over repeated gaps, '
         'EXTRAP: extension from one side only (largest error). NONE: no position.'),
        ('Result', 'Z_Floor_Applied', '-',
         'TRUE: the estimated Z fell below the 3 cm floor and was fixed at 3 cm. The position is a projection '
         'on the floor plane, not the object position; exclude it from position analysis. Also excluded from velocity.'),
        ('Result', 'Hidden_Axis_Std_cm', 'cm',
         'Std of the estimated axis in recovered frames (Z for TOP, Y for FRONT). Empty for stereo frames.'),
        ('Result', 'In_Zone', '-',
         f'TRUE inside the work zone (X ±{settings.ZONE_X_HALF:g}, Y ±{settings.ZONE_Y_HALF:g}, Z {settings.ZONE_Z_LOW:g}–{settings.ZONE_Z_HIGH:g} cm), '
         'based on the positions in this sheet.'),
        ('Result', 'Vx_cm_s, Vy_cm_s, Vz_cm_s', 'cm/s',
         'Velocity including recovered positions. Least-squares slope of position vs. time over t ± 5 frames '
         '(11 frames, about 0.17 s; Savitzky–Golay first derivative). Empty frames in the window are skipped. '
         'Recorded only if there is at least one frame on each side of t, the noise ratio is ≤ 2, and the mean position '
         'of the used frames is within 2 frames of t. Recorded only for frames with a position and no Z floor.'),
        ('Result', 'Speed_3D_cm_s', 'cm/s', '√(Vx² + Vy² + Vz²).'),
        ('Result', 'Vx_Source, Vy_Source, Vz_Source', '-',
         'Lowest grade among the frames used in the velocity window. '
         'OBSERVED (both cameras) > RAY_CONSTRAINED (axis set by one camera ray; includes the estimated-axis error '
         'scaled by the ray slope) > MODEL_INTERP (axis estimated by interpolation or Kalman) > '
         'MODEL_EXTRAP (axis estimated by extrapolation).'),
        ('Result', 'Speed_3D_Source', '-', 'Lowest grade of the three axes.'),
        ('Result', 'V_Noise_Ratio', '-', 'Velocity noise relative to using all 11 frames; 1.0 means all frames were used.'),
        ('Result', 'Vx_obs_cm_s, Vy_obs_cm_s, Vz_obs_cm_s, Speed_3D_obs_cm_s', 'cm/s',
         'Velocity from stereo frames only (Coord_Type = STEREO), same method. '
         'Compare with the velocity including recovery to check the effect of recovered positions.'),
        ('Result', 'V_obs_Noise_Ratio', '-', 'Noise ratio of the observed-only velocity.'),
        ('Result', '(Velocity note)', '-',
         settings.VEL_SLOW_PHASE_FLUCTUATION_NOTE + '. Velocity changes below this size cannot be separated from fluctuation. '
         'The position scale error (about 2.6%) carries over to velocity at the same ratio.'),
        ('Plot_Aux', 'X_vis_cm, Y_vis_cm, Z_vis_cm', 'cm',
         'Positions for plots. Stereo segments use 3D RTS smoothing; recovered frames keep the recovered values. '
         'Use the Result positions for analysis.'),
        ('Plot_Aux', 'Vis_Std_X_cm, Vis_Std_Y_cm, Vis_Std_Z_cm', 'cm', 'Std of the RTS-smoothed position (for error bands).'),
        ('Plot_Aux', 'Display_In_Zone', '-',
         'Work-zone flag for plot shading only. Short flickers at the boundary are filled; use In_Zone in Result for analysis.'),
        ('Plot_Aux', 'Top_Status, Front_Status', '-', 'Detection per camera (DETECTED / LOST).'),
        ('Plot_Aux', 'Recovery_Run_ID', '-', 'Recovery segment number, for coloring recovered segments.'),
        ('Debug', '(all)', '-', 'Intermediate postprocessing values for code checking. Column names are the internal code names.'),
        ('Contact_Events', 'Contact_Frame', '-', 'First gripper contact frame judged from the video (contact CSV input).'),
        ('Contact_Events', 'Avg_Start_Frame, Avg_End_Frame', '-',
         'V-T window used for averaging. It ends 5 frames before contact so that the velocity window does not use positions after contact.'),
        ('Contact_Events', 'Avg_Vx_cm_s, Avg_Vy_cm_s, Avg_Vz_cm_s', 'cm/s',
         'Per-axis mean of the observed-only V-T (V*_obs_cm_s) in the window.'),
        ('Contact_Events', 'Avg_Speed_cm_s', 'cm/s', 'Magnitude of the per-axis mean velocity. Approach speed without fluctuation.'),
        ('Contact_Events', 'Mean_Speed_cm_s', 'cm/s', 'Mean of the speed curve (Speed_3D_obs_cm_s). Includes fluctuation; for comparison.'),
        ('Contact_Events', 'Obs_Frames_Used, Obs_Coverage', '-',
         'Number of frames used and their fraction of the window. Treat the mean velocity with care if the fraction is low.'),
        ('Contact_Events', 'Vel_Change_In_Window_cm_s', 'cm/s',
         'Magnitude of (mean velocity of the second half - mean velocity of the first half). Large values mean the velocity changed within the window.'),
        ('Contact_Events', 'Avg_Speed_k_Spread_cm_s', 'cm/s',
         'Max - min of the mean speed computed with half-widths k = 3, 5, 7. Large values mean the result depends on the method.'),
        ('Contact_Events', 'Check_Flags', '-',
         'VEL_CHANGE: Vel_Change_In_Window_cm_s > 0.3 / VEL_CHANGE_NA: one half of the window has no observed velocity / '
         'K_SPREAD: Avg_Speed_k_Spread_cm_s > 0.1 / NO_DATA: no observed velocity to average / '
         'FRAME_NOT_FOUND: contact frame not in Result. Recheck flagged events in the video.'),
    ]
    return pd.DataFrame(rows, columns=['Sheet', 'Column', 'Unit', 'Description'])


def build_final_sheets(df, raw_input_columns, preferred_columns, summary_df, config_df):
    coord_type = df['Result_Coord_Type'].to_numpy()
    hidden_std = df['Hidden_Axis_Posterior_Std'].to_numpy(dtype=float).copy()
    hidden_std[np.isin(coord_type, ['STEREO', 'NONE'])] = np.nan
    in_zone_col = 'In_Zone_PreRTS' if 'In_Zone_PreRTS' in df.columns else 'In_Zone'
    time_s, _ = _time_seconds(df)

    result = pd.DataFrame({
        'Frame': df['Frame'].to_numpy(),
        'Time_s': time_s,
        'X_cm': df['Meas_X'].to_numpy(dtype=float),
        'Y_cm': df['Meas_Y'].to_numpy(dtype=float),
        'Z_cm': df['Meas_Z'].to_numpy(dtype=float),
        'State': df['Result_State'].to_numpy(),
        'Coord_Type': coord_type,
        'Z_Floor_Applied': df['Z_Floor_Applied'].fillna(False).astype(bool).to_numpy(),
        'Hidden_Axis_Std_cm': hidden_std,
        'In_Zone': df[in_zone_col].fillna(False).astype(bool).to_numpy(),
    })
    for col in ['Vx_cm_s', 'Vy_cm_s', 'Vz_cm_s', 'Speed_3D_cm_s',
                'Vx_Source', 'Vy_Source', 'Vz_Source', 'Speed_3D_Source', 'V_Noise_Ratio',
                'Vx_obs_cm_s', 'Vy_obs_cm_s', 'Vz_obs_cm_s', 'Speed_3D_obs_cm_s', 'V_obs_Noise_Ratio']:
        result[col] = df[col].to_numpy()

    plot_aux = pd.DataFrame({
        'Frame': df['Frame'].to_numpy(),
        'Time_s': time_s,
        'X_vis_cm': df['X_cm'].to_numpy(dtype=float),
        'Y_vis_cm': df['Y_cm'].to_numpy(dtype=float),
        'Z_vis_cm': df['Z_cm'].to_numpy(dtype=float),
        'Vis_Std_X_cm': df['RTS_Std_X'].to_numpy(dtype=float),
        'Vis_Std_Y_cm': df['RTS_Std_Y'].to_numpy(dtype=float),
        'Vis_Std_Z_cm': df['RTS_Std_Z'].to_numpy(dtype=float),
        'Display_In_Zone': df['Display_In_Zone'].fillna(False).astype(bool).to_numpy(),
        'Top_Status': df['Status_Top'].to_numpy(),
        'Front_Status': df['Status_Front'].to_numpy(),
        'Recovery_Run_ID': df['Recovery_Run_ID'].to_numpy(),
    })

    debug_exclude = set(raw_input_columns) | {
        'X_cm', 'Y_cm', 'Z_cm', 'Raw_Meas_X', 'Raw_Meas_Y', 'Raw_Meas_Z',
        'RTS_Std_X', 'RTS_Std_Y', 'RTS_Std_Z', 'Hidden_Axis_Posterior_Std',
        'Z_Floor_Applied', 'In_Zone', 'In_Zone_PreRTS', 'Display_In_Zone', 'Recovery_Run_ID',
    }
    debug_cols = ['Frame'] + [
        c for c in preferred_columns
        if c in df.columns and c not in debug_exclude and c != 'Frame'
    ] + ['V_N_Used', 'V_obs_N_Used']
    debug = df[debug_cols].copy()

    return {
        'Result': result,
        'Legend': build_legend_df(),
        'Plot_Aux': plot_aux,
        'Summary': summary_df,
        'Configuration': config_df,
        'Debug': debug,
    }


def write_final_excel(sheets, out_path):
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        format_workbook_sheets(writer.book)


def build_summary_df(df, n_outliers, n_q_runs, q_used_fallback, sigma_a,
                     reconstruction_stats, n_grab):
    coord_lost_frames = int(df['Coord_State'].eq('LOST').sum())
    grab_frames = int(df['Coord_State'].eq('GRAB').sum())
    unavailable_frames = int(df[['X_cm', 'Y_cm', 'Z_cm']].isna().any(axis=1).sum())
    available_frames = int(len(df) - unavailable_frames)

    summary_rows = [
        ('Frames', len(df)),
        ('Raw valid measurements', int(df['Raw_Meas_X'].notna().sum())),
        ('Anchor raw valid frames', int(df['Anchor_Raw_Valid'].sum())),
        ('Hampel outliers', n_outliers),
        ('Q valid runs', n_q_runs),
        ('Q fallback used', q_used_fallback),
        ('Sigma_a_X_cm_per_frame2', sigma_a[0]),
        ('Sigma_a_Y_cm_per_frame2', sigma_a[1]),
        ('Sigma_a_Z_cm_per_frame2', sigma_a[2]),
        ('TOP_RAY candidate frames', int(df['Recovery_Mode'].eq(settings.RECOVERY_TOP_RAY).sum())),
        ('FRONT_RAY candidate frames', int(df['Recovery_Mode'].eq(settings.RECOVERY_FRONT_RAY).sum())),
        ('TOP_RAY reconstructed frames', reconstruction_stats['TOP_RECON_FRAMES']),
        ('FRONT_RAY reconstructed frames', reconstruction_stats['FRONT_RECON_FRAMES']),
        ('TOP_RAY two-sided linear-interpolation segments', reconstruction_stats['TOP_INTERP_SEGMENTS']),
        ('TOP_RAY extrapolated segments', reconstruction_stats['TOP_EXTRAP_SEGMENTS']),
        ('TOP_RAY zone-constrained segments', reconstruction_stats['TOP_ZONE_SEGMENTS']),
        ('TOP_RAY 1D Kalman/RTS episodes', reconstruction_stats['TOP_KALMAN_EPISODES']),
        ('FRONT_RAY two-sided linear-interpolation segments', reconstruction_stats['FRONT_INTERP_SEGMENTS']),
        ('FRONT_RAY extrapolated segments', reconstruction_stats['FRONT_EXTRAP_SEGMENTS']),
        ('FRONT_RAY 1D Kalman/RTS episodes', reconstruction_stats['FRONT_KALMAN_EPISODES']),
        ('FRONT_RAY X-sanity rejected frames', reconstruction_stats['FRONT_X_SANITY_REJECTED_FRAMES']),
        ('Reconstruction-lost segments', reconstruction_stats['LOST_SEGMENTS']),
        ('Reconstruction-lost frames', reconstruction_stats['LOST_FRAMES']),
        ('Zone-floor reconstructed frames', int(df['Z_Floor_Applied'].fillna(False).sum())),
        ('Sparse-anchor 60-frame expanded segments', reconstruction_stats['ANCHOR_WINDOW_60_SEGMENTS']),
        ('Sparse-anchor insufficient segments', reconstruction_stats['INSUFFICIENT_ANCHOR_SEGMENTS']),
        ('Theil-Sen local models fitted', reconstruction_stats['THEIL_SEN_MODELS_FIT']),
        ('Theil-Sen slope-clipped models', reconstruction_stats['THEIL_SEN_SLOPE_CLIPPED_MODELS']),
        ('Hidden-axis uncertainty-stopped frames', reconstruction_stats['UNCERTAINTY_STOPPED_FRAMES']),
        ('Hidden-axis max-length-stopped frames', reconstruction_stats['MAX_LENGTH_STOPPED_FRAMES']),
        ('Pre-RTS in-zone coordinate frames', int(df['In_Zone_PreRTS'].fillna(False).sum())),
        ('Display-zone bridge frames', int(df['Display_Zone_Bridge'].fillna(False).sum())),
        ('Display-zone SHORT bridge frames', int(df['Display_Zone_Bridge_Type'].eq('SHORT').sum())),
        ('Display-zone STABLE_OBSERVED bridge frames', int(df['Display_Zone_Bridge_Type'].eq('STABLE_OBSERVED').sum())),
        ('Display-zone bridge runs', int(
            pd.to_numeric(df['Display_Zone_Bridge_Run_ID'], errors='coerce')
            .fillna(-1).max() + 1
        )),
        ('Display in-zone frames', int(df['Display_In_Zone'].fillna(False).sum())),
        ('Coord_State LOST frames', coord_lost_frames),
        ('GRAB frames', grab_frames),
        ('Pre-RTS finite-coordinate frames', int(df['Pre_RTS_Coord_Available'].sum())),
        ('RTS input valid frames', int(df['RTS_Input_Valid'].sum())),
        ('RTS segments', int(
            pd.to_numeric(df['RTS_Segment_ID'], errors='coerce').fillna(-1).max() + 1
        )),
        ('RTS direct recovery-mode boundaries', int(df['RTS_Hard_Boundary'].fillna(False).sum())),
        ('RTS zone-class changes', int(df['Zone_RTS_Class_Changed'].fillna(False).sum())),
        ('Final coordinate available frames', available_frames),
        ('Final coordinate unavailable frames', unavailable_frames),
        ('Final in-zone coordinate frames', int(df['In_Zone_Final'].fillna(False).sum())),
        ('GRAB events', n_grab),
    ]

    if 'Ray_Gap_cm' in df.columns:
        accepted_mask = (
            df['Meas_Status'].eq('TRIANGULATED')
            if 'Meas_Status' in df.columns
            else df[settings.RAW_MEAS_COLS].notna().all(axis=1)
        )
        gap = pd.to_numeric(
            df.loc[accepted_mask, 'Ray_Gap_cm'], errors='coerce'
        ).dropna().to_numpy()
        if gap.size:
            summary_rows += [
                ('Accepted ray gap median [cm]', float(np.median(gap))),
                ('Accepted ray gap P95 [cm]', float(np.percentile(gap, 95))),
                ('Accepted ray gap max [cm]', float(np.max(gap))),
            ]

    summary_rows += velocity_summary_rows(df)
    summary_df = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])
    return summary_df


def build_config_df(calib_front):
    config_df = pd.DataFrame([
        ('HAMPEL_K', settings.HAMPEL_K),
        ('HAMPEL_NSIGMA', settings.HAMPEL_NSIGMA),
        ('HAMPEL_ABS_FLOOR_X_cm', settings.HAMPEL_ABS_FLOOR[0]),
        ('HAMPEL_ABS_FLOOR_Y_cm', settings.HAMPEL_ABS_FLOOR[1]),
        ('HAMPEL_ABS_FLOOR_Z_cm', settings.HAMPEL_ABS_FLOOR[2]),
        ('SIG_RTS_RAW_X_cm', settings.SIG_RTS_RAW[0]),
        ('SIG_RTS_RAW_Y_cm', settings.SIG_RTS_RAW[1]),
        ('SIG_RTS_RAW_Z_cm', settings.SIG_RTS_RAW[2]),
        ('SIG_RTS_RECON_TOP_X_cm', settings.SIG_RTS_RECON_TOP[0]),
        ('SIG_RTS_RECON_TOP_Y_cm', settings.SIG_RTS_RECON_TOP[1]),
        ('SIG_RTS_RECON_TOP_Z_cm', settings.SIG_RTS_RECON_TOP[2]),
        ('SIG_RTS_RECON_TOP_finalized', settings.SIG_RTS_RECON_TOP_FINALIZED),
        ('SIG_RTS_RECON_FRONT_X_cm', settings.SIG_RTS_RECON_FRONT[0]),
        ('SIG_RTS_RECON_FRONT_Y_cm', settings.SIG_RTS_RECON_FRONT[1]),
        ('SIG_RTS_RECON_FRONT_Z_cm', settings.SIG_RTS_RECON_FRONT[2]),
        ('SIG_RTS_RECON_FRONT_finalized', settings.SIG_RTS_RECON_FRONT_FINALIZED),
        ('V_MAX_Z_cm_per_frame', settings.V_MAX_Z),
        ('V_MAX_Y_cm_per_frame', settings.V_MAX_Y),
        ('V_MAX_X_FRONT_RAY_cm_per_frame', settings.V_MAX_X_FRONT_RAY),
        ('FRONT_X_SANITY_MARGIN_cm', settings.FRONT_X_SANITY_MARGIN),
        ('MAX_EXTRAP_LEN_frames', settings.MAX_EXTRAP_LEN),
        ('TOP_ZONE_MAX_EXTRAP_LEN_frames', settings.TOP_ZONE_MAX_EXTRAP_LEN),
        ('ANCHOR_SEARCH_INITIAL_frames', settings.ANCHOR_SEARCH_INITIAL),
        ('ANCHOR_SEARCH_EXTENDED_frames', settings.ANCHOR_SEARCH_EXTENDED),
        ('ANCHOR_MODEL_MIN_POINTS', settings.ANCHOR_MODEL_MIN_POINTS),
        ('HIDDEN_EPISODE_LINK_MAX_frames', settings.HIDDEN_EPISODE_LINK_MAX),
        ('HIDDEN_REACQUIRE_MIN_CONSECUTIVE_RAW', settings.HIDDEN_REACQUIRE_MIN_CONSECUTIVE_RAW),
        ('HIDDEN_AXIS_MOTION_MODEL', settings.HIDDEN_AXIS_MOTION_MODEL),
        ('HIDDEN_HUBER_K', settings.HIDDEN_HUBER_K),
        ('HIDDEN_Q_SCALE_Z', settings.HIDDEN_Q_SCALE_Z),
        ('HIDDEN_Q_SCALE_Y', settings.HIDDEN_Q_SCALE_Y),
        ('HIDDEN_STD_STOP_Z_cm', settings.HIDDEN_STD_STOP_Z_CM),
        ('HIDDEN_STD_STOP_Y_cm', settings.HIDDEN_STD_STOP_Y_CM),
        ('HIDDEN_STD_STOP_finalized', settings.HIDDEN_STD_STOP_THRESHOLDS_FINALIZED),
        ('Hidden-axis clean-gap rule',
         'one recovery run with actual raw boundaries on both sides -> nearest-boundary linear interpolation'),
        ('Hidden-axis intermittent rule',
         'same recovery mode repeating within 60 frames without >=5 consecutive raw stereo -> one continuous 1D CV Kalman/RTS episode; intermediate/future raw are position updates, not new velocity priors'),
        ('Hidden-axis initial velocity',
         'Theil-Sen from pre-episode raw history within 30 frames, extended to 60 if fewer than 3 raw points; slope dispersion sets initial velocity covariance; intermediate anchors never recreate this prior'),
        ('Hidden-axis anchor weighting',
         'actual raw positions use Kalman gain; Huber innovation weighting inflates R for inconsistent anchors without replacing the running velocity state'),
        ('Hidden-axis one-sided stop',
         'posterior std threshold plus 150-frame general cap; continuous in-zone TOP_RAY support uses a 300-frame cap'),
        ('Hidden-axis raw reuse', 'reconstructed coordinates never enter Anchor_Raw snapshot'),
        ('DISPLAY_ZONE_BRIDGE_MAX_frames', settings.DISPLAY_ZONE_BRIDGE_MAX),
        ('DISPLAY_ZONE_STABLE_SEARCH_frames', settings.DISPLAY_ZONE_STABLE_SEARCH),
        ('DISPLAY_ZONE_STABLE_MIN_POINTS_PER_SIDE', settings.DISPLAY_ZONE_STABLE_MIN_POINTS_PER_SIDE),
        ('Display-zone bridge rule',
         'one immutable pre-RTS pass; <=7-frame gaps use original in-zone bounds; longer gaps require at least one camera detected at every frame and >=3 nearest valid pre-RTS in-zone coordinates on each side within 30 frames'),
        ('Display-zone bridge reuse',
         'never used for coordinates, GRAB, reconstruction anchors, or RTS input'),
        ('ZONE_Z_LOW_cm (In_Zone)', settings.ZONE_Z_LOW),
        ('ZONE_Z_HIGH_cm (In_Zone)', settings.ZONE_Z_HIGH),
        ('ZONE_TRACK_CENTER_Z_FLOOR_cm', settings.ZONE_TRACK_CENTER_Z_FLOOR),
        ('Top zone-supported recovery',
         'TOP_RAY + Top DETECTED + FRONT_MISSING + Top_Zone_Supported + reference anchor Z>=3 cm applies Z=max(Z,3 cm); for a one-sided forward tail from an actual in-zone raw anchor, continuous support extends the length cap from 150 to 300 frames while posterior-std stop remains'),
        ('Front zone support', 'not used'),
        ('Recovery modes', 'FRONT_MISSING->TOP_RAY; TOP_MISSING->FRONT_RAY'),
        ('Mode switch rule', 'split runs immediately; reconstructed coordinates never become anchors'),
        ('Non-recoverable reasons',
         'BOTH_MISSING, TENTATIVE_HOLD, REJECTED_*, TRIANGULATION_FAILED, resets'),
        ('GRAB_MIN_frames', settings.GRAB_MIN),
        ('GRAB rule',
         'after an in-zone pre-RTS anchor, >=GRAB_MIN consecutive both-camera-LOST frames confirm GRAB; label the entire coordinate-unavailable run containing that evidence'),
        ('GRAB onset convention',
         'back-label to the first coordinate-unavailable frame and end when a finite pre-RTS coordinate reappears; one-camera-visible outage edges remain part of the event'),
        ('RTS state mask',
         'smooth only DETECTED/SUB and zone variants; exclude GRAB/LOST and display bridges'),
        ('RTS recovery-source noise',
         'RAW, TOP_RAY, and FRONT_RAY use separate measurement sigma records; hidden-axis posterior std is saved per frame'),
        ('Final RTS reconstructed-coordinate preservation',
         'all accepted TOP_RAY/FRONT_RAY X/Y/Z coordinates are preserved after the visual 3D RTS; LOST/GRAB/display bridge are excluded'),
        ('RTS direct mode-switch rule',
         'split unless TOP_RAY and FRONT_RAY are separated by a trusted raw stereo anchor'),
        ('Top camera model', settings.TOP_CAMERA_MODEL),
        ('Top intrinsic source', 'hardcoded'),
        ('Front camera model', settings.FRONT_CAMERA_MODEL),
        ('Front intrinsic source', 'hardcoded'),
        ('Front wall Y [cm]', settings.FRONT_WALL_Y_CM),
        ('Front use_wall_transform', calib_front.get('use_wall_transform', False)),
    ], columns=['Parameter', 'Value'])
    config_df = pd.concat([config_df, velocity_config_df()], ignore_index=True)
    return config_df


PREFERRED_DEBUG_COLUMNS = [
    'Frame', 'X_cm', 'Y_cm', 'Z_cm', 'Coord_State',
    'In_Zone', 'In_Zone_PreRTS', 'In_Zone_Final',
    'Display_In_Zone', 'Display_Zone_State', 'Display_Zone_Bridge',
    'Display_Zone_Bridge_Type', 'Display_Zone_Bridge_Run_ID',
    'Display_Zone_Bridge_Length', 'Display_Zone_Left_Evidence_Count',
    'Display_Zone_Right_Evidence_Count',
    'Zone_RTS_Class_Changed', 'Final_Coord_Available',
    'State2', 'Recovery_Mode', 'Recovery_Run_ID', 'Recovery_Applied',
    'Anchor_Raw_Valid', 'Anchor_Raw_X', 'Anchor_Raw_Y', 'Anchor_Raw_Z',
    'Recovery_Left_Anchor_Index', 'Recovery_Right_Anchor_Index',
    'Recovery_Right_Used_Index', 'Hidden_Axis_Model', 'Hidden_Episode_ID',
    'Hidden_Axis_Filtered_Value', 'Hidden_Axis_Smoothed_Value',
    'Hidden_Axis_Filtered_Velocity', 'Hidden_Axis_Smoothed_Velocity',
    'Hidden_Axis_Posterior_Std', 'Hidden_Axis_Initial_Velocity',
    'Hidden_Axis_Initial_Velocity_Std', 'Hidden_Axis_Innovation_Weight',
    'Hidden_Anchor_Weight_Median', 'Hidden_Anchor_Weight_Min',
    'Hidden_Stop_Reason', 'Hidden_Uncertainty_Limit_cm',
    'Hidden_Has_Future_Raw', 'Hidden_Q_SigmaA', 'Hidden_R_Sigma',
    'Z_Floor_Applied', 'Top_Zone_Extension_Applied',
    'Hidden_Axis_Velocity_cm_per_frame',
    'Anchor_Left_Model_Valid', 'Anchor_Right_Model_Valid',
    'Anchor_Left_TheilSen_Slope_Raw', 'Anchor_Right_TheilSen_Slope_Raw',
    'Anchor_Left_TheilSen_Slope_Used', 'Anchor_Right_TheilSen_Slope_Used',
    'Anchor_Left_TheilSen_Residual_MAD_cm',
    'Anchor_Right_TheilSen_Residual_MAD_cm',
    'Anchor_Left_TheilSen_Pair_Count', 'Anchor_Right_TheilSen_Pair_Count',
    'Anchor_Left_TheilSen_Slope_Clipped',
    'Anchor_Right_TheilSen_Slope_Clipped',
    'Pre_RTS_Coord_Available', 'RTS_Input_Valid', 'RTS_Exclusion_Reason',
    'RTS_Segment_ID', 'RTS_Hard_Boundary', 'RTS_Measurement_Class',
    'Both_Cameras_Lost', 'Top_Zone_Supported',
    'Anchor_Left_Count', 'Anchor_Right_Count',
    'Anchor_Left_Search_Window', 'Anchor_Right_Search_Window',
    'Anchor_Strategy',
    'Grab_Event_ID', 'Grab_Anchor_Index', 'Grab_Event_Start_Index',
    'Grab_Loss_Start_Index', 'Grab_Confirm_Index',
    'Grab_Both_Lost_End_Index', 'Grab_Run_End_Index',
    'Grab_Anchor_Frame', 'Grab_Event_Start_Frame',
    'Grab_Loss_Start_Frame', 'Grab_Confirm_Frame',
    'Grab_Both_Lost_End_Frame', 'Grab_Run_End_Frame',
    'Grab_Inferred_Onset',
    'Filled_Type', 'Front_X_Sanity_Failed',
    'Outlier_Flag', 'Missing_Reason',
    'Raw_Meas_X', 'Raw_Meas_Y', 'Raw_Meas_Z',
    'Meas_X', 'Meas_Y', 'Meas_Z',
    'X_rts', 'Y_rts', 'Z_rts',
    'RTS_Std_X', 'RTS_Std_Y', 'RTS_Std_Z',
    'Assigned_Sigma_X', 'Assigned_Sigma_Y', 'Assigned_Sigma_Z',
    'RTS_Measurement_Source',
    'Status_Top', 'Status_Front', 'Top_px_u', 'Top_px_v',
    'Front_px_u', 'Front_px_v', 'Top_conf', 'Front_conf',
    'Candidate_X', 'Candidate_Y', 'Candidate_Z',
    'Candidate_ON_X', 'Candidate_ON_Y', 'Candidate_ON_Z',
    'Candidate_OFF_X', 'Candidate_OFF_Y', 'Candidate_OFF_Z',
    'Geometry_Status_ON', 'Geometry_Status_OFF',
    'Track_State', 'Meas_Status', 'KF_Updated',
    'Online_X', 'Online_Y', 'Online_Z',
    'KF_PosStd_X', 'KF_PosStd_Y', 'KF_PosStd_Z',
    'Ray_Gap_cm', 'Ray_s_Top_cm', 'Ray_s_Front_cm',
    'Ray_Gap_ON_cm', 'Ray_Gap_OFF_cm',
]
