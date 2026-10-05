# Variables by step

Columns produced at each step. "Internal" is the column name in the code; "Excel" is the name in the output file (`=` if identical).
Constant values are in `main_code/settings.py`.

## Inputs

| Input | Settings key | Read by |
|---|---|---|
| Top and front videos (frame-aligned) | `inputs.top_video`, `inputs.front_video` | `preprocessing/load_videos.py` |
| YOLO weights | `inputs.top_weights`, `inputs.front_weights` | `preprocessing/step01_detect_object.py` |
| Calibration NPZ (R, t) | `inputs.calib_top_npz`, `inputs.calib_front_npz` | `preprocessing/load_calibration.py`, `postprocessing/load_raw_excel.py` |
| Camera intrinsics (K, D) | `cameras.top`, `cameras.front` | same as above |

## Preprocessing → `3D_raw_online.xlsx`

### ① Object detection — `step01_detect_object.py`

| Internal | Excel (Trajectory) | Meaning |
|---|---|---|
| `Status_Top`, `Status_Front` | `Top_Status`, `Front_Status` | DETECTED / LOST |
| `Top_px_u`, `Top_px_v` | `Top_U_px`, `Top_V_px` | Top box center [px] |
| `Front_px_u`, `Front_px_v` | `Front_U_px`, `Front_V_px` | Front box center [px] |
| `Top_conf`, `Front_conf` | `Top_Conf`, `Front_Conf` | YOLO confidence |

### ② Ray generation and refraction — `common/refraction.py`

No output columns. Interfaces: water surface `Z = WATER_SURFACE_Z_CM`, front wall `Y = FRONT_WALL_Y_CM`.

### ③ Triangulation — `step03_triangulate.py`

| Internal | Excel | Sheet | Meaning |
|---|---|---|---|
| `Candidate_ON_X/Y/Z` | `Candidate_ON_X/Y/Z_cm` | Triangulation | Refraction ON candidate |
| `Candidate_OFF_X/Y/Z` | `Candidate_OFF_X/Y/Z_cm` | Triangulation | Refraction OFF candidate (comparison) |
| `Candidate_X/Y/Z` | `Candidate_X/Y/Z_cm` | Triangulation | Candidate after the ray-gap gate |
| `Geometry_Status_ON/OFF` | = | Triangulation | VALID / FAILED / NOT_PAIRED |
| `Ray_Gap_cm` | = | Trajectory | Closest distance between the two rays |
| `Ray_Gap_ON_cm`, `Ray_Gap_OFF_cm` | = | Triangulation | Ray gap for ON / OFF |
| `Ray_s_Top_cm`, `Ray_s_Front_cm` | `Top_Ray_Dist_cm`, `Front_Ray_Dist_cm` | Triangulation | Ray length from the interface to the closest point |

### ④ Gating and track confirmation — `step04_gate_and_track.py`

| Internal | Excel | Sheet | Meaning |
|---|---|---|---|
| `Meas_Status` | = | Trajectory | TRIANGULATED / NO_CANDIDATE / REJECTED_JUMP / REJECTED_MAHALANOBIS / TENTATIVE_HOLD / *_RESET |
| `Missing_Reason` | = | Trajectory | NONE / TOP_MISSING / FRONT_MISSING / BOTH_MISSING / RAY_GAP_EXCEEDED / TRIANGULATION_FAILED / REJECTED_* / TENTATIVE_HOLD |
| `Track_State` | = | Online_Filter | LOST / TENTATIVE / CONFIRMED |
| `KF_Updated` | = | Online_Filter | Filter updated in this frame |
| `Online_X/Y/Z` | `KF_X/Y/Z_cm` | Online_Filter | Gate filter state |
| `KF_PosStd_X/Y/Z` | `KF_Std_X/Y/Z_cm` | Online_Filter | Gate filter position std |

### ⑤ Gripper-frame transform — `step05_to_gripper_frame.py`

| Internal | Excel (Trajectory) | Meaning |
|---|---|---|
| `Meas_X/Y/Z` | `Meas_X/Y/Z_cm` | Accepted 3D position, origin = gripper (motor) center |

### Other raw columns

| Internal | Excel |
|---|---|
| `Frame` | = |
| `Time_sec` | `Time_s` |
| `Sig_*`, `Tracker_Ref_Frame`, `*_Source_Frame`, `*_Sync_*`, `Sync_*`, `*_Frame_Reused` | not written |

## Postprocessing → `3D_final.xlsx`

The raw Excel names are converted back to internal names when read (`settings.RAW_COLUMN_RENAME`).

### Input — `load_raw_excel.py`

| Internal | Excel | Meaning |
|---|---|---|
| `Raw_Meas_X/Y/Z` | not written | Copy of `Meas_*` before step ⑥ |

### ⑥ Outlier removal — `step06_remove_outliers.py`

| Internal | Excel | Meaning |
|---|---|---|
| `Meas_X/Y/Z` | — | Outliers set to NaN |
| `Outlier_Flag` | Debug | Outlier frame |
| `Missing_Reason` | not written | HAMPEL_OUTLIER for outlier frames |

### ⑦ Anchor snapshot — `step07_freeze_anchors.py`

| Internal | Excel | Meaning |
|---|---|---|
| `Anchor_Raw_X/Y/Z`, `Anchor_Raw_Valid` | Debug | Stereo positions after step ⑥, never updated |

### ⑧ Process noise — `step08_estimate_noise.py`

| Value | Excel | Meaning |
|---|---|---|
| `sigma_a` (3 axes) | Summary `Sigma_a_*_cm_per_frame2` | Process noise for step ⑬ |

### ⑨ Recovery mode — `step09_classify_recovery.py`

| Internal | Excel | Meaning |
|---|---|---|
| `Recovery_Mode` | Debug | NONE / TOP_RAY (front missing) / FRONT_RAY (top missing) |
| `Recovery_Run_ID` | Plot_Aux | Segment number of one recovery mode |
| `State2` | Debug | DETECTED / SUB / LOST (updated in ⑩) |
| `Top_Zone_Supported` | Debug | Top ray passes the work-zone box (Z 3–25 cm) |

### ⑩ Single-view recovery — `step10_recover_missing.py`, `step10_hidden_axis_model.py`

| Internal | Excel | Sheet | Meaning |
|---|---|---|---|
| `Meas_X/Y/Z` | `X_cm/Y_cm/Z_cm` | Result | Recovered positions filled in (pre-smoothing final position) |
| `Filled_Type` | `Coord_Type` | Result | Converted in ⑭ (e.g. KALMAN_RTS_EPISODE_* → *_KALMAN) |
| `Hidden_Axis_Posterior_Std` | `Hidden_Axis_Std_cm` | Result | Std of the estimated axis |
| `Z_Floor_Applied` | = | Result | Z fixed at the 3 cm floor |
| `Hidden_Stop_Reason` | = | Debug | NO_RAW_ANCHOR / STATE_INIT_FAILED / POSTERIOR_STD_LIMIT / MAX_EXTRAP_LEN / RAY_GEOMETRY_FAILURE / FRONT_X_SANITY / RAY_OR_SANITY_FAILURE |
| `Recovery_Applied`, `Hidden_Axis_*`, `Anchor_Left/Right_*`, `Hidden_Episode_ID` | = | Debug | Recovery diagnostics |

### ⑪ Work zone and GRAB — `step11_zone_and_grab.py`

| Internal | Excel | Sheet | Meaning |
|---|---|---|---|
| `Coord_State` | `State` (Result), `=` (Debug) | Result, Debug | DETECTED(_ZONE) → DETECTED, SUB(_ZONE) → RECOVERED, LOST, GRAB |
| `In_Zone` | not written | — | Work-zone flag on pre-smoothing positions (Result `In_Zone` has the same values via ⑫) |
| `Grab_*` | = | Debug | GRAB evidence |

### ⑫ Display zone — `step12_display_zone.py`

| Internal | Excel | Sheet | Meaning |
|---|---|---|---|
| `In_Zone_PreRTS` | `In_Zone` | Result | Copy of ⑪ `In_Zone` |
| `Display_In_Zone` | = | Plot_Aux | Plot shading only |
| `Display_Zone_*` | = | Debug | Bridging diagnostics |

### ⑬ RTS smoothing — `step13_smooth_rts.py`

| Internal | Excel | Sheet | Meaning |
|---|---|---|---|
| `X_rts/Y_rts/Z_rts` | = | Debug | RTS output |
| `X_cm/Y_cm/Z_cm` | `X_vis_cm/Y_vis_cm/Z_vis_cm` | Plot_Aux | Plot positions (RTS for stereo, recovered values elsewhere) |
| `RTS_Std_X/Y/Z` | `Vis_Std_X/Y/Z_cm` | Plot_Aux | Plot position std |
| `In_Zone_Final` | = | Debug | Work-zone flag on plot positions |

### ⑭ Velocity — `step14_velocity.py`

| Internal | Excel | Sheet |
|---|---|---|
| `Vx/Vy/Vz_cm_s`, `Speed_3D_cm_s` | = | Result |
| `Vx/Vy/Vz_Source`, `Speed_3D_Source` | = | Result |
| `Vx/Vy/Vz_obs_cm_s`, `Speed_3D_obs_cm_s` | = | Result |
| `V_Noise_Ratio`, `V_obs_Noise_Ratio` | = | Result |
| `Result_State`, `Result_Coord_Type` | `State`, `Coord_Type` | Result |
| `V_N_Used`, `V_obs_N_Used` | = | Debug |

## Naming differences to resolve

Internal names were kept unchanged so that results stay identical to the original code.

| Internal | Excel | Issue |
|---|---|---|
| `Meas_X` | Result `X_cm` | `Meas_X` changes meaning: stereo → after ⑥ → with recovered values after ⑩ |
| `X_cm` (⑬) | Plot_Aux `X_vis_cm` | Internal `X_cm` is the plot position; Excel Result `X_cm` is the pre-smoothing position |
| `In_Zone`, `In_Zone_PreRTS`, `In_Zone_Final`, `Display_In_Zone` | Result `In_Zone` = `In_Zone_PreRTS` | Four work-zone columns |
| `State2` → `Coord_State` → `Result_State` | Result `State` | Three state columns |
| `Filled_Type` → `Result_Coord_Type` | Result `Coord_Type` | |
| `Hidden_Axis_Posterior_Std` | Result `Hidden_Axis_Std_cm` | |
| `Status_Top`, `Top_px_u`, `Top_conf`, `Time_sec` | `Top_Status`, `Top_U_px`, `Top_Conf`, `Time_s` | Converted only when writing/reading the raw Excel |
| `Online_X`, `KF_PosStd_X`, `Ray_s_Top_cm` | `KF_X_cm`, `KF_Std_X_cm`, `Top_Ray_Dist_cm` | |

After renaming, `python edit_check/check_results_unchanged.py` must pass for Result, Legend, Plot_Aux, Summary, and Configuration. Debug column names are expected to change.
