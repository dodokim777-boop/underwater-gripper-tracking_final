# -*- coding: utf-8 -*-
"""
⑥ Hampel 이상치 제거

국소 선형 추세(±HAMPEL_K 프레임)를 뺀 잔차의 국소 MAD로 문턱을 정하고,
축별 최소 문턱(HAMPEL_ABS_FLOOR)을 함께 쓴다. 이상치 프레임은 Meas_*를 NaN으로 바꾸고
Missing_Reason = HAMPEL_OUTLIER로 기록한다.
출력: Outlier_Flag
"""

import numpy as np
import pandas as pd

from .. import settings


def hampel_filter(df, k=None, nsigma=None):
    if k is None:
        k = settings.HAMPEL_K
    if nsigma is None:
        nsigma = settings.HAMPEL_NSIGMA
    win = 2 * k + 1
    frames = df['Frame'].to_numpy(dtype=float)
    n = len(df)
    flag = np.zeros(n, dtype=bool)

    for axis_idx, col in enumerate(settings.MEAS_COLS):
        values = df[col].to_numpy(dtype=float)
        residual = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(values[i]):
                continue
            lo, hi = max(0, i - k), min(n, i + k + 1)
            x = frames[lo:hi] - frames[i]
            y = values[lo:hi]
            valid = np.isfinite(y)
            if valid.sum() < 3:
                continue
            slope, intercept = np.polyfit(x[valid], y[valid], 1)
            residual[i] = values[i] - intercept

        residual_s = pd.Series(residual)
        local_abs_residual_median = residual_s.abs().rolling(
            win, center=True, min_periods=3
        ).median()
        threshold = np.maximum(
            nsigma * 1.4826 * local_abs_residual_median,
            settings.HAMPEL_ABS_FLOOR[axis_idx],
        )
        flag |= (residual_s.abs() > threshold).fillna(False).to_numpy()

    df.loc[flag, settings.MEAS_COLS] = np.nan
    df['Outlier_Flag'] = flag
    df.loc[flag, 'Missing_Reason'] = 'HAMPEL_OUTLIER'
    return df, int(flag.sum())
