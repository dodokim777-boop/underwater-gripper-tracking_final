"""Step ⑦: freeze post-Hampel stereo coordinates as immutable recovery anchors."""

import numpy as np
import pandas as pd

from .. import settings


def freeze_anchor_raw_snapshot(df):
    values = df[settings.MEAS_COLS].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    for axis, col in zip(('X', 'Y', 'Z'), settings.MEAS_COLS):
        snap = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        df[f'Anchor_Raw_{axis}'] = snap
    df['Anchor_Raw_Valid'] = valid
    return df
