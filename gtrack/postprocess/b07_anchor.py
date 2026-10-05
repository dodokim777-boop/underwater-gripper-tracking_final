# -*- coding: utf-8 -*-
"""
⑦ 복원 기준 관측 고정

Hampel 이후의 실제 두 시점(stereo) 좌표를 Anchor_Raw_*로 복사해 고정한다.
이후 어느 단계에서도 갱신되지 않는 불변 스냅숏이며, 복원값이 다른 결측의 기준 관측으로
재사용되는 것을 막는다.
출력: Anchor_Raw_X / Anchor_Raw_Y / Anchor_Raw_Z, Anchor_Raw_Valid
"""

import numpy as np
import pandas as pd

from .. import settings


def freeze_anchor_raw_snapshot(df):
    """Hampel 이후의 실제 stereo 좌표를 복원 전용 immutable snapshot으로 고정한다.

    이후 Meas_X/Y/Z가 복원으로 채워져도 Anchor_Raw_*와 Anchor_Raw_Valid은
    바뀌지 않는다. 따라서 앞에서 만든 복원값이 뒤 결측의 anchor로 재사용되는
    순환/누적 구조를 원천적으로 막는다.
    """
    values = df[settings.MEAS_COLS].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    for axis, col in zip(('X', 'Y', 'Z'), settings.MEAS_COLS):
        snap = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        df[f'Anchor_Raw_{axis}'] = snap
    df['Anchor_Raw_Valid'] = valid
    return df
