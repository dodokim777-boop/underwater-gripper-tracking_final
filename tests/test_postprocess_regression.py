# -*- coding: utf-8 -*-
"""
후처리 회귀 테스트: 코드를 고친 뒤에도 결과가 바뀌지 않았는지 확인한다.

tests/data/final_expected.xlsx 는 정리 전 원본 postprocess_final_20261001.py로
tests/data/raw_synthetic.xlsx(합성 궤적: 선형 보간·Kalman episode·말단 외삽·GRAB·LOST 포함)를
처리한 결과다. 정리본이 같은 입력으로 6개 시트를 똑같이 만들어야 통과한다.

실행 (저장소 최상위 폴더에서):  python tests/test_postprocess_regression.py
의도적으로 계산을 바꾼 경우에는 결과를 확인한 뒤 final_expected.xlsx를 새 결과로 교체한다.
"""
import os
import sys
import tempfile
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from gtrack import settings  # noqa: E402
from gtrack.postprocess.pipeline import run_postprocess  # noqa: E402

DATA = os.path.join(HERE, 'data')


def main():
    out_dir = tempfile.mkdtemp()
    settings.apply_config({
        'experiment': {'name': 'regression', 'data_root': out_dir},
        'inputs': {'calib_top_npz': os.path.join(DATA, 'calib_top.npz'),
                   'calib_front_npz': os.path.join(DATA, 'calib_front.npz')},
    })
    out = os.path.join(out_dir, 'final.xlsx')
    run_postprocess(raw_path=os.path.join(DATA, 'raw_synthetic.xlsx'), out_path=out)
    expected = pd.read_excel(os.path.join(DATA, 'final_expected.xlsx'), sheet_name=None)
    actual = pd.read_excel(out, sheet_name=None)
    assert list(expected) == list(actual), f'시트 구성 다름: {list(expected)} vs {list(actual)}'
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name], check_exact=True)
        print(f'  {name:<14} 동일 {expected[name].shape}')
    print('PASS: 후처리 결과가 기준 결과와 같습니다.')


if __name__ == '__main__':
    main()
