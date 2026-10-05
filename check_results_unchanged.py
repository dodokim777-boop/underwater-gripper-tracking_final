"""Check that postprocessing results are unchanged after a code edit (run: python edit_check/check_results_unchanged.py)."""
import os
import sys
import tempfile
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from main_code import settings  # noqa: E402
from main_code.postprocessing.run_postprocessing import run_postprocessing  # noqa: E402

DATA = os.path.join(HERE, 'data')


def main():
    out_dir = tempfile.mkdtemp()
    settings.apply_config({
        'experiment': {'name': 'edit_check', 'data_root': out_dir},
        'inputs': {'calib_top_npz': os.path.join(DATA, 'calib_top.npz'),
                   'calib_front_npz': os.path.join(DATA, 'calib_front.npz')},
    })
    out = os.path.join(out_dir, 'final.xlsx')
    run_postprocessing(raw_path=os.path.join(DATA, 'raw_synthetic.xlsx'), out_path=out)
    expected = pd.read_excel(os.path.join(DATA, 'final_expected.xlsx'), sheet_name=None)
    actual = pd.read_excel(out, sheet_name=None)
    assert list(expected) == list(actual), f'Sheet list differs: {list(expected)} vs {list(actual)}'
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name], check_exact=True)
        print(f'  {name:<14} identical {expected[name].shape}')
    print('PASS: results are unchanged.')


if __name__ == '__main__':
    main()
