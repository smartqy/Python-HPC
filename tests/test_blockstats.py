import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'simulation'))

from etl.blockstats import GRID_BYTES, MASK_BYTES, block_stats, decode  # noqa: E402
from export import load_data, simulate, summary_stats  # noqa: E402


def summary_from_blocks(b):
    n = b['interior_cells'].sum()
    mean = b['sum_temp'].sum() / n
    return {
        'area_cells': n,
        'mean_temp': mean,
        'std_temp': np.sqrt(b['sum_sq_temp'].sum() / n - mean ** 2),
        'pct_above_18': b['n_above_18'].sum() * 100 / n,
        'pct_below_15': b['n_below_15'].sum() * 100 / n,
    }


@pytest.fixture(scope='module')
def solved():
    u0, mask = load_data(REPO / 'data' / 'sample', '10000')
    u, *_ = simulate(u0, mask, 'numpy', max_iter=300)
    return u[1:-1, 1:-1].copy(), mask


def test_blocks_reproduce_reference_summary(solved):
    u, mask = solved
    ref = summary_stats(np.pad(u, 1), mask)
    got = summary_from_blocks(block_stats(u, mask))
    for k, v in ref.items():
        assert abs(got[k] - v) < 1e-9, k


def test_encode_decode_roundtrip(solved):
    u, mask = solved
    grid, packed = u.tobytes(), np.packbits(mask).tobytes()
    assert len(grid) == GRID_BYTES and len(packed) == MASK_BYTES
    u2, m2 = decode(grid, packed)
    assert np.array_equal(u, u2) and np.array_equal(mask, m2)


def test_invalid_values_are_counted(solved):
    u, mask = solved[0].copy(), solved[1]
    i, j = np.argwhere(mask)[0]
    u[i, j] = np.nan
    b = block_stats(u, mask)
    assert b['n_invalid'].sum() == 1
    assert b['n_invalid'][(i // 32) * 16 + j // 32] == 1


def test_numba_matches_numpy_bitwise():
    pytest.importorskip('numba')
    u0, mask = load_data(REPO / 'data' / 'sample', '694')
    a, na, _, _ = simulate(u0, mask, 'numpy', max_iter=250)
    b, nb, _, _ = simulate(u0, mask, 'numba', max_iter=250)
    assert na == nb and np.array_equal(a, b)
