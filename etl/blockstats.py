"""Turn one simulated temperature grid into 32x32-cell block statistics.

Every measure is additive (counts, sums, sums of squares) so building-level statistics can be
rebuilt exactly from blocks: mean = sum/n, var = sumsq/n - mean^2, pct = 100*count/n.
Block-level mean/min/max are kept for analysis but are not used for aggregation.
"""
import numpy as np

SIZE = 512
BLOCK = 32
NB = SIZE // BLOCK          # 16 blocks per side, 256 per building
GRID_BYTES = SIZE * SIZE * 8
MASK_BYTES = SIZE * SIZE // 8

# Maximum principle: interior temperatures lie between the coldest and hottest fixed cell.
# That is 0, not 5: in some floor plans interior cells touch outside cells (fixed at 0 degC)
# rather than a wall, so a few cells legitimately fall below 5 degC (reported as a WARN check).
T_MIN, T_MAX, T_EPS = 0.0, 25.0, 1e-9
T_WALL_MIN = 5.0

BLOCK_COLUMNS = ['block_y', 'block_x', 'interior_cells', 'sum_temp', 'sum_sq_temp',
                 'mean_temp', 'min_temp', 'max_temp', 'n_above_18', 'n_below_15', 'n_invalid']


def decode(grid_bytes, mask_bytes):
    u = np.frombuffer(grid_bytes, dtype=np.float64).reshape(SIZE, SIZE)
    mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))[:SIZE * SIZE].reshape(SIZE, SIZE).astype(bool)
    return u, mask


def block_stats(u, mask):
    """Return a dict of 256-long arrays, one entry per block, in (block_y, block_x) order."""
    x = u.reshape(NB, BLOCK, NB, BLOCK).swapaxes(1, 2)       # (16, 16, 32, 32)
    m = mask.reshape(NB, BLOCK, NB, BLOCK).swapaxes(1, 2)
    ax = (2, 3)
    xm = np.where(m, x, 0.0)
    n = m.sum(ax)
    s = xm.sum(ax)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(n > 0, s / n, np.nan)
    invalid = m & ~((x >= T_MIN - T_EPS) & (x <= T_MAX + T_EPS))   # also catches NaN
    by, bx = np.indices((NB, NB))
    out = {
        'block_y': by, 'block_x': bx,
        'interior_cells': n,
        'sum_temp': s,
        'sum_sq_temp': (xm * xm).sum(ax),
        'mean_temp': mean,
        'min_temp': np.where(n > 0, np.where(m, x, np.inf).min(ax), np.nan),
        'max_temp': np.where(n > 0, np.where(m, x, -np.inf).max(ax), np.nan),
        'n_above_18': (m & (x > 18)).sum(ax),
        'n_below_15': (m & (x < 15)).sum(ax),
        'n_invalid': invalid.sum(ax),
    }
    return {k: v.ravel() for k, v in out.items()}
