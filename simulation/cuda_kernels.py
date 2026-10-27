"""Numba CUDA kernels for export.py's CudaSolver. Imported lazily so CPU-only machines never
touch numba.cuda. Numba only - see CudaSolver for why CuPy must not be used alongside."""
import numpy as np
from numba import cuda

TPB = 16  # threads per block along each axis
# Shared-array sizes must be compile-time constants; a global is frozen as a literal, while the
# expression TPB * TPB inside the kernel is typed as a runtime int64 and rejected by numba-cuda.
BLOCK_THREADS = TPB * TPB


@cuda.jit
def jacobi_kernel(u_old, u_new, interior_mask):
    i, j = cuda.grid(2)
    n, m = u_old.shape
    if 0 < i < n - 1 and 0 < j < m - 1 and interior_mask[i - 1, j - 1]:
        u_new[i, j] = 0.25 * (u_old[i, j - 1] + u_old[i, j + 1] + u_old[i - 1, j] + u_old[i + 1, j])


@cuda.jit
def max_abs_diff_kernel(a, b, out):
    # Max-reduce |a - b| within the block in shared memory, then one atomic per block.
    buf = cuda.shared.array(BLOCK_THREADS, np.float64)
    i, j = cuda.grid(2)
    t = cuda.threadIdx.x * TPB + cuda.threadIdx.y
    d = 0.0
    if i < a.shape[0] and j < a.shape[1]:
        d = abs(a[i, j] - b[i, j])
    buf[t] = d
    cuda.syncthreads()
    s = BLOCK_THREADS // 2
    while s > 0:
        if t < s and buf[t + s] > buf[t]:
            buf[t] = buf[t + s]
        cuda.syncthreads()
        s //= 2
    if t == 0:
        cuda.atomic.max(out, 0, buf[0])
