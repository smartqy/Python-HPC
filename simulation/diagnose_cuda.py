"""Step-by-step CUDA diagnostics for export.py's crash on the DTU GPU nodes.

Runs, in one process, the same steps export.py takes before and during CudaSolver.__init__,
printing after each one, so the first failing step is visible. Run from the repo root:

    python simulation/diagnose_cuda.py /dtu/projects/02613_2025/data/modified_swiss_dwellings/
"""
import faulthandler
import os
import sys
import traceback

faulthandler.enable()
DATA = sys.argv[1] if len(sys.argv) > 1 else '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'


def step(name, fn):
    print(f'--- {name}', flush=True)
    try:
        out = fn()
        print(f'    ok{": " + str(out) if out is not None else ""}', flush=True)
        return out
    except Exception:
        traceback.print_exc()
        print(f'    FAILED at: {name}', flush=True)
        sys.exit(1)


print('python', sys.version.split()[0], '| sys.path[0] =', sys.path[0])
print('env:', {k: v for k, v in os.environ.items() if k.startswith(('CUDA', 'NUMBA', 'LSB_', 'NVIDIA'))})

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402
import numba  # noqa: E402

print('numba', numba.__version__, '| numpy', np.__version__)


def devices():
    return [(i, d.name) for i, d in enumerate(cuda.gpus)]


def active_devnum():
    from numba.cuda.cudadrv.driver import driver
    with driver.get_active_context() as ac:
        return None if not ac else ac.devnum


step('1 list GPUs (cuda.gpus)', devices)
step('2 create context: cuda.current_context()', lambda: str(cuda.current_context()))
step('3 active context devnum vs len(cuda.gpus)', lambda: (active_devnum(), len(cuda.gpus.lst)))
step('4 to_device(np.zeros(1))', lambda: cuda.to_device(np.zeros(1)).shape)
u = np.zeros((514, 514))
step('5 np.load one building', lambda: u.__setitem__((slice(1, -1), slice(1, -1)),
                                                      np.load(os.path.join(DATA, '10000_domain.npy'))))
mask = step('6 np.load mask', lambda: np.load(os.path.join(DATA, '10000_interior.npy')))
step('7 to_device(514x514 float64)', lambda: cuda.to_device(u).shape)
step('8 to_device(mask)', lambda: cuda.to_device(mask).shape)
step('9 import pyarrow.parquet', lambda: __import__('pyarrow.parquet').__version__)
step('10 active context devnum vs len(cuda.gpus)', lambda: (active_devnum(), len(cuda.gpus.lst)))
step('11 to_device again after pyarrow', lambda: cuda.to_device(u).shape)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
K = step('12 import cuda_kernels', lambda: __import__('cuda_kernels'))
step('13 to_device after importing kernels', lambda: cuda.to_device(u).shape)


def launch():
    import math
    a, b, m = cuda.to_device(u), cuda.to_device(u), cuda.to_device(mask)
    bpg = (math.ceil(514 / K.TPB),) * 2
    K.jacobi_kernel[bpg, (K.TPB, K.TPB)](a, b, m)
    d = cuda.device_array(1)
    d.copy_to_device(np.zeros(1))
    K.max_abs_diff_kernel[bpg, (K.TPB, K.TPB)](a, b, d)
    cuda.synchronize()
    return float(d.copy_to_host()[0])


step('14 compile + launch jacobi and residual kernels', launch)


def solver():
    import export as E
    return E.simulate(u, mask, 'cuda', max_iter=300)[1:3]


step('15 export.CudaSolver via simulate(), 300 iterations', solver)
print('ALL STEPS PASSED', flush=True)
