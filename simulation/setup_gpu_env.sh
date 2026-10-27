#!/bin/bash
# One-off: build a private venv with a current Numba CUDA stack for export_job.sh.
# The course env (02613, Numba 0.58.1 + CUDA 11.8) no longer works with the nodes' CUDA driver
# (see diagnose_cuda.py). numba-cuda[cu12] ships its own CUDA 12 runtime libraries via pip and
# only needs a driver that supports CUDA >= 12.0. Run from the repo root on a GPU node:
#   bash simulation/setup_gpu_env.sh            # creates ~/venv-gpu (override with GPU_VENV=...)
set -euo pipefail
VENV=${GPU_VENV:-$HOME/venv-gpu}

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613            # only used for its Python 3.11 interpreter
python -m venv "$VENV"
conda deactivate

source "$VENV/bin/activate"
pip install --upgrade pip
# Same pyarrow as the local/Docker readers, so there is no writer/reader version skew.
pip install "numpy<2.3" "numba==0.61.2" "numba-cuda[cu12]" "pyarrow==20.0.0"

nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import numba, numba.cuda; print('numba', numba.__version__); numba.cuda.detect()"
echo "venv ready: $VENV"
