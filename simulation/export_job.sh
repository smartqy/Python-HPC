#!/bin/sh
# Full-dataset export on DTU HPC as an LSF job array: one array element = one batch of 200 buildings.
# 4571 buildings / 200 = 23 batches. Each element writes data/landing/*/batch_id=<n>/ and can be
# re-submitted on its own (e.g. bsub -J "export[7]") because a batch is overwritten in place.
#   bash simulation/setup_gpu_env.sh   # once
#   mkdir -p logs && bsub < simulation/export_job.sh
# Expected ~1.5-3 min per batch: e8 took 1.13 s/building at a fixed 20,000 iterations; with early
# stopping the sample averages ~7,300 iterations. -W 00:30 leaves a 10x margin.
#BSUB -q gpua10
#BSUB -J export[1-23]
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o logs/export_%J_%I.out
#BSUB -e logs/export_%J_%I.err

# Private venv from simulation/setup_gpu_env.sh: the course env's Numba 0.58.1 no longer works
# with the nodes' CUDA driver (simulation/diagnose_cuda.py).
source "${GPU_VENV:-$HOME/venv-gpu}/bin/activate"
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_${LSB_JOBID}_${LSB_JOBINDEX}

python simulation/export.py \
    --data-dir /dtu/projects/02613_2025/data/modified_swiss_dwellings/ \
    --out-dir data/landing \
    --backend cuda \
    --batch-size 200 \
    --batch-id $((LSB_JOBINDEX - 1))
