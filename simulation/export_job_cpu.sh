#!/bin/sh
# CPU variant of export_job.sh (Numba parallel backend), one array element = one batch of 200 buildings.
# Use this while the GPU path is broken in the 02613 env: its Numba 0.58.1 no longer works with the
# node's CUDA driver (simulation/diagnose_cuda.py: active context reports device 254 of 1, then
# segfault; the course's own e8.py crashes the same way). Results match the NumPy reference bit for bit.
#   mkdir -p logs && bsub < simulation/export_job_cpu.sh
# Expected ~5-10 min per batch (~2 s/building on 4-8 cores; sample averages ~7,300 iterations).
#BSUB -q hpc
#BSUB -J export[1-23]
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 01:00
#BSUB -o logs/export_cpu_%J_%I.out
#BSUB -e logs/export_cpu_%J_%I.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Use only the cores LSF gave us, and a per-task JIT cache so 23 tasks don't race on one directory.
export NUMBA_NUM_THREADS=${LSB_DJOB_NUMPROC:-8}
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_${LSB_JOBID}_${LSB_JOBINDEX}

python simulation/export.py \
    --data-dir /dtu/projects/02613_2025/data/modified_swiss_dwellings/ \
    --out-dir data/landing \
    --backend numba \
    --batch-size 200 \
    --batch-id $((LSB_JOBINDEX - 1))
