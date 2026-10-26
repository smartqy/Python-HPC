#!/bin/sh
# Full-dataset export on DTU HPC as an LSF job array: one array element = one batch of 200 buildings.
# 4571 buildings / 200 = 23 batches. Each element writes data/landing/*/batch_id=<n>/ and can be
# re-submitted on its own (e.g. bsub -J "export[7]") because a batch is overwritten in place.
#   bsub < simulation/export_job.sh
#BSUB -q c02613
#BSUB -J export[1-23]
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o logs/export_%J_%I.out
#BSUB -e logs/export_%J_%I.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulation/export.py \
    --data-dir /dtu/projects/02613_2025/data/modified_swiss_dwellings/ \
    --out-dir data/landing \
    --backend cuda \
    --batch-size 200 \
    --batch-id $((LSB_JOBINDEX - 1))
