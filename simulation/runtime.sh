#!/bin/bash
#BSUB -J runtime
#BSUB -q hpc
#BSUB -W 4
#BSUB -R "rusage[mem=256MB]"
#BSUB -o runtime_%J.out
#BSUB -e runtime_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python runtime.py --n 20