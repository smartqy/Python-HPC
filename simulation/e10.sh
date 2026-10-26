#!/bin/sh
#BSUB -q c02613
#BSUB -J e10                        
#BSUB -n 4                              
#BSUB -R "span[hosts=1]"               
#BSUB -R "rusage[mem=6GB]"
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -W 03:00                          
#BSUB -o e10.out
#BSUB -e e10.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613                   

nvidia-smi
python e10.py 50        
