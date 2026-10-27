#!/bin/sh
#BSUB -q c02613
#BSUB -J e9                        
#BSUB -n 4                              
#BSUB -R "span[hosts=1]"               
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -W 00:55                          
#BSUB -o e9_%J.out
#BSUB -e e9_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613                   

nvidia-smi
python e9.py 50            
