#!/bin/sh
#BSUB -q c02613
#BSUB -J e8                         
#BSUB -n 4                              
#BSUB -R "span[hosts=1]"               
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -W 00:55                          
#BSUB -o e8_%J.out    
#BSUB -e e8_%J.err    

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613                   

nvidia-smi
python e8.py 50            
