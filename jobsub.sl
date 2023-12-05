#!/bin/bash

#SBATCH -N 1                   # 1 node
#SBATCH -n 1		       # 1 task		
#SBATCH --cpus-per-task=1      # 1 CPU
#SBATCH --mem=6g	       # 6 GB RAM
#SBATCH --time 144:00:00	       # 144 hour time limit
#SBATCH -p a100-gpu
##SBATCH -p volta-gpu
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --mail-type=end,fail,start
#SBATCH --mail-user=sridhark@email.unc.edu


module add anaconda/2021.11

source activate braintypicality
python -u GAN4Seg_TrainingImpl.py
