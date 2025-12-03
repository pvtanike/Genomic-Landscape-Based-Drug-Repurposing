#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --mem=40g
#SBATCH -t 06-00:00:00
#SBATCH --qos=gpu_access
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=start,end
#SBATCH --mail-user=<enter your email here>

module purge
module load anaconda
module activate pytorch-gpu
conda run -n pytorch-gpu python metastastis_GPU_full_extract_09_18_24.py
