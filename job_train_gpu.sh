#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=gpu_ttt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
python3 Ultimate_TTT/training.py -g