#!/bin/bash
 
#SBATCH --job-name=macs_ood #Job name
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1           #Assign GPUs
#SBATCH --time=24:00:00        #Time Limit
#SBATCH --output=./MaCS_%x_%j.log 
#SBATCH --error=./error_%x_%j.log
#SBATCH --mail-user=s223127906@deakin.edu.au
#SBATCH --mail-type=ALL

python /home/s223127906/devs/margin_ood/eccv24_final.py