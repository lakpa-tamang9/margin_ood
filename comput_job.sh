#!/bin/bash
 
#SBATCH --job-name=macs_ood #Job name
#SBATCH --partition=hpc-dgx-b-1
#SBATCH --gres=gpu:2           #Assign GPUs
#SBATCH --time=24:00:00        #Time Limit
#SBATCH --mem=40536
#SBATCH --output=./MaCS_%x_%j.log 
#SBATCH --error=./error_%x_%j.log
#SBATCH --mail-user=s223127906@deakin.edu.au
#SBATCH --mail-type=ALL

python /home/s223127906/devs/margin_ood/eccv24_final.py