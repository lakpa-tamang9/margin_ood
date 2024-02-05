#!/bin/bash
 
#SBATCH --job-name=oe_ood #Job name
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-dev-1080
#SBATCH --gres=gpu:2           #Assign GPUs
#SBATCH --time=24:00:00        #Time Limit
#SBATCH --mem=40536
#SBATCH --output=./MaPS_%x_%j.log 
#SBATCH --error=./error_%x_%j.log
#SBATCH --mail-user=s223127906@deakin.edu.au
#SBATCH --mail-type=ALL

module load cuda/11.4

source /home/s223127906/.conda/etc/profile.d/conda.sh
conda activate /home/s223127906/.conda/envs/cssr

module load python

cd /home/s223127906/deakin_devs/margin_ood
srun python mlm_none_train_oe_margin_finetune.py -d cifar10 --detectors msp
