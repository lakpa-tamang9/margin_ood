#!/bin/bash
 
#SBATCH --job-name=gpu_network #Job name
#SBATCH --partition=dgx        #Run on a single gpu
#SBATCH --gres=gpu:4           #Assign GPUs
#SBATCH --time=5:00:00        #Time Limit
#SBATCH --mem=40536
#SBATCH --output=/home/s223127906/devs/margin_ood/oe_hendrycks_etal.log #Standard output and error log
#SBATCH --mail-user=s223127906@deakin.edu.au
#SBATCH --mail-type=ALL
 
python /Users/lakpa/devs/deakin/research/development/margin_ood/train_with_outlier_loader.py 