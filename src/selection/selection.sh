#!/bin/bash
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --constraint=rhel8
#SBATCH --job-name=train_resnet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lobanov@email.unc.edu
#SBATCH --output=/pine/scr/l/o/lobanov/src/selection/slurm_output/test_train%A.out



python train_selection_cnn.py --verbose --odir /pine/scr/l/o/lobanov/src/selection/lexEXACT_EXACT/ --lr_decay 0.0001