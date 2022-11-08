#!/bin/bash
#SBATCH --partition=volta-gpu
#SBATCH --constraint=rhel8
#SBATCH --job-name=re_seriate_val
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lobanov@email.unc.edu
#SBATCH --output=/pine/scr/l/o/lobanov/src/selection/seriate_match%A.out



python seriate_file.py
