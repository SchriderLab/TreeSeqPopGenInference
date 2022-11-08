#!/bin/bash
#SBATCH --partition=general
#SBATCH --constraint=rhel8
#SBATCH --job-name=merge_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lobanov@email.unc.edu
#SBATCH --output=/pine/scr/l/o/lobanov/src/selection/slurm_output/merge_data%A.out

python merge_data.py -i sep_seriated_npzs -o compressed_npz_selection.npz
