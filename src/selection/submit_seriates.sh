#!/bin/bash
#SBATCH --job-name=seriate_all
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lobanov@email.unc.edu
#SBATCH --output=/pine/scr/l/o/lobanov/src/selection/seriate_selection/seriate_selection%A.out
##SBATCH --array=0-3000


python lex_selection_train.py
# python seriate_modular.py \
#     -o /pine/scr/l/o/lobanov/src/selection/sep_seriated_npzs \
#     --dataset-index ${1} \
#     --sample-index $SLURM_ARRAY_TASK_ID

