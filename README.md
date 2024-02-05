# Tree sequences as a general-purpose tool for population genetic inference

This repository contains details and instructions for replicating results in the associated manuscript.

To install required packages: `pip install -r requirements.txt`

---

## Simulations

Simulations were done using the relevant files in `src/data/`, named with the task being simulated (i.e. `simulate_recombination.py`). These scripts were launched using the `src/SLURM/simulate_demography_data.py` and `src/SLURM/simulate_grids.py` scripts on a SLURM cluster.

## Processing

### GCN

Trees were inferred using RELATE with `src/SLURM/relate_distributed.py` and `src/SLURM/format_relate_distributed.py`.

### CNN

Genotype matrices were formatted and seriated using `src/SLURM/format_genomat_distributed.py`.

Both are written for submission on SLURM clusters.

## Training

Models were trained using the `src/models/train_cnn.py` and `src/models/train_gcn.py` scripts, with relevant layers and functions imported from the `src/models` directory.

## Testing

Models were run on testing data using the relevant scripts in `/src/viz` and resulting final plots for the manuscript are located in `/plots`.