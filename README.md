# Tree sequences as a general-purpose tool for population genetic inference

This repository contains details and instructions for replicating results in the associated manuscript.
The code relies mostly on torch and torch-geometric.  We used torch==2.1.1+cu121 but any version compatible with the current torch-geometric should work. 

After installing torch (https://pytorch.org/get-started/locally/), we recommend installing torch-geometric (and its pre-requisities) from source:

```
# install the pre-reqs (pyg-lib, torch-cluster, torch-scatter, torch-sparse, and torch-spline-conv)
git clone https://github.com/pyg-team/pyg-lib.git
cd pyg-lib
python3 setup.py install && cd ..

git clone https://github.com/rusty1s/pytorch_cluster.git
cd pytorch_cluster
python3 setup.py install && cd ..

git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
python3 setup.py install && cd ..

git clone https://github.com/rusty1s/pytorch_sparse.git
cd pytorch_sparse
python3 setup.py install && cd ..

git clone https://github.com/rusty1s/pytorch_spline_conv.git
cd pytorch_spline_conv
python3 setup.py install && cd ..

# finally install torch geometric
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric
python3 setup.py install
```
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
