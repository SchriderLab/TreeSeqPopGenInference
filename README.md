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
We use various genetic simulators to accomplish the experiments.  We include the source for ms twice in two different folders as it is meant to be built two different ways: 
```
cd msdir/
gcc -O3 -o ms ms.c streec.c rand1.c -lm

cd ../msdir_rand2/
gcc -O3 -o ms ms.c streec.c rand2.c -lm
```
This is done as our simulation commands for ms that include no introgression between populations produce a SegFault when ms is built with rand1.c for whatever reason.

Relate is included as a submodule in this repo and should be built like:
```
cd relate/build
cmake ..
make all
```

---
## Simulations

### Recombination

We simulate replicates with a present day sample size of 50 and with N (the effective population size) = 14714, mu (mutation rate) 1.5e-8, and a variable recombination rate to predict from inferred trees or the binary genotype matrix.  The ms simulation commands we used are included in the repo as `recom.runms.sh`.  To run them all (~225k replicates) locally:

```
# will take a while
python3 src/data/simulate_recombination.py --odir data/recom
```

or with sbatch on a cluster with SLURM:

```
python3 src/data/simulate_recombination.py --odir data/recom --slurm
```

We can now infer the tree sequences for each replicate by calling Relate via Python like:

```
python3 src/data/relate.py --L 20000 --mu 1.5e-8 --r 1e-7 --N 14714 --idir data/recom/ --odir data/recom_relate --n_samples 50
```

### Demography

For demography the sample size is again 50 and we simulate over 5 different demographic parameters detailed in the manuscript.  To generate locally:

```
python3 src/
```

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
