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

For demography the sample size is again 50 and we simulate over 5 different demographic parameters detailed in the manuscript.  To generate (100k replicates):

```
python3 src/SLURM/simulate_demography_data.py --odir data/demography
python3 src/SLURM/simulate_demography_data.py --odir data/demography --slurm # if on a cluster
```

To run Relate on each of the output folders:

```
python3 src/SLURM/relate_distributed.py --L 150000 --mu 1.2e-9 --r 1e-8 --N 1000 --n_samples 50 --idir data/demography --odir data/demography_relate
python3 src/SLURM/relate_distributed.py --L 150000 --mu 1.2e-9 --r 1e-8 --N 1000 --n_samples 50 --idir data/demography --odir data/demography_relate --slurm # if on a cluster
```

Here we used the src/SLURM script which calls the src/data/relate.py routine that we used in the recombination case. `simulate_demography_data.py` writes the ms files to individual folders which allows for the work of Relate to be spread over many cpus via sbatch if available.

### Introgression

We simulated a demographic model of introgression between Drosophila Sechellia and D. Simulans using the same routine as in https://github.com/SchriderLab/introNets.  The demographic model parameters were estimated using DADI (https://github.com/SchriderLab/introNets/tree/main/dadiBootstrapCode).  We include the resulting parameters in `params.txt`.  To simulate (with ms):

```
python3 src/data/simulate_msmodified.py --ifile params.txt --odir data/dros/ab --direction ab # use --slurm with this script as well if you have sbatch
python3 src/data/simulate_msmodified.py --ifile params.txt --odir data/dros/ba --direction ba
python3 src/data/simulate_msmodified.py --ifile params.txt --odir data/dros/bi --direction bi
```

This will generate 1000 replicates for each of the 42 parameters that we include from the params.txt file (we throw our parameters that have a log-likelihood > -2000).

Then we can infer the tree sequences using Relate:

```
python3 src/SLURM/relate_distributed.py --idir data/dros/ab --odir data/dros_relate/ab \
          --L 10000 --N 266863 --r 2e-8 --mu 5e-9 --n_samples 34 # use --slurm with this script as well if you have sbatch
python3 src/SLURM/relate_distributed.py --idir data/dros/ba --odir data/dros_relate/ba \
          --L 10000 --N 266863 --r 2e-8 --mu 5e-9 --n_samples 34
python3 src/SLURM/relate_distributed.py --idir data/dros/bi --odir data/dros_relate/bi \
          --L 10000 --N 266863 --r 2e-8 --mu 5e-9 --n_samples 34
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
