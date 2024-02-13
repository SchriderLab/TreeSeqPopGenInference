# Tree sequences as a general-purpose tool for population genetic inference

This repository contains details and instructions for replicating results in the associated manuscript.
The code relies mostly on torch and torch-geometric.  We used torch==2.1.1+cu121 but any version compatible with the current torch-geometric should work. 

Other python pre-requisites:
```
pip install h5py matplotlib scipy numpy mpi4py
```
mpi4py and MPI are needed to run the parallel formatting routines for training CNN models that run on sorted (and matched) genotype matrices.  If you only want to use the GCN part of this repo, you won't need them.

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

For the selection sims, you'll need `discoal`: 

```
git clone https://github.com/kr-colab/discoal.git
cd discoal
make discoal
```

Relate is included as a submodule in this repo and should be built like:
```
cd relate/build
cmake ..
make all
```

---
## Simulations and inference with Relate

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

This results are compressed ms text files (*.msOut.gz) which each have 1000 replicates in them and look like:

```
//	12.0006	319.080439014646
segsites: 70
positions: 0.0014 0.0161 0.0231 0.0525 0.0580 ...
0000001000000000000111000001000001100011110000101000000001000000010100
0101001000001000101001000001000001101000000000101000001010000001110100
0000001000001000000111000000010001110000000000000000000001000001110010
0100001000000000000111000001000100001000000100001000000001011110010100
...
```

This is the format of the genotype data our routines expect if you wish to use this with other simulation results or real data.  In the formatting step, we take the parameters to predict from the line with `\\` or ignore them in the case that we are doing classification.  

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
python3 src/SLURM/relate_distributed.py --L 150000 --mu 1.2e-9 --r 1e-8 --N 1000 --n_samples 50 \
                    --idir data/demography --odir data/demography_relate
python3 src/SLURM/relate_distributed.py --L 150000 --mu 1.2e-9 --r 1e-8 --N 1000 --n_samples 50 \
                    --idir data/demography --odir data/demography_relate --slurm # if on a cluster
```

Here we used the src/SLURM script which calls the src/data/relate.py routine that we used in the recombination case. `simulate_demography_data.py` writes the ms files to individual folders which allows for the work of Relate to be spread over many cpus via sbatch if available.

### Introgression

We simulated a demographic model of introgression between Drosophila Sechellia and D. Simulans using the same routine as in https://github.com/SchriderLab/introNets.  The demographic model parameters were estimated using DADI (https://github.com/SchriderLab/introNets/tree/main/dadiBootstrapCode).  We include the resulting parameters in `params.txt`.  To simulate (with ms):

```
python3 src/data/simulate_msmodified.py --ifile params.txt --odir data/dros/ab --direction ab # use --slurm with this script as well if you have sbatch
python3 src/data/simulate_msmodified.py --ifile params.txt --odir data/dros/ba --direction ba
python3 src/data/simulate_msmodified.py --ifile params.txt --odir data/dros/bi --direction bi
```

This will generate 1000 replicates for each of the 42 parameters that we include from the params.txt file (we throw out parameters that have a log-likelihood < -2000).

Then we can infer the tree sequences using Relate:

```
python3 src/SLURM/relate_distributed.py --idir data/dros/ab --odir data/dros_relate/ab \
          --L 10000 --N 266863 --r 2e-8 --mu 5e-9 --n_samples 34 # use --slurm with this script as well if you have sbatch
python3 src/SLURM/relate_distributed.py --idir data/dros/ba --odir data/dros_relate/ba \
          --L 10000 --N 266863 --r 2e-8 --mu 5e-9 --n_samples 34
python3 src/SLURM/relate_distributed.py --idir data/dros/bi --odir data/dros_relate/bi \
          --L 10000 --N 266863 --r 2e-8 --mu 5e-9 --n_samples 34
```

### Selection

We simulated five different scenarios involving selection as detailed in the paper.  The current routine is only compatible with SLURM:

```
# simulates 100k of each category by submitting 10000 jobs with 10 replicates each
python3 src/data/simulate_selection.py --odir /work/users/d/d/ddray/selection_sims
```

It's helpful for this case to chunk the data for pre-processing.  This script splits and copies the ms files in selection_sims to some number of individual folders in `--odir`:

```
python3 src/data/chunk_data.py --idir /work/users/d/d/ddray/selection_sims \
           --odir /work/users/d/d/ddray/selection_sims_chunked --n_per 250 # the number of ms files per folder
```

Finally, we can infer with relate via:

```
python3 src/SLURM/relate_distributed.py --idir /work/users/d/d/ddray/selection_sims_chunked \
                    --L 110000 --mu 1.5e-8 --r 1e-8 --N 10000 --slurm
```

## Processing

### GCN

First we convert the tree sequences output via our Relate routines (which are output as compressed Newick test to *.anc.gz files) to arrays of node features and edge indices and save them to a an hdf5 file.  For recombination: 

```
python3 src/data/format_relate.py --idir data/recom_relate --ms_dir data/recom/ --pop_sizes 50,0 --ofile recom.hdf5
```

Next we split the data into a training and validation set, and trim sequences longer than 128 trees:

```

```

### CNN

## Training

Models were trained using the `src/models/train_cnn.py` and `src/models/train_gcn.py` scripts, with relevant layers and functions imported from the `src/models` directory.

## Testing

Models were run on testing data using the relevant scripts in `/src/viz` and resulting final plots for the manuscript are located in `/plots`.
