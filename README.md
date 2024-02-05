# Tree sequences as a general-purpose tool for population genetic inference

This repository contains details and instructions for replicating results in the associated manuscript.

To install required packages: `pip install -r requirements.txt`

---

## Table of Contents
- [The Unreasonable Effectiveness of Graph Convolutional Networks on Population Genetics Inference](#the-unreasonable-effectiveness-of-graph-convolutional-networks-on-population-genetics-inference)
  - [Table of Contents](#table-of-contents)
  - [Historical Recombination](#historical-recombination)
    - [Simulation](#simulation)
    - [Data Preparation](#data-preparation)
      - [GCN](#gcn)
      - [CNN](#cnn)
    - [Network Training](#network-training)
      - [GCN](#gcn-1)
      - [CNN](#cnn-1)
    - [Plotting and Comparison](#plotting-and-comparison)
  - [Selection Detection](#selection-detection)
    - [Simulation](#simulation-1)
    - [Data Preparation](#data-preparation-1)
      - [GCN](#gcn-2)
      - [CNN](#cnn-2)
    - [Network Training](#network-training-1)
      - [GCN](#gcn-3)
      - [CNN](#cnn-3)
    - [Plotting and Comparison](#plotting-and-comparison-1)
  - [Introgression Detection](#introgression-detection)
    - [Simulation](#simulation-2)
    - [Data Preparation](#data-preparation-2)
      - [GCN](#gcn-4)
      - [CNN](#cnn-4)
    - [Network Training](#network-training-2)
      - [GCN](#gcn-5)
      - [CNN](#cnn-5)
    - [Plotting and Comparison](#plotting-and-comparison-2)
  - [Demographic Inference](#demographic-inference)
    - [Simulation](#simulation-3)
    - [Data Preparation](#data-preparation-3)
      - [GCN](#gcn-6)
      - [CNN](#cnn-6)
    - [Network Training](#network-training-3)
      - [GCN](#gcn-7)
      - [CNN](#cnn-7)
    - [Plotting and Comparison](#plotting-and-comparison-3)

---

## Historical Recombination

### Simulation

Data for the historical recombination experiments was generated using the [msprime](https://msprime.readthedocs.io/en/stable/) python package. The script used to generate the data can be found at [src/data/simulate_recombination.py](src/data/simulate_recombination.py). The script was run using the following command:

```bash
foo
```

### Data Preparation

#### GCN

*Dylan

#### CNN

Msprime output was seriated using the [ORTools package](https://developers.google.com/optimization).

### Network Training

#### GCN

#### CNN

The script used to train the CNN models for rho estimation can be found at [src/models/cnn_training/train_rho_cnn.py](src/models/cnn_training/train_rho_cnn.py).

```bash
$ python rho_cnn_keras.py -h
usage: rho_cnn_keras.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--net {2d,1d}]
                        [--nolog] [--in-train IN_TRAIN] [--in-val IN_VAL]

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --net {2d,1d}
  --nolog               Don't log transform data
  --in-train IN_TRAIN   Path to training data hdf5 file
  --in-val IN_VAL       Path to validation datahdf5 file
```

Example usage:

```bash
python rho_cnn_keras.py \
            --in-train four_problems/recombination/n256_cosine.hdf5 \
            --in-val four_problems/recombination/n256_cosine_val.hdf5 \
            --net 1d
```

### Plotting and Comparison

CSV files containing the predicted values on the evaluation set for both GCN and CNN models were plotted and compared using the Jupyter notebook at [src/models/cnn_training/plot_rho_preds.ipynb](src/models/cnn_training/plot_rho_preds.ipynb).

---

## Selection Detection

### Simulation

### Data Preparation

#### GCN

#### CNN

### Network Training

#### GCN

#### CNN

The script used to train the CNN models for selection can be found at [src/models/cnn_training/selection_cnn_keras.py](src/models/cnn_training/selection_cnn_keras.py).

```bash
$ python selection_cnn_keras.py -h
usage: selection_cnn_keras.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                              [--in-train IN_TRAIN] [--in-val IN_VAL] [--out-prefix OUT_PREFIX]
                              [--net {1d,2d}]

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --in-train IN_TRAIN   Path to training data hdf5 file
  --in-val IN_VAL       Path to validation data hdf5 file
  --out-prefix OUT_PREFIX
                        Prefix for output files
  --net {1d,2d}
```

Example usage:

```bash
python selection_cnn_keras.py \
            --in-train four_problems/selection/n256_cosine.hdf5 \
            --in-val four_problems/selection/n256_cosine_val.hdf5 \
            --net 1d \
            --out-prefix n256_cosine_1d
```

### Plotting and Comparison

CSV files containing the predicted values on the evaluation set for both GCN and CNN models were plotted and compared using the Jupyter notebook at [src/models/cnn_training/plot_selection_preds.ipynb](src/models/cnn_training/plot_selection_preds.ipynb).

---

## Introgression Detection

### Simulation

### Data Preparation

#### GCN

#### CNN

### Network Training

#### GCN

#### CNN

The script used to train the CNN models for selection can be found at [src/models/cnn_training/intro_cnn_keras.py](src/models/cnn_training/intro_cnn_keras.py).

```bash
$ python intro_cnn_keras.py -h
usage: intro_cnn_keras.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--in-train IN_TRAIN]
                          [--in-val IN_VAL] [--out-prefix OUT_PREFIX] [--net {1d,2d}]
                          [--encoding {01,0255,neg11}]

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --in-train IN_TRAIN   Path to training data hdf5 file
  --in-val IN_VAL       Path to validation data hdf5 file
  --out-prefix OUT_PREFIX
                        Prefix for output files
  --net {1d,2d}
  --encoding {01,0255,neg11}
                        Encoding of data
```

Example usage:

```bash
python intro_cnn_keras.py \
                --in-train four_problems/dros/1_3/n256.hdf5 \
                --in-val four_problems/dros/1_3/n256_val.hdf5 \
                --out-prefix intro/1_3_n256 \
                --net 1d
```

### Plotting and Comparison

CSV files containing the predicted values on the evaluation set for both GCN and CNN models were plotted and compared using the Jupyter notebook at [src/models/cnn_training/plot_intro_preds.ipynb](src/models/cnn_training/plot_intro_preds.ipynb).

---

## Demographic Inference

### Simulation

### Data Preparation

#### GCN

#### CNN

### Network Training

#### GCN

#### CNN

The script used to train the CNN models for selection can be found at [src/models/cnn_training/intro_cnn_keras.py](src/models/cnn_training/intro_cnn_keras.py).

```bash
$ python demo_cnn_keras.py -h
usage: demo_cnn_keras.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                         [--conv-blocks CONV_BLOCKS] [--net {2d,1d}]
                         [--encoding {01,0255,neg11}] [--in-train IN_TRAIN] [--in-val IN_VAL]

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --conv-blocks CONV_BLOCKS
                        Number of convolutional blocks
  --net {2d,1d}
  --encoding {01,0255,neg11}
                        Encoding of the input data
  --in-train IN_TRAIN   Path to the training data hdf5 file
  --in-val IN_VAL       Path to the validation data hdf5 file
```

Example usage:

```bash
python demo_cnn_keras.py \
                    --in-train four_problems/demography/n256_cosine.hdf5 \
                    --in-val four_problems/demography/n256_cosine_val.hdf5 \
                    --conv-blocks 3 \
                    --net 1d \
                    --encoding 01
```

### Plotting and Comparison

CSV files containing the predicted values on the evaluation set for both GCN and CNN models were plotted and compared using the Jupyter notebook at [src/models/cnn_training/plot_demo_preds.ipynb](src/models/cnn_training/plot_demo_preds.ipynb).
