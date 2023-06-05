# The Unreasonable Effectiveness of Graph Convolutional Networks on Population Genetics Inference

This repository contains details and instructions for replicating results in the associated manuscript.

All packages associated with running the CNN sections can be found in the conda environment specified at [src/models/cnn_training/keras_env.yaml](src/models/cnn_training/keras_env.yaml)

Requirements for the GCN sections: [requirements.txt](requirements.txt).

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

#### CNN

Msprime output was seriated using the ORTools package (**TODO**: link). The script used to seriate the data can be found at [foo](foo.py). The script was run using the following command:

```bash
foo
```

### Network Training

#### GCN

#### CNN

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

### Plotting and Comparison

---

## Introgression Detection

### Simulation

### Data Preparation

#### GCN

#### CNN

### Network Training

#### GCN

#### CNN

### Plotting and Comparison

---

## Demographic Inference

### Simulation

### Data Preparation

#### GCN

#### CNN

### Network Training

#### GCN

#### CNN

### Plotting and Comparison

