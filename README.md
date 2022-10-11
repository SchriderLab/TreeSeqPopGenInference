# Comparing sequence-based pop gen inference methods with tree-based methods

### Goals: 
- Replicate [Flagel et al 2019](https://academic.oup.com/mbe/article/36/2/220/5229930) using both sequence and tree inference methods
  - [Github link here](https://github.com/flag0010/pop_gen_cnn/tree/master)
- Use interpretable machine learning approaches to better understand important features to each method type

---

## Types of approaches

### Sequence-based approaches

- Use multiple sequence alignments as input to CNN
- Architectures are already built but should be tested to see if we can improve them

### Treeseq-based approaches

- Summary statistic based
  - Phylogenetic distance measures (RF, NNI; see [RF pilot study here](/pilot/pilot.ipynb))
  - Branch length distributions
  
- Direct learning approach
  - Using graph convolutional networks (GCNs) to learn directly from trees
  - Need some method of learning from the entire sequence as well, 1DCNN on extracted features?
  
---

## Problems defined in Flagel et al.

### 1. Estimate $\theta$ 

- Sort chromosomes by genetic similarity
- 1DCNN

### 2. Detect introgression
All training and validation should be done with: 
```
/pine/scr/d/d/ddray/intro_trees.hdf5
/pine/scr/d/d/ddray/intro_trees_val.hdf5
```
respectively.

Leaderboard:
|Model   |NLLLoss   |Validation accuracy   |n_gcn_layers   |tree_sequence_length   |n_per_class_batch   |gru_hidden_dim   |n_parameters   |lr, decay, steps_per_epoch   |sampling
|---|---|---|---|---|---|---|---|---|---|
|GATConvRNN   |0.327077   |0.86688   |12   |54   |16   |128    |1,164,549    |1e-5, 0.98, 1000    |sequential
|GATConvRNN   |**0.29515**   |0.88429   |16   |54   |16   |128    |1,164,549    |1e-5, 0.98, 1000    |sequential

|Model   |NLLLoss   |Validation accuracy   |n_cnn_layers   |tree_sequence_length   |n_per_class_batch   |n_parameters   |lr,  steps_per_epoch   |sampling |seriated |metric |data
|---|---|---|---|---|---|---|---|---|---|---|---|
|PyTorch CNN |0.311942  |0.875257 |5  |2000  |16  |8,292,611   |1e-6,3000   |sequential |True |cosine |intro_trees
|ResNet|**0.308413** |0.881486 |20 |2000 |16  |11,174,915 |1e-5,3000 |sequential |True |cosine |intro_trees
|Keras CNN |0.2991 |0.8955 |5 |1201 |256 |5,032,067 |1e-3,870 |sequential |False |N/A |big_sim

- D. simulans and D. sechelia
- Would we want to still compare to FILET?

### 3. Rho estimation

- Uses read fraction per site rather than allelic assignments
- Similarly to above, do we compare to LDHat?
- Phased vs unphased? 

### 4. Positive selection

- 5-class problem (hard sweep, hard linked, soft sweep, soft linked, neutral)
- Read counts here? Or just alignments?

### 5. Demographic inference

- Infer pop size and timings of a three-pop model

