# Comparing sequence-based pop gen inference methods with tree-based methods

### Goals: 
- Replicate [Flagel et al 2019](https://academic.oup.com/mbe/article/36/2/220/5229930) using both sequence and tree inference methods
- Use interpretable machine learning approaches to better understand important features to each method type

---

## Types of approaches

### Sequence-based approaches

- Use multiple sequence alignments as input to CNN
- Architectures are already built but should be tested to see if we can improve them

### Treeseq-based approaches

- Summary statistic based
  - Phylogenetic distance measures (RF, NNI; see [RF pilot study here](/pilot/pilot.ipynb)
  - Branch length distributions
  
- Direct learning approach
  - Using graph convolutional networks (GCNs) to learn directly from trees
  - Need some method of learning from the entire sequence as well, 1DCNN on extracted features?
  
---

## Problems defined in Flagel et al.

### Estimate $\theta$ 

- Sort chromosomes by genetic similarity
- 1DCNN

### Detect introgression

- D. simulans and D. sechelia
- Would we want to still compare to FILET?

### Rho estimation

- Uses read fraction per site rather than allelic assignments
- Similarly to above, do we compare to LDHat?
- Phased vs unphased? 

### Positive selection

- 5-class problem (hard sweep, hard linked, soft sweep, soft linked, neutral)
- Read counts here? Or just alignments?

### Demographic inference

- Infer pop size and timings of a three-pop model

