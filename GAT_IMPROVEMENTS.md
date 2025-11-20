# Enhanced Graph Attention Network (GAT) Implementation

## Overview
This document describes the improvements made to the GAT + RNN architecture in this repository. The enhanced model provides better performance through multi-head attention, attention-based pooling, and improved residual connections.

## Key Improvements

### 1. Multi-Head Attention
- **Default**: 4 attention heads (configurable via `--n_heads`)
- **Previous**: Single-head attention (n_heads=1)
- **Benefit**: Allows the model to attend to different aspects of the graph structure simultaneously

### 2. Attention-Based Graph Pooling
- **New Component**: `AttentionPooling` class
- **Mechanism**: Learns to weight nodes when aggregating graph-level features
- **Benefit**: More informative graph representations compared to simple mean pooling

### 3. Enhanced Residual Connections
- **New Component**: `EnhancedGATLayer` with learnable gating
- **Features**:
  - Learnable gating parameter for residual connections
  - ELU activation (often better than ReLU for GATs)
  - Proper layer normalization
- **Benefit**: Better gradient flow and more stable training

### 4. Layer-Wise Attention Aggregation
- **Feature**: Weighted combination of outputs from all GAT layers
- **Mechanism**: Learnable weights that combine layer outputs
- **Benefit**: Captures information from different network depths

## Usage

### Training
To use the enhanced GAT model, add the `--enhanced` flag:

```bash
python3 src/models/train_gcn.py \
    --ifile <training_file> \
    --ifile_val <validation_file> \
    --odir <output_directory> \
    --model gru \
    --enhanced \
    --n_heads 4 \
    --n_gcn_iter 6 \
    --gcn_dim 64 \
    --hidden_dim 256 \
    --n_gru_layers 2 \
    --n_classes 3 \
    --in_dim 6 \
    --n_epochs 100 \
    --lr 0.0001
```

### Evaluation
Similarly, use the `--enhanced` flag for evaluation:

```bash
python3 src/viz/eval_gcn_classification.py \
    --ifile <test_file> \
    --weights <model_weights> \
    --model gru \
    --enhanced \
    --n_heads 4 \
    --n_gcn_iter 6 \
    --gcn_dim 64
```

## Architecture Details

### EnhancedGATSeqClassifier
The enhanced model includes:
1. **Node Embedding**: Linear projection of input features
2. **Multi-Head GAT Layers**: Stack of enhanced GAT layers with residual connections
3. **Layer Aggregation**: Weighted combination of all layer outputs
4. **Attention Pooling**: Attention-based graph-level feature extraction
5. **GRU Processing**: Sequential processing of graph features
6. **Output MLP**: Final classification/regression head

### Key Parameters
- `n_heads`: Number of attention heads (default: 4, recommended: 4-8)
- `gcn_dim`: GAT layer dimension (should be divisible by n_heads)
- `n_gcn_iter`: Number of GAT layers
- `use_attention_pooling`: Enable attention-based pooling (default: True)
- `use_layer_aggregation`: Enable layer-wise aggregation (default: True)

## Performance Expectations
The enhanced model should provide:
- **Better accuracy** through multi-head attention
- **More stable training** with improved residual connections
- **Richer representations** via attention pooling
- **Better generalization** through layer aggregation

## Backward Compatibility
The original `GATSeqClassifier` remains unchanged and can still be used by omitting the `--enhanced` flag. This ensures backward compatibility with existing trained models and scripts.

## Technical Notes
- The enhanced model uses ELU activation instead of ReLU for better gradient flow
- Learnable gating in residual connections helps balance between skip connections and new features
- Attention pooling learns which nodes are most important for the task
- Layer aggregation allows the model to use information from all network depths

