# SGEMM GPU Kernel Performance Experiments

This directory contains the implementation for replicating the SGEMM regression experiments from the GloNets paper. These experiments demonstrate GloNet's depth resilience and self-regulation capabilities on a real-world regression task.

## 📊 Experiment Overview

**Task**: Predict GPU kernel execution time for matrix multiplication (SGEMM)  
**Dataset**: UCI SGEMM GPU kernel performance dataset  
**Architecture**: Fully connected networks with varying depths (10-1000 blocks)  
**Key Finding**: GloNet maintains stable performance regardless of depth, while ResNet degrades

## 🗂️ Directory Structure

```
resnet_sgemm/
├── _src/
│   ├── models.py              # GloNet, ResNet, MLP implementations
│   ├── train_lib.py           # Training loop and evaluation
│   ├── get_data.py            # Data loading utilities
│   ├── log_training.py        # Metrics logging and visualization
│   ├── utils.py               # Helper functions
│   ├── download_and_prepare_data.py  # Dataset setup
│   └── config.py              # Configuration management
├── data/                      # Dataset storage (auto-created)
├── training_results/          # Experiment outputs
├── train.py                   # Main training script
└── train_script.sh           # Batch experiment runner
```

## 🚀 Quick Start

### 1. Setup Data

Download and prepare the SGEMM dataset:
```bash
cd resnet_sgemm
python _src/download_and_prepare_data.py
```

This will:
- Download the UCI SGEMM dataset (241,600 samples)
- Preprocess and normalize features
- Create train/validation/test splits
- Save processed data in `data/processed/`

### 2. Single Experiment

Run a single training experiment:
```bash
python train.py \
    --model=GloNet50 \
    --epochs=200 \
    --batch_size=1024 \
    --learning_rate=0.01 \
    --features=16
```

### 3. Replicate Paper Results

Run the full experiment suite:
```bash
bash train_script.sh
```

This trains models with depths: 10, 24, 50, 100, 200 blocks for both GloNet and ResNet variants.

## ⚙️ Configuration Options

| Parameter | Description | Default | Paper Values |
|-----------|-------------|---------|--------------|
| `--model` | Architecture type | `ResNet50` | `GloNet10`, `GloNet24`, `GloNet50`, `GloNet100`, `GloNet200` |
| `--epochs` | Training epochs | `10` | `200` |
| `--batch_size` | Batch size | `1024` | `1024` |
| `--learning_rate` | Learning rate | `0.01` | `0.01` |
| `--features` | Hidden layer size | `16` | `16` |
| `--l2_reg_alpha` | L2 regularization | `1e-5` | `1e-5` |
| `--optimizer` | Optimizer type | `ADAM` | `ADAM` |

## 🏗️ Model Architectures

### GloNet Architecture
```python
Input(14) → Dense(16) → [Dense(16) → ReLU] × N → GloNet Layer → Dense(1)
                         ↓         ↓         ↓
                         └─────────┼─────────┘
                                   ↓
                              Sum Layer
```

### ResNet Baseline
```python
Input(14) → Dense(16) → [ResBlock(16)] × N/2 → Dense(1)
```

**Note**: ResNet blocks are halved because each ResNet block contains 2 dense layers.

## 📈 Expected Results

### Training Time Comparison
| Depth | ResNet (seconds) | GloNet (seconds) | Speedup |
|-------|-----------------|------------------|---------|
| 10    | 42              | 27               | 1.6x    |
| 50    | 96              | 55               | 1.7x    |
| 200   | 231             | 121              | 1.9x    |
| 1000  | 675             | 289              | 2.3x    |

### Performance (Test MSE)
| Depth | ResNet | GloNet |
|-------|--------|--------|
| 10    | 0.018  | 0.021  |
| 50    | 0.020  | 0.020  |
| 200   | 0.027  | 0.021  |
| 600   | 0.040  | 0.022  |
| 1000  | 0.048  | 0.022  |

## 🔍 Analysis Features

### Automatic Logging
Each training session generates:
- **Training curves**: Loss progression over epochs
- **L1 norm analysis**: Shows which layers are actively used
- **Parameter distributions**: Weight usage across network depth
- **PDF report**: Comprehensive training summary

### Self-Regulation Visualization
The L1 norm plots demonstrate GloNet's key property:
- **GloNet**: Uses only first ~12 blocks regardless of total depth
- **ResNet**: Uses all blocks, leading to degradation at extreme depths

### Output Structure
```
training_results/
└── [ModelName]/
    └── session_[N]/
        ├── config.json              # Training configuration
        ├── train_state_params.pkl   # Model parameters
        ├── training_metrics.pkl     # Raw metrics data
        ├── training_metrics.png     # Visualizations
        └── training_report.pdf      # Complete report
```

## 🧪 Experiment Variations

### Compare with Baselines
```bash
# Vanilla network (no skip connections)
python train.py --model=MLP --features=16 --num_layers=50

# ResNet with different depths
python train.py --model=ResNet100 --features=16

# GloNet with extreme depth
python train.py --model=GloNet200 --features=16
```

### Ablation Studies
```bash
# Different feature sizes
python train.py --model=GloNet50 --features=32
python train.py --model=GloNet50 --features=64

# Different regularization
python train.py --model=GloNet50 --l2_reg_alpha=1e-4
python train.py --model=GloNet50 --l2_reg_alpha=1e-6
```

## 💡 Key Insights

1. **Training Speed**: GloNet trains ~2x faster than ResNet due to:
   - No batch normalization overhead
   - Simpler gradient flow

2. **Depth Resilience**: GloNet performance plateaus early:
   - Consistent ~0.022 MSE from 10 to 1000 blocks
   - ResNet degrades beyond 200 blocks

3. **Self-Regulation**: L1 norm analysis shows:
   - GloNet uses only first 12-15 blocks
   - Later blocks contribute negligibly
   - Automatic depth selection without NAS

4. **Regularization**: GloNet's aggregation provides implicit regularization:
   - No batch normalization needed
   - Stable training across depths

## 🔧 Troubleshooting

### Data Issues
```bash
# Re-download if data is corrupted
rm -rf data/
python _src/download_and_prepare_data.py
```

### Memory Issues
```bash
# Reduce batch size for large models
python train.py --model=GloNet200 --batch_size=512
```

### Slow Training
```bash
# Enable JIT compilation (should be automatic)
export JAX_ENABLE_X64=False
```

## 📚 References

- [UCI SGEMM Dataset](https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance)
- [Original Paper Results](https://arxiv.org/abs/2311.15947) - Figure 2, Table 1