# MNIST Classification Experiments

This directory implements the MNIST classification experiments from the GloNets paper, demonstrating GloNet's self-regulation capabilities and depth resilience on a classic computer vision task using fully connected architectures.

## 📊 Experiment Overview

**Task**: Handwritten digit classification (10 classes)  
**Dataset**: MNIST (28×28 grayscale images, 60k train + 10k test)  
**Architecture**: Fully connected networks (flattened 784-dimensional input)  
**Key Finding**: GloNet automatically selects optimal depth and maintains performance across varying network depths

## 🗂️ Directory Structure

```
linear_mnist/
├── _src/
│   ├── models.py              # GloNet and MLP implementations
│   ├── train_lib.py           # Training and evaluation logic
│   ├── get_data.py            # MNIST data loading
│   ├── log_training.py        # Metrics tracking and visualization
│   ├── utils.py               # Model creation and utilities
│   └── config.py              # Configuration management
├── training_results/          # Experiment outputs
└── train.py                   # Main training script
```

## 🚀 Quick Start

### Single Experiment
```bash
cd linear_mnist
python train.py \
    --model=GloNet \
    --epochs=10 \
    --features=256 \
    --num_layers=10 \
    --batch_size=256
```

### Compare Architectures
```bash
# GloNet with different depths
python train.py --model=GloNet --num_layers=10 --features=256
python train.py --model=GloNet --num_layers=50 --features=256
python train.py --model=GloNet --num_layers=100 --features=256

# Traditional MLP baseline
python train.py --model=MLP --num_layers=10 --features=256
```

## ⚙️ Configuration Options

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--model` | Architecture | `GloNet` | `GloNet`, `MLP` |
| `--epochs` | Training epochs | `10` | `10-50` |
| `--batch_size` | Batch size | `256` | `128-512` |
| `--learning_rate` | Learning rate | `0.01` | `0.001-0.1` |
| `--features` | Hidden layer size | `256` | `128-512` |
| `--num_layers` | Number of layers | `10` | `5-200` |
| `--optimizer` | Optimizer | `ADAM` | `ADAM`, `SGD` |

## 🏗️ Model Architectures

### GloNet Implementation
```python
class GloNet(nn.Module):
    def __call__(self, x):
        # x shape: (batch_size, 784) - flattened MNIST images
        layers_outputs_sum = jnp.zeros((x.shape[0], self.features))
        
        # Accumulate outputs from all hidden layers
        for layer in range(self.num_layers - 1):
            x = nn.Dense(features=self.features)(x)
            x = nn.relu(x)
            layers_outputs_sum += x  # Key: sum all intermediate outputs
        
        # Final classification from accumulated features
        logits = nn.Dense(features=10)(layers_outputs_sum)  # 10 MNIST classes
        return logits
```

### Traditional MLP Baseline
```python
class MLP(nn.Module):
    def __call__(self, x):
        # Standard feedforward network
        for layer in range(self.num_layers - 1):
            x = nn.Dense(features=self.features)(x)
            x = nn.relu(x)
        
        logits = nn.Dense(features=10)(x)
        return logits
```

## 📈 Expected Results

### Classification Accuracy
| Depth | MLP Accuracy | GloNet Accuracy | GloNet Advantage |
|-------|-------------|----------------|------------------|
| 6     | ~97.5%      | ~98.0%         | Comparable       |
| 24    | ~97.0%      | ~98.0%         | Better           |
| 50    | ~95.0%      | ~98.0%         | Significant      |
| 100   | ~92.0%      | ~98.0%         | Major            |
| 200   | ~85.0%      | ~98.0%         | Dramatic         |

### Training Characteristics
- **GloNet**: Consistent learning curves regardless of depth
- **MLP**: Increasingly slower convergence with depth
- **Performance**: GloNet plateau around 4-8 effective layers

## 🔍 Self-Regulation Analysis

### Layer Usage Visualization
The experiments generate L1 norm plots showing:

```
GloNet (50 layers):
Layer 1: ████████████████████ (High usage)
Layer 2: ███████████████     (High usage)
Layer 3: ██████████          (Medium usage)
Layer 4: ████                (Low usage)
Layer 5-50: ▒               (Negligible usage)
```

### Key Insights
1. **Automatic Depth Selection**: GloNet uses only first 4-8 layers
2. **Consistent Pattern**: Same usage pattern regardless of total depth
3. **No Architecture Search**: Optimal depth emerges automatically

## 🧪 Depth Resilience Experiments

### Systematic Depth Study
```bash
# Test various depths to replicate Figure 5b from paper
for depth in 6 10 24 50 60 80 100 200; do
    python train.py --model=GloNet --num_layers=$depth --epochs=50
    python train.py --model=MLP --num_layers=$depth --epochs=50
done
```

### Expected Depth Resilience
- **GloNet**: Stable ~98% accuracy across all depths
- **MLP**: Degradation starts around 24+ layers
- **Training Speed**: GloNet converges faster at all depths

## 📊 Monitoring and Visualization

### Automatic Outputs
Each experiment generates:
- **Loss curves**: Training and validation loss over epochs
- **Accuracy curves**: Classification accuracy progression
- **L1 norm analysis**: Layer-wise parameter usage
- **Intermediate outputs**: Feature activation patterns

### Visualization Examples
```
training_results/GloNet/session_1/
├── training_metrics.png      # Combined plots
├── training_report.pdf       # Full analysis
└── training_metrics.pkl      # Raw data for analysis
```

## 💡 Key Experiments to Run

### 1. Basic Comparison
```bash
# Compare GloNet vs MLP at moderate depth
python train.py --model=GloNet --num_layers=24 --epochs=20
python train.py --model=MLP --num_layers=24 --epochs=20
```

### 2. Extreme Depth Test
```bash
# Test depth resilience
python train.py --model=GloNet --num_layers=200 --epochs=20
python train.py --model=MLP --num_layers=200 --epochs=20
```

### 3. Self-Regulation Analysis
```bash
# Same features, different depths - should show similar layer usage
python train.py --model=GloNet --num_layers=50 --features=256 --epochs=30
python train.py --model=GloNet --num_layers=100 --features=256 --epochs=30
```

### 4. Feature Size Impact
```bash
# Test with different hidden layer sizes
python train.py --model=GloNet --features=128 --num_layers=50
python train.py --model=GloNet --features=512 --num_layers=50
```

## 🔧 Technical Details

### Data Preprocessing
- **Normalization**: Pixel values scaled to [0, 1]
- **Flattening**: 28×28 images → 784-dimensional vectors
- **Batching**: Configurable batch size (default: 256)
- **Shuffling**: Training data shuffled each epoch

### Training Setup
- **Loss**: Cross-entropy loss
- **Optimizer**: Adam (default) or SGD with momentum
- **Evaluation**: Accuracy on test set after each epoch
- **Early Stopping**: Not implemented (fixed epochs for fair comparison)

### Memory Requirements
| Depth | Parameters (256 features) | Memory Usage |
|-------|---------------------------|--------------|
| 10    | ~2.1M                     | ~100MB       |
| 50    | ~10.3M                    | ~400MB       |
| 200   | ~40.7M                    | ~1.5GB       |

## 🎯 Reproducing Paper Results

### Figure 5a: Layer Usage Pattern
```bash
python train.py --model=GloNet --num_layers=50 --epochs=200 --features=256
# Check L1 norm plot in training_results/GloNet/session_X/training_metrics.png
```

### Figure 5b: Depth Resilience
```bash
# Run systematic depth study
for depth in 6 10 24 50 60 80 100 200; do
    echo "Training depth: $depth"
    python train.py --model=GloNet --num_layers=$depth --epochs=200
    python train.py --model=MLP --num_layers=$depth --epochs=200
done
```

## 🔍 Analysis Tips

### Interpreting L1 Norm Plots
- **High values**: Layer is actively contributing
- **Low values**: Layer is being ignored by the network
- **Pattern**: Should show exponential decay for GloNet

### Performance Metrics
- **Training Loss**: Should decrease smoothly
- **Validation Accuracy**: Should plateau around 98% for GloNet
- **Convergence Speed**: GloNet typically faster than MLP

### Troubleshooting
- **Poor convergence**: Try lower learning rate (0.001)
- **Memory issues**: Reduce batch size or features
- **Slow training**: Ensure JAX is using GPU if available

## 📚 Related Work Comparison

| Architecture | Depth Limit | Performance | Training Speed |
|--------------|-------------|-------------|----------------|
| Standard MLP | ~20 layers  | Degrades    | Slow (deep)    |
| ResNet       | ~50 layers  | Good        | Medium         |
| **GloNet**   | **Unlimited** | **Stable** | **Fast**     |