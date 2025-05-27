# GloNets: Globally Connected Neural Networks - Replication

This repository contains the implementation and experimental code to replicate the results from the paper **"GloNets: Globally Connected Neural Networks"** by Di Cecco et al.

## 📖 Paper Summary

GloNets introduce a novel architecture that overcomes depth-related performance degradation in deep neural networks. Unlike ResNet which uses skip connections between blocks, GloNet connects all intermediate layer outputs directly to the final classification/regression head through a simple summation layer.

**Key Benefits:**
- 🚀 **Faster Training**: Trains in ~50% less time than ResNet
- 📊 **Self-Regulation**: Automatically determines effective network depth
- 🎯 **Depth Resilience**: Performance remains stable regardless of network depth
- 🔍 **Explainable-by-Design**: Linear aggregation enables feature importance analysis

## 🏗️ Repository Structure

```
├── linear_mnist/          # MNIST classification with fully connected networks
├── resnet_cifar10/        # CIFAR-10 experiments with convolutional architectures
├── resnet_imagenette/     # Imagenette experiments (subset of ImageNet)
├── resnet_sgemm/          # SGEMM GPU kernel performance regression
├── environment.yaml       # Conda environment specification
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate initialization_v1
```

Or install dependencies with pip:
```bash
pip install -r requirements.txt
```

### 2. Run Experiments

Each experiment directory contains a `train.py` script with configurable parameters:

```bash
# MNIST Classification
cd linear_mnist
python train.py --model=GloNet --epochs=10 --features=256

# CIFAR-10 Classification  
cd resnet_cifar10
python train.py --model=GloNet20 --epochs=200 --batch_size=256

# SGEMM Regression
cd resnet_sgemm
python train.py --model=GloNet50 --epochs=200 --learning_rate=0.01

# Imagenette Classification
cd resnet_imagenette  
python train.py --model=GloNet34 --epochs=10 --batch_size=64
```

### 3. Batch Experiments

Use the provided shell scripts to run multiple experiments:
```bash
cd resnet_sgemm
bash train_script.sh  # Runs GloNet10, GloNet24, GloNet50, etc.
```

## 🧪 Experiments Overview

| Experiment | Dataset | Task | Models | Key Findings |
|------------|---------|------|--------|--------------|
| **SGEMM** | GPU Kernel Performance | Regression | GloNet vs ResNet (10-1000 blocks) | GloNet maintains performance at extreme depths |
| **MNIST** | Handwritten Digits | Classification | GloNet vs MLP vs ResNet | Self-regulation capabilities demonstrated |
| **CIFAR-10** | Natural Images | Classification | GloNet20 vs ResNet20 | Comparable performance, faster training |
| **Imagenette** | ImageNet Subset | Classification | GloNet vs ResNet variants | Scalability to larger images |

## 🏛️ Architecture Details

### Traditional Network
```
Input → Block₀ → Block₁ → ... → Block_L → Head → Output
```

### GloNet Architecture  
```
Input → Block₀ → Block₁ → ... → Block_L
         ↓        ↓              ↓
         └────────┼──────────────┘
                  ↓
               GloNet Layer (Σ)
                  ↓
                Head → Output
```

The GloNet layer computes: `x_{L+1} = Σᵢ₌₁ᴸ xᵢ`

## 📊 Key Results

- **Training Speed**: GloNet trains 2x faster than ResNet (no batch normalization needed)
- **Depth Resilience**: Performance stable from 10 to 1000+ blocks
- **Self-Regulation**: Network automatically uses only first ~10-15% of blocks
- **Comparable Accuracy**: Matches ResNet performance across all tasks

## 🛠️ Implementation Notes

### GloNet Layer Implementation
```python
class GloNet(nn.Module):
    def __call__(self, x):
        layers_outputs_sum = jnp.zeros((x.shape[0], self.features))
        
        for layer in range(self.num_layers - 1):
            x = nn.Dense(features=self.features)(x)
            x = nn.relu(x)
            layers_outputs_sum += x  # Accumulate all layer outputs
        
        # Final prediction from accumulated features
        logits = nn.Dense(features=self.output_dim)(layers_outputs_sum)
        return logits
```

### Key Differences from ResNet
- **No skip connections between blocks** (only to final layer)
- **No batch normalization** (GloNet provides its own regularization)
- **Simple summation** instead of element-wise addition
- **All blocks contribute equally** to final prediction

## 📈 Monitoring Training

Each experiment automatically generates:
- **Training metrics plots** (loss, accuracy curves)
- **L1 norm analysis** (showing which layers are used)
- **PDF reports** with configuration and results
- **Saved model checkpoints** for analysis

Results are saved in `training_results/[model]/session_[id]/`

## 🔬 Reproducing Paper Results

The experiments are designed to replicate the key findings from the paper:

1. **Figure 2 (SGEMM)**: Run `resnet_sgemm/train_script.sh`
2. **Figure 5 (MNIST)**: Run various depths in `linear_mnist/`
3. **CIFAR-10 experiments**: Use `resnet_cifar10/train_script.sh`
4. **Self-regulation analysis**: Check L1 norm plots in training outputs

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 Citation

If you use this code, please cite the original paper:
```bibtex
@article{dicecco2023glonets,
  title={GloNets: Globally Connected Neural Networks},
  author={Di Cecco, Antonio and Metta, Carlo and Fantozzi, Marco and Morandin, Francesco and Parton, Maurizio},
  journal={arXiv preprint arXiv:2311.15947},
  year={2023}
}
```

## 📞 Contact

For questions about the implementation, please open an issue in this repository.

## 🔗 Links

- [Original Paper](https://arxiv.org/abs/2311.15947)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)