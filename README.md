# SODA-Net: Cross-Subject Motor Imagery Decoding via Damped Wave-Equation Prior and Contrastive Alignment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **SODA-Net** (Oscillation Dynamics Alignment Network), a parallel dual-branch framework for cross-subject motor imagery EEG decoding that integrates neurophysiologically motivated damped wave-equation priors with data-driven contrastive learning.

## 🎯 Key Features

- **Dual-Branch Architecture**: AGTCNet (discriminative branch) + MI-ODNet (dynamics-aware branch)
- **Neurophysiologically Motivated**: Damped wave-equation prior encodes temporal continuity, rhythmicity, and finite persistence
- **Contrastive Learning**: Multi-view (CVC) and cross-modal (CMC) alignment for robust feature learning
- **Zero Inference Cost**: MI-ODNet operates only during training, discarded at test time
- **State-of-the-Art Performance**: 80.71%–91.92% accuracy across 5 public MI-EEG datasets
- **Comprehensive Evaluation**: Leave-one-subject-out (LOSO) cross-validation for true cross-subject transfer

## 📊 Performance

| Dataset | Subjects | Classes | Channels | SODA-Net | Best Baseline | Improvement |
|---------|----------|---------|----------|----------|---------------|-------------|
| BNCI2014-001 | 9 | 4 | 22 | **80.71%** | 77.77% | +2.94% |
| BNCI2014-002 | 14 | 2 | 15 | **75.71%** | 63.21% | +12.50% |
| BNCI2014-004 | 9 | 2 | 3 | **83.14%** | 84.25% | -1.11% |
| Lee2019 | 54 | 2 | 62 | **82.44%** | 74.35% | +8.09% |
| Zhou2016 | 4 | 3 | 14 | **91.92%** | 88.70% | +3.22% |

## 🏗️ Architecture

```
                    ┌─────────────────────────────────┐
                    │   Input EEG (C × T)             │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                  │
    ┌─────────▼─────────┐            ┌─────────▼──────────┐
    │   AGTCNet Branch   │            │  MI-ODNet Branch   │
    │  (Discriminative)  │            │ (Dynamics-Aware)   │
    ├────────────────────┤            ├────────────────────┤
    │ • Temporal Conv    │            │ • DCT Transform    │
    │ • Spatial Conv     │            │ • Wave Operator    │
    │ • Graph Attention  │            │ • Modal Evolution  │
    │ • Temporal Context │            │ • IDCT Transform   │
    └─────────┬──────────┘            └─────────┬──────────┘
              │                                  │
              │        ┌────────────────────┐   │
              └───────►│  Contrastive Loss  │◄──┘
                       │   (CVC + CMC)      │
                       └────────────────────┘
                                  │
                       ┌──────────▼──────────┐
                       │  Classification     │
                       │  (AGTCNet only)     │
                       └─────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SODA-Net.git
cd SODA-Net

# Create virtual environment
conda create -n sodanet python=3.8
conda activate sodanet

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=1.8.0
numpy>=1.19.0
scipy>=1.5.0
pandas>=1.1.0
scikit-learn>=0.23.0
tqdm>=4.50.0
mne>=0.23.0
```

### Data Preparation

The framework supports automatic dataset download and preprocessing:

```bash
# Download datasets (automatically triggered on first run)
python download_datasets.py --dataset BNCI2014002
```

Supported datasets:
- `BNCI2014001` (BCI Competition IV 2a, 4-class MI)
- `BNCI2014002` (14 subjects, 2-class MI)
- `BNCI2014004` (9 subjects, 2-class MI, 3 channels)
- `Lee2019` (54 subjects, 2-class MI)
- `Zhou2016` (4 subjects, 3-class MI)

### Training

```bash
# Train SODA-Net with LOSO cross-validation
python MVCNet_LOSO.py [GPU_ID] [LAMBDA_CVC] [LAMBDA_CMC]

# Example: Train on GPU 0 with default hyperparameters
python MVCNet_LOSO.py 0 0.1 0.1

# Example: Train on CPU
python MVCNet_LOSO.py

# Specify dataset via environment variable
MVCNET_DATASET=BNCI2014002 python MVCNet_LOSO.py 0
```

### Hyperparameter Configuration

Key hyperparameters (command-line arguments):

```bash
python MVCNet_LOSO.py GPU_ID λ_CVC λ_CMC λ_channel λ_temporal λ_subject λ_adv λ_mixup
```

| Argument | Description | Default | Range |
|----------|-------------|---------|-------|
| `λ_CVC` | Cross-view contrastive loss weight | 0.1 | [0.01, 1.0] |
| `λ_CMC` | Cross-modal contrastive loss weight | 0.1 | [0.01, 1.0] |
| `λ_channel` | Channel-level contrastive weight | 0.0 | [0.0, 0.5] |
| `λ_temporal` | Temporal-level contrastive weight | 0.0 | [0.0, 0.5] |
| `λ_subject` | Subject-level contrastive weight | 0.0 | [0.0, 0.5] |

### Inference

```bash
# Load trained model and evaluate
from models.AGTCNet import AGTCNet
import torch

# Load checkpoint
checkpoint = torch.load('runs/BNCI2014002/AGTCNet_FULL_S0_seed1.ckpt')
model = AGTCNet(n_classes=2, Chans=15, Samples=2561)
model.load_state_dict(checkpoint['base_network'])
model.eval()

# Predict
with torch.no_grad():
    features, outputs = model(input_tensor)
    predictions = outputs.argmax(dim=1)
```

## 📁 Project Structure

```
SODA-Net/
├── MVCNet_LOSO.py              # Main training script (LOSO cross-validation)
├── MVCNet_Ablation_LOSO.py     # Ablation study script
├── models/
│   ├── AGTCNet.py              # Adaptive Graph-Temporal Context Network
│   └── Conformer.py            # Conformer baseline
├── utils/
│   ├── network.py              # WaveFormer encoder implementation
│   ├── contrastive_loss.py    # NTXent, SupCon losses
│   ├── hierarchical_contrastive.py  # Multi-granularity contrastive loss
│   ├── data_augment.py         # EEG data augmentation
│   ├── dataloader.py           # Dataset loading utilities
│   ├── utils.py                # General utilities
│   └── auto_download_data.py   # Automatic dataset downloader
├── data/                       # Dataset directory (auto-created)
├── runs/                       # Saved models and checkpoints
├── logs/                       # Training logs
└── requirements.txt            # Python dependencies
```

## 🔬 Methodology

### 1. Dual-Branch Framework

**AGTCNet (Discriminative Branch)**:
- Temporal convolution → Spatial convolution
- Graph attention over feature channels (GCAT)
- Temporal context encoder with self-attention (TCE)
- End-to-end classification

**MI-ODNet (Dynamics-Aware Branch)**:
- DCT-II modal decomposition
- Damped wave equation: $\frac{\partial^2 u}{\partial t^2} + \alpha\frac{\partial u}{\partial t} = c^2\frac{\partial^2 u}{\partial x^2}$
- Analytical solution in frequency domain
- WaveFormer architecture with SiLU gating

### 2. Contrastive Alignment

**Cross-View Contrastive (CVC)**:
- Aligns augmented views of the same trial
- Augmentations: temporal cropping, frequency shift, amplitude scaling, Gaussian noise

**Cross-Modal Contrastive (CMC)**:
- Aligns AGTCNet features with MI-ODNet embeddings
- Enforces consistency between discriminative and dynamics-aware representations

### 3. Training Objective

```
L_total = L_cls + λ₁·L_CVC + λ₂·L_CMC + L_aux
```

where:
- `L_cls`: Cross-entropy classification loss
- `L_CVC`: Cross-view contrastive loss (InfoNCE)
- `L_CMC`: Cross-modal contrastive loss (InfoNCE)
- `L_aux`: Mixup data augmentation loss

## 🧪 Ablation Studies

Run ablation experiments to assess component contributions:

```bash
# Ablate MI-ODNet (remove CMC loss)
python MVCNet_Ablation_LOSO.py 0 0.1 0.0

# Ablate cross-view contrast (remove CVC loss)
python MVCNet_Ablation_LOSO.py 0 0.0 0.1

# Ablate GCAT module
MVCNET_DISABLE_GCAT=1 python MVCNet_LOSO.py 0

# Ablate TCE module
MVCNET_DISABLE_TCE=1 python MVCNet_LOSO.py 0
```

### Ablation Results (BNCI2014002)

| Configuration | Accuracy | Δ |
|---------------|----------|---|
| Full Model | 75.71% | - |
| w/o L_CMC | 73.97% | -1.74% |
| w/o L_CVC | 74.11% | -1.60% |
| w/o GCAT | 75.62% | -0.09% |
| w/o TCE | 73.35% | -2.36% |

## 📈 Evaluation Metrics

SODA-Net reports comprehensive metrics:

- **Accuracy**: Classification accuracy
- **AUC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve
- **Kappa**: Cohen's kappa coefficient
- **F1**: F1 score (macro-averaged for multi-class)
- **Recall**: Recall (macro-averaged for multi-class)

## 🔧 Advanced Configuration

### Dataset-Specific Settings

Modify hyperparameters in `MVCNet_LOSO.py`:

```python
# BNCI2014002 (15 channels, 2561 samples @ 512 Hz)
if data_name == 'BNCI2014002':
    paradigm, N, chn, class_num = 'MI', 14, 15, 2
    time_sample_num, sample_rate = 2561, 512
    feature_deep_dim = 160  # AGTCNet: F2=8, temporal_gtc=20
```

### Encoder Options

Switch between encoder architectures:

```python
args.encoder = 'WaveFormer'  # WaveFormer (default), Transformer, Conformer
```

### Data Augmentation

Configure augmentation strategies:

```python
args.aug = True
args.augmethod1 = 'flip'   # Time-domain flipping
args.augmethod2 = 'freq'   # Frequency shifting
args.augmethod3 = 'cr'     # Channel resampling
```

### Early Stopping

Control early stopping behavior:

```bash
# Fast ablation experiments (patience=5)
MVCNET_PATIENCE=5 python MVCNet_LOSO.py 0

# Full training (patience=100, default)
python MVCNet_LOSO.py 0
```

## 📊 Visualization

Generate analysis figures:

```bash
# t-SNE visualization of learned features
python generate_tsne_visualization.py

# Temporal dynamics visualization
python visualize_temporal_enhanced.py

# Wave operator evolution
python generate_wave1d_figures_P1.py

# Spectrogram analysis
python generate_spectrogram2.py
```

## 🎓 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{sodanet2026,
  title={SODA-Net: Cross-Subject Motor Imagery Decoding via Damped Wave-Equation Prior and Contrastive Alignment},
  author={[Authors]},
  journal={IEEE Transactions on [Journal Name]},
  year={2026},
  volume={XX},
  pages={XXX--XXX},
  doi={XXX}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: BCI Competition IV, BNCI Horizon 2020, OpenBMI
- **Baseline Models**: EEGNet, FBCNet, ADFCNN, Conformer
- **Frameworks**: PyTorch, MNE-Python, scikit-learn

## 📧 Contact

- **Author**: Ziwei Wang
- **Email**: [your.email@example.com]
- **Project**: [https://github.com/yourusername/SODA-Net](https://github.com/yourusername/SODA-Net)

## 🐛 Known Issues

- CUDA memory requirements: ~8GB for training (batch_size=64)
- Training time: ~2-4 hours per dataset (LOSO, 5 seeds)
- Early stopping disabled by default for full model training

## 🔄 Updates

- **v1.0.0** (2026-07): Initial release
- Support for 5 public MI-EEG datasets
- LOSO cross-validation framework
- Comprehensive ablation tools

## 📚 References

1. Tangermann et al. (2012). "Review of the BCI Competition IV." *Frontiers in Neuroscience*.
2. Lawhern et al. (2018). "EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces." *Journal of Neural Engineering*.
3. Pfurtscheller & Lopes da Silva (1999). "Event-related EEG/MEG synchronization and desynchronization." *Clinical Neurophysiology*.

---

**Star ⭐ this repository if you find it useful!**
