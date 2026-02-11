# Nuisance Aware Representation Refinement (NARR) for Robust Deepfake Detection

## Overview

This repository implements **NARR (Nuisance Aware Representation Refinement)**, a novel approach for deepfake detection that addresses the challenge of robustness under various corruptions and cross-dataset scenarios.

NARR learns to identify and mitigate nuisance factors in feature representations while preserving discriminative information for deepfake detection. The method combines multi-scale nuisance estimation, adaptive feature refinement, and domain adversarial training to achieve state-of-the-art performance on corrupted and out-of-distribution data.

## Key Features

- **Multi-Scale Nuisance Estimation**: Uses dilated convolutions to capture nuisance at different scales
- **Adaptive Feature Refinement**: Learnable gating mechanisms for channel-wise and spatial feature modulation
- **Domain Adversarial Training**: Improves generalization across different data distributions
- **Contrastive Invariance Learning**: Ensures robustness to image corruptions
- **Transformer-based Classification**: Token-based processing for better feature aggregation

## Model Architecture

![NARR Design](NARR%20design.jp)

*The NARR architecture consists of a CNN backbone, multi-scale nuisance estimator, adaptive gates for feature refinement, and a transformer-based classifier.*

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/narr-deepfake-detection.git
cd narr-deepfake-detection

# Install dependencies
pip install torch torchvision torchaudio
pip install tqdm scikit-learn pillow
```

## Dataset Preparation

The code expects datasets in the following structure:

```
FFPP_CViT/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/

DFDC/
└── validation/
    ├── real/
    └── fake/

CelebDF_images/
└── test/
    ├── real/
    └── fake/
```

## Training

Run the training notebook:

```bash
jupyter notebook NARR.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model training with multi-objective loss
- Validation and checkpoint saving
- Comprehensive evaluation on multiple test sets

## Evaluation

The model is evaluated on:
- **FF++ Test Set**: In-distribution performance
- **JPEG Compression**: Robustness to compression artifacts
- **DFDC**: Cross-dataset generalization
- **Celeb-DF**: Cross-dataset generalization

## Results

| Dataset | AUC | Accuracy | F1-Score |
|---------|-----|----------|----------|
| FF++ Test | - | - | - |
| FF++ + JPEG 75% | - | - | - |
| DFDC | - | - | - |
| Celeb-DF | - | - | - |

*Results will be populated after running the evaluation cells in the notebook.*

## Citation

If you use this code in your research, please cite:

```bibtex
@article{narr2024,
  title={Nuisance Aware Representation Refinement for Robust Deepfake Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.