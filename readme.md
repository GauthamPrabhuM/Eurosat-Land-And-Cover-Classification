# EuroSAT Land Cover Classification

Deep learning models for multi-class land-use classification using Sentinel-2 multispectral satellite imagery. Implements transfer learning and evaluates multiple CNN and transformer-based architectures.

## Features

- **Multiple Architectures**: ResNet50, EfficientNetB3, Vision Transformer (ViT)
- **Transfer Learning**: Pre-trained ImageNet weights with fine-tuning
- **Multispectral Processing**: Handles RGB bands from Sentinel-2 imagery
- **Comprehensive Evaluation**: Confusion matrices, per-class metrics, error analysis
- **Data Augmentation**: Random flips, rotations, color jitter for robustness

## Tech Stack

- PyTorch
- TensorFlow (compatible)
- torchvision (ResNet50, EfficientNetB3, ViT)
- scikit-learn (metrics)
- matplotlib/seaborn (visualization)

## Dataset

EuroSAT contains 27,000 labeled Sentinel-2 satellite image patches covering 10 land use classes:

1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial
6. Pasture
7. Permanent Crop
8. Residential
9. River
10. Sea/Lake

**Download**: [EuroSAT Dataset](https://github.com/phelber/EuroSAT)

## Installation

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pillow
```

## Project Structure

```
eurosat-classification/
├── train.py              # Training script with all models
├── evaluate.py           # Evaluation and visualization
├── requirements.txt
├── data/
│   └── EuroSAT/
│       ├── train/
│       ├── val/
│       └── test/
└── README.md
```

## Usage

### Training

```bash
# Train ResNet50
python train.py

# Modify MODEL_NAME in train.py for other architectures:
# - resnet50
# - efficientnet_b3
# - vit (Vision Transformer)
```

**Configuration** (in `train.py`):
```python
TRAIN_DIR = 'data/EuroSAT/train'
VAL_DIR = 'data/EuroSAT/val'
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_NAME = 'resnet50'
```

### Evaluation

```bash
python evaluate.py
```

Generates:
- Overall accuracy
- Per-class precision/recall/F1
- Confusion matrix heatmap
- Per-class accuracy bar chart
- Top misclassification analysis

### Model Comparison

Uncomment the comparison section in `evaluate.py` to compare multiple architectures:

```python
compare_models(test_loader, class_names)
```

## Model Architectures

### ResNet50
- 50-layer deep residual network
- Transfer learning from ImageNet
- Modified final FC layer for 10 classes

### EfficientNetB3
- Compound scaling method
- Balanced depth, width, resolution
- Efficient parameter usage

### Vision Transformer (ViT)
- Transformer-based architecture
- Patch-based image processing
- Attention mechanisms for spatial relationships

## Training Features

- **Transfer Learning**: ImageNet pre-trained weights
- **Data Augmentation**: Horizontal/vertical flips, rotations, color jitter
- **Learning Rate Scheduling**: ReduceLROnPlateau with patience
- **Early Stopping**: Best model saved based on validation accuracy
- **Cross-Entropy Loss**: Standard classification objective
- **Adam Optimizer**: Adaptive learning rates

## Expected Results

Typical accuracies on EuroSAT test set:

- ResNet50: ~95-97%
- EfficientNetB3: ~96-98%
- Vision Transformer: ~97-99%

## Multispectral Feature Fusion

The `MultispectralFusion` class in `train.py` demonstrates custom layer design for combining different spectral bands (adaptable for full 13-band Sentinel-2 data).

## Visualization Examples

The evaluation script produces:
- **Confusion Matrix**: Shows class-wise prediction patterns
- **Per-Class Accuracy**: Identifies challenging classes
- **Model Comparison**: Benchmark different architectures

## Data Preparation

Expected directory structure:
```
data/EuroSAT/
├── train/
│   ├── AnnualCrop/
│   ├── Forest/
│   └── ...
├── val/
│   ├── AnnualCrop/
│   └── ...
└── test/
    ├── AnnualCrop/
    └── ...
```

## Custom Dataset

To use your own satellite imagery:

1. Organize images in class folders
2. Update paths in `train.py` and `evaluate.py`
3. Adjust `NUM_CLASSES` parameter
4. Modify transforms if needed (e.g., for different resolutions)

## Citation

If using EuroSAT dataset:
```
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
```

## Future Work

- Full 13-band multispectral processing
- Temporal sequence analysis
- Semantic segmentation variants
- Attention visualization
- Model ensemble methods

## Acknowledgements

This project was refactored and documented with assistance from generative AI tools to improve code structure, documentation, and developer experience. The AI helped with:

- Restructuring the repository into a small package layout.
- Clarifying README instructions and usage examples.
- Adding lightweight wrappers and stability improvements in training/evaluation scripts.

All code changes were reviewed and adapted by the repository owner. See `CREDITS.md` for details.