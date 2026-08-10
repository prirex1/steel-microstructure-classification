# steel-microstructure-classification
ResNet-50 classification of steel microstructure phases with Grad-CAM and UMAP analysis
# Steel Microstructure Phase Classification

A deep learning pipeline that classifies steel microstructure phases from micrograph images, combining a fine-tuned ResNet-50 with explainability (Grad-CAM) and unsupervised feature analysis (UMAP).

## Problem

Identifying microstructure phases in steel (spheroidite, network cementite, spheroidite+widmanstatten) is a core task in materials characterization, traditionally done by manual inspection under a microscope. This project automates that classification using computer vision, while keeping the model's decisions interpretable — critical for a domain where trust in the model's reasoning matters as much as accuracy.

## Dataset

[UHCS (Ultra High Carbon Steel) microstructure dataset](https://www.kaggle.com/datasets/safi842/highcarbon-micrographs/data) from Kaggle — 550 labeled micrographs across 3 phase classes:

| Class | Count | Share |
|---|---|---|
| Spheroidite | 372 | 67.6% |
| Network cementite | 101 | 18.4% |
| Spheroidite + Widmanstätten | 77 | 14.0% |

The class imbalance (spheroidite dominates ~2:1 over the other two combined) directly shaped the modeling choices below.

## Approach

- **Backbone**: ResNet-50, fine-tuned on the micrograph classification task
- **Class imbalance**: handled with weighted cross-entropy loss rather than oversampling, to avoid duplicating a limited dataset
- **Explainability**: Grad-CAM (`pytorch-grad-cam`) to visualize which regions of each micrograph drove the model's prediction — useful for sanity-checking that the model is attending to actual phase boundaries and not artifacts
- **Feature analysis**: UMAP projection of penultimate-layer embeddings to inspect how well the three classes separate in learned feature space before the final classification layer
- **Compute**: trained and evaluated entirely on CPU

## Results

- **86% overall test accuracy** across 3 classes
- **100% F1-score on network cementite** — despite being the minority class, the model separates it cleanly, likely due to its visually distinct network-like grain boundary structure
- Confusion matrix and training curves included in the repo (`confusion_matrix.png`, `training_curves.png`)

## Stack

`PyTorch` · `torchvision` · `OpenCV (headless)` · `scikit-learn` · `scikit-image` · `pytorch-grad-cam` · `umap-learn` · `matplotlib` · `seaborn` · `Jupyter` (Anaconda environment)

## Repo structure

```
├── steel_microstructure_cnn.ipynb      # Main training + evaluation notebook (ResNet-50)
├── steel_microstructure_unet.ipynb     # U-Net experimentation notebook
├── microstructure_eda.ipynb            # Exploratory data analysis
├── train_split.csv / val_split.csv / test_split.csv
├── class_distribution.png
├── confusion_matrix.png
├── training_curves.png
├── gradcam_results.png                 # Grad-CAM visualizations
├── umap_features.png                   # UMAP projection of learned features
├── texture_features.png
├── intensity_histograms.png
├── micrograph_grid.png
└── resnet50_microstructure.pth         # Trained model weights (~90MB)
```

## Running it

1. Clone the repo and set up the environment (Anaconda recommended):
   ```
   conda create -n microstructure python=3.10
   conda activate microstructure
   pip install torch torchvision opencv-python-headless scikit-learn scikit-image pytorch-grad-cam umap-learn matplotlib seaborn jupyter
   ```
2. Open `steel_microstructure_cnn.ipynb` and run cells sequentially — the notebook covers data loading, training, evaluation, and Grad-CAM/UMAP generation.
3. Pretrained weights (`resnet50_microstructure.pth`) are included in the repo for direct inference without retraining.

## Notes

- The `steel_microstructure_unet.ipynb` notebook contains an alternate segmentation-style approach explored during development; the ResNet-50 classifier is the primary, reported model.
- Results are based on a CPU-only training setup — no GPU-specific optimizations were required given the dataset size.
