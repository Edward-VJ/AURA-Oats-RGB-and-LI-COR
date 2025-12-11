
# AURA-Oats: Codebase Documentation
### RGB-Based High-Throughput Phenotyping Pipeline  
*Companion to the manuscript: “Field-Based Robotic Phenotyping of Oats Using RGB Imaging to Predict LI-600 Physiological Traits”*

This repository contains the scripts and notebooks used to construct datasets, perform image segmentation, extract RGB metrics, train deep learning regression models, and evaluate drought-related physiological predictions. The codebase reflects the full experimental workflow described in the associated manuscript.

This repository is intended as a scientific research toolkit rather than a turnkey software package. Many scripts operate as modular tools and require manual parameter settings, dataset path adjustments, and repeated execution for different experimental conditions.

---
## 0. environment setup

### 0.1 Clone this Repository

```bash
git clone https://github.com/Edward-VJ/AURA-Oats-RGB-and-LI-COR.git
cd AURA-Oats-RGB-and-LI-COR
```

### 0.2 Create the Conda Environment


```bash
conda env create -f environment.yml
conda activate paper_1
```

### 0.3 External Research Models (Required Repositories)
These repositories must be installed manually

Depth Anything V2
https://github.com/DepthAnything/Depth-Anything-V2

Meta SAM2
https://github.com/facebookresearch/sam2

LangSAM
https://github.com/luca-medeiros/lang-segment-anything


## 1. Dataset Construction and Preprocessing

### `batch_depth_creation.pt`  
Converts RGB images to depth images using Depth Anything V2

### `train_classification.ipynb`  
Trains the YOLOv11 classifier used to label SAM2-generated organ masks (Leaf, Panicle, PanicleClump, Noise).

### `best.pt`  
Final YOLOv11 model weights used in organ-level classification.

### `sam2_full_pipeline.ipynb`  
Implements the full segmentation workflow, including:
- Depth-Anything background masking  
- LangSAM whole-plant isolation  
- SAM2 organ segmentation  
- YOLOv11 classification  
- Depth-based top-leaf selection  

Outputs whole-plant masks and top-leaf masks for subsequent analysis.

### `Extract RGB Metrics.ipynb`  
Computes RGB vegetation indices (NGRDI, TGI, VARI, GLI, a*, etc.) and 13 descriptive statistics for top-leaf and whole-plant regions.

### `Extract RGB Metrics OD.ipynb`  
Equivalent to the above, adapted for the outdoor-field dataset, including colour-balancing steps and central-cropping logic.

### `plywood_extraction.py`
Extracts masks of plywood from background

### `Correlation Calculations 2025.ipynb`  
Calculate correlations between RGB metrics and licor measurements.

### `lam_mask_retriever.py`  
Retrieves whole-plant masks using LangSAM and the “oat plant” prompt.

### `lam_dataset_csv_builder.py`  
Builds a structured CSV linking plant identifiers, views, dates, masks, and RGB metrics.

### `split_test_training.py`  
Generates train/validation/test splits, allowing specific pots or dates to be fixed for test sets or generalisation analysis.

---

## 2. Deep Learning Model Training

### Outdoor Data Training Scripts

#### `multispectral-training-hard-lr-single.py`  
Trains single-output regression models for individual LI-600 traits.

#### `multispectral-training-hard-lr.py`  S
Trains multi-output models that predict all LI-600 traits simultaneously.

### Greenhouse Data Training Scripts

#### `old_data_single_var_combined_training.py`  
Legacy single-metric training pipeline used in early experiments.

#### `resnet_training_multivariable.ipynb`  
Trains ResNet-based regression models on greenhouse datasets.

#### `swin_training_multivariable.py`  
Trains Swin Transformer models (single- and multi-output). These models exhibited strong performance for several LI-600 traits.

---

## 3. Exporting and Organising Experiment Results

### `paper2_experiments_csv_export.py`  
Exports deep learning results into structured CSV files used for comparative analysis.

### `paper2_experiments_singlevar_csv_export.py`  
Exports results for single-metric models.

---

## 4. Verification and Evaluation Tools

### `sam2_manual_Segmentation_oats.ipynb`  
Interactive tool for creating human-annotated segmentation masks and assessing inter-rater and pipeline agreement.

### `calcualte_iou_bulk.py`  
Computes IoU between human masks and pipeline masks for mAP@50 segmentation accuracy.

### `infer_predict_export.py`  
Runs inference using trained models and exports predicted physiological values - csv witha ll values, scatterplots and some examples images.

### `infer_predict_export_od.py`  
Same as above but for outdoor data.

### `drought_analysis_multiclass_cnn_rf.ipynb`  
Evaluates drought-classification performance for multi-output regression models.

### `resnet_testing_multivariable.py`  
Evaluates trained ResNet models on held-out test sets.

### `swin_testing_multivariable.py`  
Evaluates Swin Transformer models and generates R² values reported in the manuscript.

---

## 5. Notes on Workflow and Reproducibility

This repository reflects a modular, research-oriented workflow. Many scripts require manual adjustment of:
- dataset paths  
- segmentation source  
- model checkpoints  
- selected LI-600 target variable  
- train/validation/test splits  

This design supports systematic experimentation, ablation studies, and evaluation across segmentation methods and model architectures.



---


