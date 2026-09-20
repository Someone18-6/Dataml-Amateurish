# Chest X-Ray Multi-Condition Detection

Our work for **Varchas — DataML** on multi-condition chest X-ray detection using classical machine learning and handcrafted image features.

## Problem

This is a **multi-label classification** problem involving five chest X-ray conditions:

* Atelectasis
* Effusion
* Infiltration
* Nodule
* Pneumothorax

Each X-ray can contain more than one condition.

## Pipeline

```text
Chest X-Ray
    ↓
Grayscale Conversion
    ↓
CLAHE Preprocessing
    ↓
Resize to 128×128
    ↓
HOG + Intensity Features
    ↓
XGBoost
    ↓
Five Disease Predictions
    ↓
AUROC Evaluation
```

### Pipeline Justification

Grayscale conversion keeps the relevant intensity and structural information from the X-ray.

CLAHE improves local contrast and helps highlight subtle structures.

The images are resized to 128×128 to keep the feature representation manageable.

HOG captures local edge and structural information, while normalized intensity features preserve pixel-level information.

The combined features are given to XGBoost, with one binary classifier trained for each disease.

### Feature Extraction

CLAHE is used to improve local image contrast.

HOG captures local edge and structural information from the X-ray.

Intensity features preserve normalized pixel intensity information.

The HOG and intensity features are concatenated into a single feature vector.

```text
HOG features       : 1,764
Intensity features : 16,384
Total features     : 18,148
```

## Data Splitting

The dataset is split using **Base ID grouping** to reduce the possibility of related/repeated images appearing across training and validation sets.

Locally available images:

```text
Training   : 1603
Validation : 202
Test       : 1122
```

## Experiments

| Experiment | Pipeline | Macro AUROC |
| ---------- | -------- | ----------: |
| EXP-01 | HOG + SVM | 0.5779 |
| EXP-02 | HOG + Intensity + SVM | 0.6208 |
| EXP-03 | HOG + Intensity + Random Forest | 0.6594 |
| EXP-04 | HOG + Intensity + Logistic Regression | 0.6311 |
| EXP-05 | CLAHE + HOG + Intensity + Random Forest | 0.6970 |
| **EXP-06** | **CLAHE + HOG + Intensity + XGBoost** | **0.7091** |
| EXP-07 | CLAHE + 160×160 + HOG + Intensity + XGBoost | 0.7027 |

EXP-06 currently gives the highest Macro AUROC on the locally available validation subset.

EXP-02 showed that adding intensity features improved the HOG + SVM baseline. EXP-05 showed an improvement after introducing CLAHE, and EXP-06 further improved the score by using XGBoost.

EXP-07 tested 160×160 instead of 128×128. The Macro AUROC decreased from 0.7091 to 0.7027, so the 128×128 configuration was retained.

## Evaluation

We use **AUROC (Area Under the Receiver Operating Characteristic Curve)** for each disease.

The final metric is **Macro AUROC**, calculated as the mean AUROC across the five conditions.

| Disease | AUROC |
| ------- | -----: |
| Atelectasis | 0.5839 |
| Effusion | 0.8382 |
| Infiltration | 0.7080 |
| Nodule | 0.7020 |
| Pneumothorax | 0.7132 |
| **Mean AUROC** | **0.7091** |

## Predictions

The final predictions are generated using the EXP-06 pipeline.

```text
Grayscale
   ↓
CLAHE
   ↓
128×128
   ↓
HOG + Intensity
   ↓
XGBoost
   ↓
Five Disease Probabilities
```

Prediction file:

```text
Data/test_predictions.csv
```

The file contains probability predictions for the five diseases for the available 1122 test images.

## Constraints

The project follows the competition constraints:

* No CNNs
* No deep learning
* No pretrained models

The approach therefore uses **classical machine learning with handcrafted image features**.

## Current Best Pipeline

```text
Grayscale
   ↓
CLAHE
   ↓
128×128
   ↓
HOG + Intensity
   ↓
XGBoost
```

**Best observed Macro AUROC: 0.7091**

> Note: The reported experiments were evaluated using the images currently available locally (1603 training images and 202 validation images). Some images referenced by the split CSV files are not currently present in the local dataset. Therefore, 0.7091 represents the Mean AUROC on the locally available validation subset.
