# Chest X-Ray Multi-Condition Detection — DataML Technical Championship

## Problem Understanding

Given a chest X-ray image, predict independent probabilities (0–1) for the presence of
five conditions:

- Atelectasis
- Effusion
- Infiltration
- Nodule
- Pneumothorax

A single X-ray may show zero, one, or several of these at once — this is a **multi-label**
problem, not multi-class. Each condition gets its own probability score per image.

**Constraint:** classical ML only. No deep learning, no CNNs, no pretrained models or
pretrained feature extractors at any stage. All image representations must be hand-designed
and explicitly justified.

## Repository Structure

```
data/            # train_labels.csv lives here; raw images are NOT committed (see .gitignore)
src/             # reusable Python modules (EDA, feature extraction, modeling)
notebooks/       # exploratory notebooks
reports/         # write-up, figures, checkpoint notes
outputs/         # predictions.csv, saved models (large binaries gitignored)
```

## Status

- [x] Repo scaffolded
- [ ] EDA (class balance, co-occurrence, image properties)
- [ ] Feature engineering — texture (GLCM/LBP)
- [ ] Feature engineering — intensity/histogram
- [ ] Feature engineering — shape/segmentation
- [ ] Feature engineering — edge/gradient (HOG)
- [ ] Feature engineering — frequency domain (Gabor/FFT)
- [ ] Model training + validation split
- [ ] Final predictions.csv
- [ ] Write-up (reports/)

## Approach (will be expanded as work progresses)

We are treating this as 5 independent binary classification problems (One-vs-Rest),
each scored by ROC-AUC on a held-out validation split, since it maps directly onto the
"probability per condition" requirement and keeps each feature's contribution to each
disease interpretable and easy to justify.

## How to Reproduce

```bash
pip install -r requirements.txt
python src/eda.py            # generates EDA summary + figures in reports/
python src/features.py       # extracts feature matrix to outputs/
python src/train.py          # trains models, outputs/predictions.csv
```
