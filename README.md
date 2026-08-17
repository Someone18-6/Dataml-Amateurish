# Chest X-Ray Multi-Condition Detection

## What we're solving

We're given chest X-rays and need to figure out, for each one, how likely it is to show
each of five conditions: **Atelectasis, Effusion, Infiltration, Nodule, and Pneumothorax**.

This isn't a "pick one" problem — a single X-ray can show multiple conditions at once, or
none at all. So instead of classifying each image into one bucket, we're predicting five
separate probabilities per image, one per condition.

The real constraint on this challenge is **no deep learning**. No CNNs, no pretrained models,
no learned feature extractors — everything has to be built by hand: we choose what to measure
in each image, extract it ourselves, and justify why it should matter medically. The modeling
part (logistic regression, random forest, etc.) is almost secondary to getting the feature
representation right.

## How the repo is organized
data/ # train_labels.csv + raw images (images are gitignored, not pushed)
src/ # our actual code — EDA, feature extraction, training
notebooks/ # scratch work / exploration
reports/ # write-up and figures for submission
outputs/ # predictions.csv and saved models
## Where we are right now

- [x] Repo set up
- [ ] EDA — class balance, which conditions co-occur, basic image stats
- [ ] Texture features (GLCM / LBP)
- [ ] Intensity / histogram features
- [ ] Shape & segmentation features
- [ ] Edge / gradient features (HOG)
- [ ] Frequency-domain features (Gabor / FFT)
- [ ] Train + validate models
- [ ] Generate final predictions.csv
- [ ] Write-up

## Our approach so far

We're treating this as five separate binary classification problems rather than one
multi-class problem — one classifier per condition, each scored on ROC-AUC against a
held-out validation split. This keeps things simple to reason about: we can tell exactly
which features are actually helping with which disease, instead of everything getting
blended into one score.

## Running it yourself

```bash
pip install -r requirements.txt
python src/eda.py            # EDA summary + figures → reports/
python src/features.py       # builds the feature matrix → outputs/
python src/train.py          # trains models, writes outputs/predictions.csv
```
