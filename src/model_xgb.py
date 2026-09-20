from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]

DISEASES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]


X_train = np.load(ROOT / "Data" / "X_train_clahe.npy")
y_train = np.load(ROOT / "Data" / "y_train_clahe.npy")

X_val = np.load(ROOT / "Data" / "X_val_clahe.npy")
y_val = np.load(ROOT / "Data" / "y_val_clahe.npy")

X_test = np.load(ROOT / "Data" / "X_test_clahe.npy")

test_files = sorted(
    [file.name for file in (ROOT / "datasets" / "test").glob("*.png")]
)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

val_predictions = np.zeros(
    (len(X_val), len(DISEASES))
)

test_predictions = np.zeros(
    (len(X_test), len(DISEASES))
)

auc_scores = []

for i, disease in enumerate(DISEASES):

    print("\nTraining:", disease)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train[:, i]
    )

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    val_predictions[:, i] = val_scores
    test_predictions[:, i] = test_scores

    auc = roc_auc_score(
        y_val[:, i],
        val_scores
    )

    auc_scores.append(auc)

    print("AUROC:", round(auc, 4))


macro_auc = np.mean(auc_scores)

print("\nEXP-06: CLAHE + HOG + Intensity + XGBoost")

for disease, auc in zip(DISEASES, auc_scores):
    print(disease + ":", round(auc, 4))

print("\nMacro AUROC:", round(macro_auc, 4))


validation_results = pd.DataFrame(
    val_predictions,
    columns=DISEASES
)

validation_results.insert(
    0,
    "Image Index",
    np.arange(len(val_predictions))
)

validation_results.to_csv(
    ROOT / "Data" / "validation_predictions.csv",
    index=False
)


test_results = pd.DataFrame(
    test_predictions,
    columns=DISEASES
)

test_results.insert(
    0,
    "Image Index",
    test_files
)

test_results.to_csv(
    ROOT / "Data" / "test_predictions.csv",
    index=False
)


experiment_results = pd.DataFrame({
    "Disease": DISEASES,
    "AUROC": auc_scores
})

experiment_results.loc[len(experiment_results)] = [
    "Macro AUROC",
    macro_auc
]

experiment_results.to_csv(
    ROOT / "Data" / "EXP-06_results.csv",
    index=False
)

print("\nFiles saved successfully.")