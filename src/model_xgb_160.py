import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


DISEASES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]


X_train = np.load("Data/X_train_160.npy")
y_train = np.load("Data/y_train_160.npy")

X_val = np.load("Data/X_val_160.npy")
y_val = np.load("Data/y_val_160.npy")


print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


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

    scores = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(
        y_val[:, i],
        scores
    )

    auc_scores.append(auc)

    print("AUROC:", round(auc, 4))


macro_auc = np.mean(auc_scores)


print("EXP-07: 160x160 + XGBoost")
print("Macro AUROC:", round(macro_auc, 4))