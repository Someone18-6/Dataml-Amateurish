import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


DISEASES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]


# Load CLAHE features
X_train = np.load("Data/X_train_clahe.npy")
y_train = np.load("Data/y_train_clahe.npy")

X_val = np.load("Data/X_val_clahe.npy")
y_val = np.load("Data/y_val_clahe.npy")


print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


auc_scores = []


for i, disease in enumerate(DISEASES):

    print("\nTraining:", disease)

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train[:, i])

    scores = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val[:, i], scores)

    auc_scores.append(auc)

    print("AUROC:", round(auc, 4))


macro_auc = np.mean(auc_scores)


print("EXP-05: CLAHE + Random Forest")
print("Macro AUROC:", round(macro_auc, 4))
