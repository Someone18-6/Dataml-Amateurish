import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


DISEASES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]


X_train = np.load("Data/X_train.npy")
y_train = np.load("Data/y_train.npy")

X_val = np.load("Data/X_val.npy")
y_val = np.load("Data/y_val.npy")


print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


auc_scores = []


for i, disease in enumerate(DISEASES):

    print("\nTraining:", disease)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train[:, i])

    scores = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val[:, i], scores)

    auc_scores.append(auc)

    print("AUROC:", round(auc, 4))


macro_auc = np.mean(auc_scores)

print("Macro AUROC:", round(macro_auc, 4))
