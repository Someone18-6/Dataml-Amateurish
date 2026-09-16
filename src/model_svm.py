import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


DISEASES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]


# Load extracted features
X_train = np.load("Data/X_train.npy")
y_train = np.load("Data/y_train.npy")

X_val = np.load("Data/X_val.npy")
y_val = np.load("Data/y_val.npy")


print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


auc_scores = []


for i, disease in enumerate(DISEASES):

    print("\nTraining:", disease)

    model = SVC(
        kernel="linear",
        class_weight="balanced"
    )

    model.fit(X_train, y_train[:, i])

    scores = model.decision_function(X_val)

    auc = roc_auc_score(y_val[:, i], scores)

    auc_scores.append(auc)

    print("AUROC:", round(auc, 4))


macro_auc = np.mean(auc_scores)

print("Macro AUROC:", round(macro_auc, 4))