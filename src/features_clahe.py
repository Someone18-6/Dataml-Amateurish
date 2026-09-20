from pathlib import Path

import cv2
import pandas as pd
import numpy as np
from skimage.feature import hog


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "datasets" / "train"
TEST_DIR = ROOT / "datasets" / "test"

DISEASES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]


def extract_features(df, image_dir, include_labels=True):

    features = []
    labels = []
    image_names = []
    missing = 0

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    for _, row in df.iterrows():

        image_name = row["Image Index"]
        image_path = image_dir / image_name

        if not image_path.exists():
            missing += 1
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            missing += 1
            continue

        image = clahe.apply(image)

        image = cv2.resize(image, (128, 128))

        hog_features = hog(
            image,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        intensity = image.astype(np.float32) / 255.0
        intensity = intensity.flatten()

        feature = np.concatenate([
            hog_features,
            intensity
        ])

        features.append(feature)
        image_names.append(image_name)

        if include_labels:
            labels.append([
                row["Atelectasis"],
                row["Effusion"],
                row["Infiltration"],
                row["Nodule"],
                row["Pneumothorax"]
            ])

    print("Missing images:", missing)

    X = np.array(features)

    if include_labels:
        y = np.array(labels)
    else:
        y = None

    return X, y, image_names


if __name__ == "__main__":

    train_df = pd.read_csv(
        ROOT / "Data" / "train_split.csv"
    )

    print("\nExtracting training features...")

    X_train, y_train, train_names = extract_features(
        train_df,
        TRAIN_DIR,
        include_labels=True
    )

    val_df = pd.read_csv(
        ROOT / "Data" / "validation_split.csv"
    )

    print("\nExtracting validation features...")

    X_val, y_val, val_names = extract_features(
        val_df,
        TRAIN_DIR,
        include_labels=True
    )

    test_files = sorted(
        [file.name for file in TEST_DIR.glob("*.png")]
    )

    test_df = pd.DataFrame({
        "Image Index": test_files
    })

    print("\nExtracting test features...")

    X_test, _, test_names = extract_features(
        test_df,
        TEST_DIR,
        include_labels=False
    )

    print("\n========== FEATURE SHAPES ==========")

    print("Training features:", X_train.shape)
    print("Training labels:", y_train.shape)

    print("Validation features:", X_val.shape)
    print("Validation labels:", y_val.shape)

    print("Test features:", X_test.shape)

    np.save(
        ROOT / "Data" / "X_train_clahe.npy",
        X_train
    )

    np.save(
        ROOT / "Data" / "y_train_clahe.npy",
        y_train
    )

    np.save(
        ROOT / "Data" / "X_val_clahe.npy",
        X_val
    )

    np.save(
        ROOT / "Data" / "y_val_clahe.npy",
        y_val
    )

    np.save(
        ROOT / "Data" / "X_test_clahe.npy",
        X_test
    )

    print("\nCLAHE features saved successfully.")