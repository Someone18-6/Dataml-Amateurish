import os
import cv2
import pandas as pd
import numpy as np
from skimage.feature import hog

IMAGE_DIR = r"C:\Users\K SRAVYA\Downloads\Chest_xrays\Chest_xrays\Dataml-Amateurish\datasets\train"


def extract_features(df):
    features = []
    labels = []
    missing = 0

    for _, row in df.iterrows():

        image_name = row["Image Index"]
        image_path = os.path.join(IMAGE_DIR, image_name)

        if not os.path.exists(image_path):
            missing += 1
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            missing += 1
            continue

        # CLAHE preprocessing
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        image = clahe.apply(image)

        # EXP-07: higher resolution
        image = cv2.resize(image, (160, 160))

        # HOG features
        hog_features = hog(
            image,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        # Intensity features
        intensity = image.astype(np.float32) / 255.0
        intensity = intensity.flatten()

        # Combine HOG + intensity
        feature = np.concatenate([
            hog_features,
            intensity
        ])

        features.append(feature)

        labels.append([
            row["Atelectasis"],
            row["Effusion"],
            row["Infiltration"],
            row["Nodule"],
            row["Pneumothorax"]
        ])

    print("Missing images:", missing)

    return np.array(features), np.array(labels)


if __name__ == "__main__":

    train_df = pd.read_csv("Data/train_split.csv")
    val_df = pd.read_csv("Data/validation_split.csv")

    X_train, y_train = extract_features(train_df)
    X_val, y_val = extract_features(val_df)

    print("Training features:", X_train.shape)
    print("Training labels:", y_train.shape)

    print("Validation features:", X_val.shape)
    print("Validation labels:", y_val.shape)

    np.save("Data/X_train_160.npy", X_train)
    np.save("Data/y_train_160.npy", y_train)
    np.save("Data/X_val_160.npy", X_val)
    np.save("Data/y_val_160.npy", y_val)

    print("\nEXP-07 features saved successfully.")