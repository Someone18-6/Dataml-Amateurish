"""
Exploratory Data Analysis for the Chest X-Ray Multi-Condition Detection challenge.

Run: python src/eda.py
Expects:
  data/train_labels.csv
  data/train/*.png
Writes summary figures to reports/figures/
"""

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["Atelectasis", "Effusion", "Infiltration", "Nodule", "Pneumothorax"]


def load_labels():
    df = pd.read_csv(DATA_DIR / "train_labels.csv")
    return df


def class_balance(df):
    print("\n=== Class balance (positive rate per condition) ===")
    counts = df[LABEL_COLS].sum()
    rates = df[LABEL_COLS].mean()
    summary = pd.DataFrame({"positive_count": counts, "positive_rate": rates})
    print(summary)

    ax = summary["positive_count"].plot(kind="bar", title="Positive count per condition")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png")
    plt.close()

    # how many images have 0 / 1 / 2+ conditions
    n_labels = df[LABEL_COLS].sum(axis=1)
    print("\n=== Number of conditions per image ===")
    print(n_labels.value_counts().sort_index())

    return summary


def co_occurrence(df):
    print("\n=== Label co-occurrence matrix (counts) ===")
    co = df[LABEL_COLS].T.dot(df[LABEL_COLS])
    print(co)

    fig, ax = plt.subplots()
    im = ax.imshow(co.values, cmap="Blues")
    ax.set_xticks(range(len(LABEL_COLS)))
    ax.set_yticks(range(len(LABEL_COLS)))
    ax.set_xticklabels(LABEL_COLS, rotation=45, ha="right")
    ax.set_yticklabels(LABEL_COLS)
    for i in range(len(LABEL_COLS)):
        for j in range(len(LABEL_COLS)):
            ax.text(j, i, co.values[i, j], ha="center", va="center", fontsize=8)
    ax.set_title("Label co-occurrence")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "co_occurrence.png")
    plt.close()

    return co


def image_properties(df, sample_n=None):
    """Extends check_p.py: image mode/size checks, now also intensity stats."""
    train_path = DATA_DIR / "train"
    files = list(train_path.glob("*.png"))
    if sample_n:
        files = files[:sample_n]

    modes = Counter()
    sizes = Counter()
    means, stds = [], []
    errors = []

    for f in files:
        try:
            with Image.open(f) as img:
                modes[img.mode] += 1
                sizes[img.size] += 1
                arr = np.array(img.convert("L"), dtype=np.float32)
                means.append(arr.mean())
                stds.append(arr.std())
        except Exception as e:
            errors.append((f.name, str(e)))

    print("\n=== Image properties ===")
    print("Total images checked:", len(files))
    print("Modes:", modes)
    print("Sizes:", sizes)
    print("Unreadable images:", len(errors))
    if errors:
        print("First few errors:", errors[:5])

    if means:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(means, bins=30)
        axes[0].set_title("Per-image mean intensity")
        axes[1].hist(stds, bins=30)
        axes[1].set_title("Per-image intensity std")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "intensity_distributions.png")
        plt.close()

    return {"modes": modes, "sizes": sizes, "n_errors": len(errors)}


def main():
    df = load_labels()
    print("Total labeled training images:", len(df))
    class_balance(df)
    co_occurrence(df)
    image_properties(df)
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
