import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\K SRAVYA\Downloads\Chest_xrays\Chest_xrays\Dataml-Amateurish\datasets\train_labels.csv"
)

image_col = "Image Index"

df["Base_ID"] = (
    df[image_col]
    .astype(str)
    .str.rsplit("_", n=1)
    .str[0]
)

base_id_counts = df["Base_ID"].value_counts()

total_images = len(df)

repeated_ids = base_id_counts[base_id_counts > 1]
single_ids = base_id_counts[base_id_counts == 1]

repeated_images = repeated_ids.sum()
single_images = single_ids.sum()

repeated_pct = (repeated_images / total_images) * 100
single_pct = (single_images / total_images) * 100

print("Base ID Analysis")
print("-" * 30)
print(f"Total images: {total_images:,}")
print(f"Repeated Base IDs: {len(repeated_ids):,}")
print(f"Images from repeated IDs: {repeated_images:,}")
print(f"Images from single-image IDs: {single_images:,}")
print(f"Repeated-image percentage: {repeated_pct:.1f}%")

labels = ["Single-image IDs", "Images from repeated IDs"]
values = [single_images, repeated_images]
percentages = [single_pct, repeated_pct]

fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.bar(labels, values, width=0.55)

for bar, value, pct in zip(bars, values, percentages):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{value:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold"
    )

ax.set_title(
    "More Than 1 in 3 Images Belong to Repeated Base IDs",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_ylabel("Number of Images", fontsize=12)

ax.text(
    0.5,
    0.91,
    f"{repeated_images:,} of {total_images:,} images "
    f"({repeated_pct:.1f}%) come from repeated IDs",
    transform=ax.transAxes,
    ha="center",
    fontsize=12
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylim(0, max(values) * 1.15)

plt.tight_layout()

plt.savefig(
    "base_id_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()