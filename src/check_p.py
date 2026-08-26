import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_path = Path("train")

# Get image filenames
image_files = {
    file.name
    for file in train_path.glob("*.png")
}

# Read CSV
df = pd.read_csv("train_labels.csv")

csv_files = set(df["Image Index"])

# Compare
images_not_in_csv = image_files - csv_files
csv_not_in_images = csv_files - image_files

print("Images in train folder:", len(image_files))
print("Rows in CSV:", len(df))

print("Images without CSV labels:", len(images_not_in_csv))
print("CSV entries without image:", len(csv_not_in_images))

print("Duplicate Image Index values:", df["Image Index"].duplicated().sum())

df["base_id"] = df["Image Index"].str.split("_").str[0]

print(df["base_id"].nunique())

print(df["base_id"].value_counts().head(20))


group_sizes = df["base_id"].value_counts()

print("Unique base IDs:", group_sizes.shape[0])
print("Base IDs with >1 image:", (group_sizes > 1).sum())
print("Images belonging to repeated groups:", group_sizes[group_sizes > 1].sum())



# example_id = "00010007"

# print(
#     df[df["base_id"] == example_id]
#     [["Image Index",
#       "Atelectasis",
#       "Effusion",
#       "Infiltration",
#       "Nodule",
#       "Pneumothorax"]]
# )


# example_id = "00017138"

# print(
#     df[df["base_id"] == example_id]
#     [["Image Index",
#       "Atelectasis",
#       "Effusion",
#       "Infiltration",
#       "Nodule",
#       "Pneumothorax"]]
# )


# df["suffix"] = df["Image Index"].str.extract(r"_(\d+)\.png")[0].astype(int)

# print(df["suffix"].describe())

# print(df["suffix"].value_counts().head(20))



# img = Image.open(train_path / "00000084_000.png")

# print("Mode:", img.mode)

# arr = np.array(img)

# print("Shape:", arr.shape)
# print("Min:", arr.min())
# print("Max:", arr.max())

# print("Alpha channel:")
# print("Min:", arr[:, :, 3].min())
# print("Max:", arr[:, :, 3].max())


# rgb = arr[:, :, :3]

# print("R == G:", np.array_equal(rgb[:, :, 0], rgb[:, :, 1]))
# print("G == B:", np.array_equal(rgb[:, :, 1], rgb[:, :, 2]))
# print("R == B:", np.array_equal(rgb[:, :, 0], rgb[:, :, 2]))




# train_path = Path("train")
# files = list(train_path.glob("*.png"))

# mins = []
# maxs = []
# means = []

# for file in files:
#     with Image.open(file) as img:
#         img = img.convert("L")
#         arr = np.array(img)

#         mins.append(arr.min())
#         maxs.append(arr.max())
#         means.append(arr.mean())

# print("Images:", len(files))

# print("Minimum pixel value:")
# print(min(mins), max(mins))

# print("\nMaximum pixel value:")
# print(min(maxs), max(maxs))

# print("\nMean pixel value:")
# print(min(means), max(means))





# plt.hist(means, bins=50)

# plt.title("Distribution of Mean Pixel Intensity")
# plt.xlabel("Mean Pixel Value")
# plt.ylabel("Number of Images")

# plt.show()


# #contrast Distribution
# contrasts = []

# for file in files:
#     with Image.open(file) as img:
#         img = img.convert("L")
#         arr = np.array(img)

#         contrasts.append(arr.std())

# print("Minimum contrast:", min(contrasts))
# print("Maximum contrast:", max(contrasts))

# #Viualize using histogram
# plt.hist(contrasts, bins=50)

# plt.title("Distribution of Image Contrast")
# plt.xlabel("Pixel Intensity Standard Deviation")
# plt.ylabel("Number of Images")

# plt.show()




diseases = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]

for disease in diseases:

    brightness = []

    for _, row in df[df[disease] == 1].iterrows():

        image_path = train_path / row["Image Index"]

        with Image.open(image_path) as img:
            img = img.convert("L")
            arr = np.array(img)

            brightness.append(arr.mean())

    print("\n", disease)
    print("Images:", len(brightness))
    print("Average:", np.mean(brightness))
    print("Minimum:", np.min(brightness))
    print("Maximum:", np.max(brightness))



disease = "Atelectasis"

brightness = []

for _, row in df[df[disease] == 1].iterrows():
    image_path = train_path / row["Image Index"]

    with Image.open(image_path) as img:
        img = img.convert("L")
        arr = np.array(img)
        brightness.append(arr.mean())

print("Number of images:", len(brightness))
print("Average brightness:", np.mean(brightness))
print("Minimum:", np.min(brightness))
print("Maximum:", np.max(brightness))



diseases = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]

averages = [
    127.8534,
    122.9188,
    120.8767,
    128.5476,
    120.4586
]

plt.bar(diseases, averages)

plt.title("Average Brightness by Disease")
plt.xlabel("Disease")
plt.ylabel("Average Pixel Intensity")
plt.xticks(rotation=30)

plt.show()


diseases = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "Nodule",
    "Pneumothorax"
]

for disease in diseases:
    count = df[disease].sum()
    percentage = count / len(df) * 100

    print(f"{disease}: {count} images ({percentage:.2f}%)")


condition_counts = df[diseases].sum(axis=1)

distribution = condition_counts.value_counts().sort_index()

for number, count in distribution.items():
    percentage = count / len(df) * 100
    print(
        f"{number} diseases: "
        f"{count} images ({percentage:.2f}%)"
    )

co_occurrence = df[diseases].T.dot(df[diseases])

plt.figure(figsize=(8, 6))

sns.heatmap(
    co_occurrence,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Disease Co-occurrence")
plt.xlabel("Disease")
plt.ylabel("Disease")

plt.show()

co_occurrence_percent = (
    co_occurrence.div(df[diseases].sum(axis=0), axis=1) * 100
)

print(co_occurrence_percent.round(2))


label_corr = df[diseases].corr()

print(label_corr.round(2))


plt.figure(figsize=(8, 6))

sns.heatmap(
    label_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Label Correlation")
plt.show()



for disease in diseases:

    contrast = []

    for _, row in df[df[disease] == 1].iterrows():

        image_path = train_path / row["Image Index"]

        with Image.open(image_path) as img:
            img = img.convert("L")
            arr = np.array(img)

            contrast.append(arr.std())

    print("\n", disease)
    print("Average contrast:", np.mean(contrast))
    print("Minimum:", np.min(contrast))
    print("Maximum:", np.max(contrast))