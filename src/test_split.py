import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Path to CSV
csv_path = r"C:\Users\K SRAVYA\Downloads\Chest_xrays\Chest_xrays\train_labels.csv"

# Load dataset
df = pd.read_csv(csv_path)

# Create base_id from Image Index
df["base_id"] = df["Image Index"].str.rsplit("_", n=1).str[0]

print("Total images:", len(df))
print("Unique base IDs:", df["base_id"].nunique())

# -------------------------------
# STEP 1: Train = 80%, Temporary = 20%
# -------------------------------

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    gss.split(df, groups=df["base_id"])
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

# -------------------------------
# STEP 2: Validation = 10%, Test = 10%
# -------------------------------

gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    gss2.split(temp_df, groups=temp_df["base_id"])
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

# -------------------------------
# RESULTS
# -------------------------------

print("\n--- SPLIT RESULTS ---")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

print("\nTraining groups:", train_df["base_id"].nunique())
print("Validation groups:", val_df["base_id"].nunique())
print("Test groups:", test_df["base_id"].nunique())

# -------------------------------
# CHECK FOR GROUP OVERLAP
# -------------------------------

train_groups = set(train_df["base_id"])
val_groups = set(val_df["base_id"])
test_groups = set(test_df["base_id"])

print("\n--- GROUP OVERLAP CHECK ---")

print("Train and Validation overlap:", len(train_groups & val_groups))
print("Train and Test overlap:", len(train_groups & test_groups))
print("Validation and Test overlap:", len(val_groups & test_groups))
# Save splits
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("validation_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

print("\nSplit files saved:")
print("train_split.csv")
print("validation_split.csv")
print("test_split.csv")