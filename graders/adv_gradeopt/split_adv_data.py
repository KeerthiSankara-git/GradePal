
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import os

os.makedirs("data/adv_splits", exist_ok=True)
os.makedirs("data/adv_train", exist_ok=True)

# Load adversarial dataset
df = pd.read_csv("data/adversarial_dataset_with_feedback.csv")

# Create stratification key (attack_type + label)
df["stratify_key"] = df["attack_type"].astype(str) + "_" + df["output_label"].astype(str)

# Step 1: split train (70%) vs temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["stratify_key"]
)

# Step 2: split temp → val (10%) + test (20%)
temp_df["stratify_key"] = temp_df["attack_type"].astype(str) + "_" + temp_df["output_label"].astype(str)

val_df, test_df = train_test_split(
    temp_df,
    test_size=2/3,   # 20% total
    random_state=42,
    stratify=temp_df["stratify_key"]
)

# Drop helper column
train_df = train_df.drop(columns=["stratify_key"])
test_df = test_df.drop(columns=["stratify_key"])
val_df = val_df.drop(columns=["stratify_key"])

# Save
train_df.to_csv("data/adv_splits/adv_train.csv", index=False)
test_df.to_csv("data/adv_splits/adv_test.csv", index=False)
val_df.to_csv("data/adv_splits/adv_val.csv", index=False)

print("Split done:")
print(f"Train: {len(train_df)}")
print(f"Test: {len(test_df)}")
print(f"Val: {len(val_df)}")


# Combine adversarial training data with original training data
orig_train = pd.read_csv("data/train.csv")
adv_train = pd.read_csv("data/adv_splits/adv_train.csv")

# Combine
train_combined = pd.concat([orig_train, adv_train], ignore_index=True)

# Shuffle
train_combined = train_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
train_combined.to_csv("data/adv_train/original_plus_adv_train.csv", index=False)

print(f"Final train size: {len(train_combined)}")


orig_val = pd.read_csv("data/val.csv")
adv_val  = pd.read_csv("data/adv_splits/adv_val.csv")

# Combine
combined_val = pd.concat([orig_val, adv_val], ignore_index=True)

# Shuffle
combined_val = combined_val.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
combined_val.to_csv("data/adv_train/original_plus_adv_val.csv", index=False)

print(f"Combined VAL size: {len(combined_val)}")