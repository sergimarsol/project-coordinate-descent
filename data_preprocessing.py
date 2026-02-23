"""
Step 0 — Data Preprocessing
============================
Load the UCI Wine dataset, keep only classes 1 and 2,
convert labels (1 -> +1, 2 -> -1), and standardize features.
"""

import numpy as np
import os

# ──────────────────────────────────────────────
# 1. Load raw data
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "wine", "wine.data")

FEATURE_NAMES = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

raw = np.loadtxt(DATA_PATH, delimiter=",")
print(f"Raw dataset shape: {raw.shape}")  # expect (178, 14)

# ──────────────────────────────────────────────
# 2. Filter: keep only class 1 and class 2
# ──────────────────────────────────────────────
labels_raw = raw[:, 0].astype(int)
mask = (labels_raw == 1) | (labels_raw == 2)
data_filtered = raw[mask]
print(f"After filtering classes 1 & 2: {data_filtered.shape}")  # expect (130, 14)

# ──────────────────────────────────────────────
# 3. Separate labels and features
# ──────────────────────────────────────────────
labels = data_filtered[:, 0].astype(int)
X = data_filtered[:, 1:]

print(f"Class distribution before conversion: {dict(zip(*np.unique(labels, return_counts=True)))}")

# ──────────────────────────────────────────────
# 4. Convert labels: class 1 -> +1, class 2 -> -1
# ──────────────────────────────────────────────
y = np.where(labels == 1, 1, -1)

print(f"Label distribution after conversion: +1 -> {np.sum(y == 1)}, -1 -> {np.sum(y == -1)}")

# ──────────────────────────────────────────────
# 5. Standardize features (zero mean, unit variance)
# ──────────────────────────────────────────────
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# Guard against zero-variance features (shouldn't happen here, but safe practice)
X_std[X_std == 0] = 1.0

X_standardized = (X - X_mean) / X_std

print(f"\nFeature statistics after standardization:")
print(f"  Mean (should be ~0): {X_standardized.mean(axis=0).round(6)}")
print(f"  Std  (should be ~1): {X_standardized.std(axis=0).round(6)}")

# ──────────────────────────────────────────────
# 6. Summary
# ──────────────────────────────────────────────
n_samples, n_features = X_standardized.shape
print(f"\n{'='*50}")
print(f"Preprocessing complete")
print(f"  Samples : {n_samples}")
print(f"  Features: {n_features}")
print(f"  Labels  : +1 (class 1) = {np.sum(y == 1)}, -1 (class 2) = {np.sum(y == -1)}")
print(f"{'='*50}")

# ──────────────────────────────────────────────
# 7. Save processed data for downstream steps
# ──────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(__file__)

# Save features as TSV with a header row
header = "\t".join(FEATURE_NAMES)
np.savetxt(
    os.path.join(OUTPUT_DIR, "X.tsv"),
    X_standardized,
    delimiter="\t",
    header=header,
    comments="",
    fmt="%.8f",
)

# Save labels as a single-column TSV
np.savetxt(
    os.path.join(OUTPUT_DIR, "y.tsv"),
    y,
    delimiter="\t",
    header="label",
    comments="",
    fmt="%d",
)

print(f"\nSaved X.tsv  shape={X_standardized.shape}")
print(f"Saved y.tsv  shape={y.shape}")
