"""
Phase 3b (Block D): Late-delivery classification.

Reads:
    A_preprocessing_eda/outputs/cleaned_supply_chain.csv

Writes:
    D_late_model_report/outputs/late_model_results.csv
    D_late_model_report/outputs/late_predictions.csv
    D_late_model_report/outputs/late_odds_ratios.csv
    D_late_model_report/outputs/late_model_summary.txt
    D_late_model_report/outputs/plots/confusion_matrix.png
    D_late_model_report/outputs/plots/roc_curve.png

Project rules:
    - Target: Late_delivery_risk
    - 80/20 stratified split
    - random_state = 42
    - Logistic Regression, max_iter = 1000
    - Scale numeric variables using training data only
    - Use Shipping Mode OR Days for shipment (scheduled), never both
    - Keep Order Status out
    - No delivery/profit leakage
"""

import sys
from pathlib import Path

# Ensure Unicode characters (✓, —) print correctly on Windows cp1252 terminals.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Paths and constants
# ============================================================

SEED = 42

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "A_preprocessing_eda" / "outputs" / "cleaned_supply_chain.csv"

OUT = ROOT / "D_late_model_report" / "outputs"
PLOTS = OUT / "plots"

OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Feature specification
# ============================================================

TARGET = "Late_delivery_risk"

NUMERIC_FEATURES = [
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate",
]

DUMMY_GROUPS = {
    "Shipping Mode": "Shipping Mode_",
    "Market": "Market_",
    "Order Region": "Order Region_",
    "Customer Segment": "Customer Segment_",
    "Type": "Type_",
    "Category Name_grouped": "Category Name_grouped_",
}

# Columns that MUST NOT become model features
FORBIDDEN_FEATURES = {
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Benefit_signed_log",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
    "loss",
    "late",
}


# ============================================================
# 3. Read only the columns we need
# ============================================================

print("=" * 80)
print("PHASE 3b — LATE DELIVERY MODEL")
print("=" * 80)

print("\nLoading cleaned dataset...")

# Read header first so that we can discover the existing dummy columns.
header = pd.read_csv(DATA, nrows=0)
all_columns = list(header.columns)

if TARGET not in all_columns:
    raise ValueError(f"Required target column '{TARGET}' was not found.")

# Discover existing one-hot columns created during Phase 1.
dummy_columns = {}
reference_levels = {}

for group_name, prefix in DUMMY_GROUPS.items():
    cols = [c for c in all_columns if c.startswith(prefix)]

    if not cols:
        raise ValueError(
            f"No one-hot columns found for '{group_name}' using prefix '{prefix}'."
        )

    dummy_columns[group_name] = cols

# These are the columns actually read from the CSV.
required_columns = (
    NUMERIC_FEATURES
    + [TARGET, "Shipping Mode", "Days for shipment (scheduled)"]
    + [
        c
        for group_cols in dummy_columns.values()
        for c in group_cols
    ]
)

# Remove accidental duplicates while preserving order.
required_columns = list(dict.fromkeys(required_columns))

df = pd.read_csv(
    DATA,
    usecols=required_columns,
    low_memory=False,
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# 4. Data integrity checks
# ============================================================

print("\nRunning data integrity checks...")

# No missing values in model inputs or target.
if df.isnull().any().any():
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])
    raise ValueError("Missing values detected in modelling data.")

# Target must be binary.
target_values = sorted(df[TARGET].unique().tolist())

if target_values != [0, 1]:
    raise ValueError(
        f"Unexpected target values in {TARGET}: {target_values}"
    )

# Phase 1's independent validator establishes that `late` equals
# `Late_delivery_risk` for the shared cleaned file. This model deliberately
# reads only the official target, so the duplicate target cannot become a feature.

# Confirm the scheduled-days / shipping-mode relationship.
schedule_map = (
    df.groupby("Shipping Mode")["Days for shipment (scheduled)"]
    .unique()
    .to_dict()
)

expected_schedule_map = {
    "Same Day": {0},
    "First Class": {1},
    "Second Class": {2},
    "Standard Class": {4},
}

for mode, expected_days in expected_schedule_map.items():
    if mode not in schedule_map:
        raise ValueError(f"Missing expected Shipping Mode: {mode}")

    observed = set(schedule_map[mode])

    if observed != expected_days:
        raise ValueError(
            f"Unexpected scheduled-day mapping for {mode}: "
            f"observed={observed}, expected={expected_days}"
        )

# Make sure the forbidden fields aren't accidentally included.
used_feature_names = set(NUMERIC_FEATURES)

for cols in dummy_columns.values():
    used_feature_names.update(cols)

unexpected = used_feature_names.intersection(FORBIDDEN_FEATURES)

if unexpected:
    raise ValueError(
        f"Forbidden/leaking features detected: {sorted(unexpected)}"
    )

print("✓ No missing values")
print("✓ Target is binary")
print("✓ Shipping Mode / scheduled-day mapping verified")
print("✓ No forbidden model features")


# ============================================================
# 5. Choose one reference level per categorical group
# ============================================================

print("\nSelecting reference levels...")

selected_dummy_columns = []

for group_name, cols in dummy_columns.items():

    # Most frequent category becomes the reference level.
    # This mirrors the approach used in the team's Phase 3a code.
    frequencies = df[cols].sum().sort_values(ascending=False)

    reference_dummy = frequencies.index[0]

    reference_levels[group_name] = reference_dummy

    usable = [c for c in cols if c != reference_dummy]

    selected_dummy_columns.extend(usable)

    readable_reference = reference_dummy.replace(
        DUMMY_GROUPS[group_name], ""
    )

    print(f"{group_name}: reference = {readable_reference}")

MODEL_FEATURES = NUMERIC_FEATURES + selected_dummy_columns


# ============================================================
# 6. Build X and y
# ============================================================

X = df[MODEL_FEATURES].astype(float)
y = df[TARGET].astype(int)

print("\nFinal model feature count:", len(MODEL_FEATURES))

print("\nTarget distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))


# ============================================================
# 7. Train/test split
# ============================================================

print("\nCreating 80/20 stratified split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows : {len(X_test):,}")


# ============================================================
# 8. Scale numeric features — TRAINING DATA ONLY
# ============================================================

print("\nScaling numeric features using training data only...")

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(
    X_train[NUMERIC_FEATURES]
)

X_test_scaled[NUMERIC_FEATURES] = scaler.transform(
    X_test[NUMERIC_FEATURES]
)

print("✓ Scaler fitted only on training data")


# ============================================================
# 9. Logistic regression
# ============================================================

print("\nTraining logistic regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
)

model.fit(X_train_scaled, y_train)

print("✓ Model trained")


# ============================================================
# 10. Predictions
# ============================================================

# Probability of class 1 = probability that the order is late.
test_prob = model.predict_proba(X_test_scaled)[:, 1]

# Standard 0.50 classification threshold.
test_pred = (test_prob >= 0.50).astype(int)


# ============================================================
# 11. Metrics
# ============================================================

accuracy = accuracy_score(y_test, test_pred)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    test_prob,
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_pred,
).ravel()

print("\n" + "=" * 80)
print("TEST-SET RESULTS")
print("=" * 80)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
)


# ============================================================
# 12. Save model results
# ============================================================

results = pd.DataFrame(
    [
        {
            "model": "Logistic Regression",
            "split": "test",
            "random_state": SEED,
            "n_test": len(y_test),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        }
    ]
)

results.to_csv(
    OUT / "late_model_results.csv",
    index=False,
)


# ============================================================
# 13. Odds ratios
# ============================================================

print("\nCalculating odds ratios...")

odds_ratios = pd.DataFrame(
    {
        "feature": MODEL_FEATURES,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0]),
    }
)

# Sort by distance from OR=1 so the strongest effects are visible first.
odds_ratios["abs_log_odds_ratio"] = np.abs(
    np.log(odds_ratios["odds_ratio"])
)

odds_ratios = odds_ratios.sort_values(
    "abs_log_odds_ratio",
    ascending=False,
).drop(columns=["abs_log_odds_ratio"])

odds_ratios.to_csv(
    OUT / "late_odds_ratios.csv",
    index=False,
)

print("\nTop odds ratios:")
print(odds_ratios.head(10).round(4).to_string(index=False))


# ============================================================
# 14. Predictions file
# ============================================================

print("\nCreating predictions file for all rows...")

# Transform all rows using the scaler already fitted on training data.
X_all_scaled = X.copy()

X_all_scaled[NUMERIC_FEATURES] = scaler.transform(
    X[NUMERIC_FEATURES]
)

all_prob = model.predict_proba(X_all_scaled)[:, 1]
all_pred = (all_prob >= 0.50).astype(int)

# Determine which original rows belong to the test set.
train_indices = set(X_train.index)
test_indices = set(X_test.index)

split_labels = np.where(
    X.index.isin(test_indices),
    "test",
    "train",
)

predictions = pd.DataFrame(
    {
        "row_id": X.index,
        "split": split_labels,
        "late_actual": y.values,
        "late_probability": all_prob,
        "late_prediction": all_pred,
    }
)

predictions.to_csv(
    OUT / "late_predictions.csv",
    index=False,
)


# ============================================================
# 15. Confusion matrix plot
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

cm = np.array(
    [
        [tn, fp],
        [fn, tp],
    ]
)

im = ax.imshow(cm)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
        )

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Predicted On-time", "Predicted Late"]
)

ax.set_yticklabels(
    ["Actual On-time", "Actual Late"]
)

ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
ax.set_title("Late-Delivery Logistic Regression\nConfusion Matrix")

plt.tight_layout()

plt.savefig(
    PLOTS / "confusion_matrix.png",
    dpi=150,
)

plt.close()


# ============================================================
# 16. ROC curve
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    test_prob,
)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})",
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Late-Delivery Model")
ax.legend()

plt.tight_layout()

plt.savefig(
    PLOTS / "roc_curve.png",
    dpi=150,
)

plt.close()


# ============================================================
# 17. Human-readable summary
# ============================================================

summary_path = OUT / "late_model_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("PHASE 3b — LATE DELIVERY MODEL\n")
    f.write("=" * 70 + "\n\n")

    f.write("Target:\n")
    f.write("  Late_delivery_risk (1 = late, 0 = not late)\n\n")

    f.write("Split:\n")
    f.write("  80/20 stratified train/test split\n")
    f.write(f"  random_state = {SEED}\n\n")

    f.write("Features:\n")
    for feature in MODEL_FEATURES:
        f.write(f"  - {feature}\n")

    f.write("\nFeatures deliberately excluded:\n")
    for feature in sorted(FORBIDDEN_FEATURES):
        f.write(f"  - {feature}\n")

    f.write("\nReference levels:\n")
    for group, reference in reference_levels.items():
        readable = reference.replace(
            DUMMY_GROUPS[group],
            "",
        )
        f.write(f"  - {group}: {readable}\n")

    f.write("\nTest-set metrics:\n")
    f.write(f"  Accuracy : {accuracy:.6f}\n")
    f.write(f"  Precision: {precision:.6f}\n")
    f.write(f"  Recall   : {recall:.6f}\n")
    f.write(f"  F1       : {f1:.6f}\n")
    f.write(f"  ROC-AUC  : {roc_auc:.6f}\n\n")

    f.write("Confusion matrix:\n")
    f.write(f"  TN = {tn:,}\n")
    f.write(f"  FP = {fp:,}\n")
    f.write(f"  FN = {fn:,}\n")
    f.write(f"  TP = {tp:,}\n")

print("\n" + "=" * 80)
print("PHASE 3b COMPLETE")
print("=" * 80)

print(f"\nResults saved to:")
print(f"  {OUT}")

print("\nFiles created:")
print("  - late_model_results.csv")
print("  - late_predictions.csv")
print("  - late_odds_ratios.csv")
print("  - late_model_summary.txt")
print("  - plots/confusion_matrix.png")
print("  - plots/roc_curve.png")
