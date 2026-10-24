
# ============================================================
# EV BATTERY HEALTH PREDICTION
# UNIT 2 - NUMPY AND PANDAS PRACTICAL
# Text output only; no graphs
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

DATASET_FILE = "Dataset/EV_Battery_Dataset.csv"
REPORT_FILE = "unit2_numpy_pandas_report.txt"

# ------------------------------------------------------------
# 1. LOAD THE DATA
# ------------------------------------------------------------

if not os.path.exists(DATASET_FILE):
    print("ERROR: Dataset file not found:", DATASET_FILE)
    sys.exit(1)

df = pd.read_csv(DATASET_FILE)

print("=" * 60)
print("EV BATTERY DATA - UNIT 2 PRACTICAL")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("Total records:", len(df))
print("Total columns:", len(df.columns))

# ------------------------------------------------------------
# 2. NUMPY ARRAY FUNDAMENTALS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("1. NUMPY ARRAY FUNDAMENTALS")
print("=" * 60)

sample = np.array([95.0, 88.0, 80.0, 92.0, 85.0])

print("Battery health values:", sample)
print("Array dimension:", sample.ndim)
print("Array shape:", sample.shape)
print("Number of elements:", sample.size)
print("Data type:", sample.dtype)

print("Mean:", np.mean(sample))
print("Median:", np.median(sample))
print("Minimum:", np.min(sample))
print("Maximum:", np.max(sample))
print("Standard deviation:", np.std(sample))

# ------------------------------------------------------------
# 3. NUMPY VECTOR OPERATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("2. NUMPY VECTOR OPERATIONS")
print("=" * 60)

health = np.array([95.0, 88.0, 80.0])

print("Original values:", health)
print("Add 2:", health + 2)
print("Subtract 5:", health - 5)
print("Multiply by 2:", health * 2)
print("Divide by 2:", health / 2)

print(
    "Values below 90:",
    health[health < 90]
)

# ------------------------------------------------------------
# 4. PANDAS DATAFRAME FUNDAMENTALS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("3. PANDAS DATAFRAME FUNDAMENTALS")
print("=" * 60)

print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes.to_string())

print("\nDataset information:")
df.info()

# ------------------------------------------------------------
# 5. SELECT COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. SELECTING COLUMNS")
print("=" * 60)

if "SoH_Percent" in df.columns:
    print("\nFirst five SoH values:")
    print(df["SoH_Percent"].head().to_string(index=False))

if all(
    col in df.columns
    for col in ["Vehicle_Age_Months", "SoH_Percent"]
):
    print("\nAge and SoH:")
    print(
        df[["Vehicle_Age_Months", "SoH_Percent"]]
        .head()
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 6. FILTER ROWS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. FILTERING DATA")
print("=" * 60)

if "SoH_Percent" in df.columns:
    soh = pd.to_numeric(
        df["SoH_Percent"], errors="coerce"
    )

    healthy_values = df.loc[soh >= 90]

    print("Records with SoH >= 90:", len(healthy_values))

    print("\nExample records:")
    print(healthy_values.head().to_string(index=False))

    low_health = df.loc[soh < 80]

    print("\nRecords with SoH < 80:", len(low_health))

# ------------------------------------------------------------
# 7. NUMPY STATISTICS ON REAL DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. NUMPY STATISTICS ON REAL DATA")
print("=" * 60)

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    values = pd.to_numeric(
        df[column], errors="coerce"
    ).to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        continue

    print(f"\nColumn: {column}")
    print("Count:", len(values))
    print("Mean:", np.mean(values))
    print("Median:", np.median(values))
    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))
    print("Standard deviation:", np.std(values, ddof=1)
          if len(values) > 1 else 0.0)

# ------------------------------------------------------------
# 8. CREATE A NEW FEATURE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("7. FEATURE ENGINEERING")
print("=" * 60)

if "Total_Charging_Cycles" in df.columns:
    cycles = pd.to_numeric(
        df["Total_Charging_Cycles"], errors="coerce"
    )

    # Create a descriptive category for learning.
    # These thresholds are examples, not medical or
    # engineering standards for battery condition.
    df["Charging_Cycle_Group"] = pd.cut(
        cycles,
        bins=[-np.inf, 200, 500, 800, np.inf],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
    )

    print("Charging-cycle groups:")
    print(
        df["Charging_Cycle_Group"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nExample records:")
    columns_to_show = [
        col for col in [
            "Total_Charging_Cycles",
            "Charging_Cycle_Group"
        ]
        if col in df.columns
    ]

    print(df[columns_to_show].head().to_string(index=False))

else:
    print("Total_Charging_Cycles column not found.")

# ------------------------------------------------------------
# 9. GROUPBY ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("8. GROUPBY ANALYSIS")
print("=" * 60)

if (
    "Battery_Type" in df.columns
    and "SoH_Percent" in df.columns
):
    group_summary = df.groupby(
        "Battery_Type",
        observed=True
    )["SoH_Percent"].agg(
        ["count", "mean", "median", "min", "max"]
    )

    print("\nBattery health by battery type:")
    print(group_summary.to_string())

else:
    print("Required columns for group analysis not found.")

# ------------------------------------------------------------
# 10. CORRELATION AND COVARIANCE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("9. CORRELATION AND COVARIANCE")
print("=" * 60)

selected_columns = [
    col for col in [
        "Vehicle_Age_Months",
        "Total_Charging_Cycles",
        "Temperature_C",
        "Internal_Resistance_Ohm",
        "SoH_Percent"
    ]
    if col in df.columns
]

if len(selected_columns) >= 2:
    selected_data = df[selected_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    print("\nPearson correlation matrix:")
    print(selected_data.corr().round(3).to_string())

    print("\nCovariance matrix:")
    print(selected_data.cov().round(3).to_string())

else:
    print("Not enough selected numerical columns.")

# ------------------------------------------------------------
# 11. SAVE THE REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("10. SAVE OUTPUT")
print("=" * 60)

report_lines = [
    "EV BATTERY HEALTH PREDICTION - UNIT 2 REPORT",
    "",
    f"Dataset records: {df.shape[0]}",
    f"Dataset columns: {df.shape[1]}",
    "",
    "Column names:",
    *df.columns.astype(str).tolist(),
    "",
    "Numerical descriptive statistics:",
    df.describe().to_string(),
    "",
    "Missing values by column:",
    df.isna().sum().to_string()
]

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))

print("Report saved to:", os.path.abspath(REPORT_FILE))
print("UNIT 2 PRACTICAL COMPLETED")
