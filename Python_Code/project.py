import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Step 1: Load the supplied Kaggle CSV from the project folder.
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
print("First five records from the supplied synthetic dataset:")
print(df.head().to_string(index=False))

# Step 2: Inspect the table, its data types, summary values, skewness, and kurtosis.
print("\nDataset shape (rows, columns):", df.shape)
print("\nDataset information:")
df.info()
print("\nSummary of all columns:")
print(df.describe(include="all").to_string())

numeric_columns = df.select_dtypes(include="number").columns
text_columns = df.select_dtypes(include=["object", "category"]).columns
print("\nNumeric columns:", numeric_columns.tolist())
print("Text columns:", text_columns.tolist())
print("\nSkewness of numeric columns:")
print(df[numeric_columns].skew().to_string())
print("\nKurtosis of numeric columns:")
print(df[numeric_columns].kurt().to_string())

# Step 3: Record data quality before cleaning.
feature_columns = [
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
]
before_rows = len(df)
before_missing = int(df.isna().sum().sum())
before_duplicates = int(df.duplicated().sum())
before_outliers = 0

# Count IQR outliers only in the five planned input measurements.
for column in feature_columns:
    first_quartile = df[column].quantile(0.25)
    third_quartile = df[column].quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_limit = first_quartile - 1.5 * interquartile_range
    upper_limit = third_quartile + 1.5 * interquartile_range
    before_outliers += int(
        ((df[column] < lower_limit) | (df[column] > upper_limit)).sum()
    )

# Remove exact duplicate rows before filling any missing values.
df = df.drop_duplicates().copy()

# Fill numeric gaps with the median and text gaps with the most common value.
for column in df.columns:
    if df[column].isna().sum() > 0:
        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())

# Cap extreme input measurements at their IQR limits; keep the SOH target unchanged.
for column in feature_columns:
    first_quartile = df[column].quantile(0.25)
    third_quartile = df[column].quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_limit = first_quartile - 1.5 * interquartile_range
    upper_limit = third_quartile + 1.5 * interquartile_range
    df[column] = df[column].clip(lower=lower_limit, upper=upper_limit)

# Count remaining input outliers after cleaning.
after_outliers = 0
for column in feature_columns:
    first_quartile = df[column].quantile(0.25)
    third_quartile = df[column].quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_limit = first_quartile - 1.5 * interquartile_range
    upper_limit = third_quartile + 1.5 * interquartile_range
    after_outliers += int(
        ((df[column] < lower_limit) | (df[column] > upper_limit)).sum()
    )

before_after = pd.DataFrame(
    {
        "Check": [
            "Rows (records)",
            "Missing cells (values)",
            "Duplicate rows (rows)",
            "IQR outlier values in five input measurements (values)",
        ],
        "Before cleaning": [
            before_rows,
            before_missing,
            before_duplicates,
            before_outliers,
        ],
        "After cleaning": [
            len(df),
            int(df.isna().sum().sum()),
            int(df.duplicated().sum()),
            after_outliers,
        ],
    }
)
print("\nBefore-and-after data-quality table:")
print(before_after.to_string(index=False))

# Step 4: Save the required EDA charts in the Reports folder.
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="SoH_Percent", bins=30, kde=True)
plt.title("Battery State of Health Distribution")
plt.xlabel("State of Health (%)")
plt.ylabel("Number of records (rows)")
plt.tight_layout()
plt.savefig("Reports/soh_histogram.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot(df["SoH_Percent"], orientation="horizontal")
plt.title("State of Health Box Plot")
plt.xlabel("State of Health (%)")
plt.tight_layout()
plt.savefig("Reports/soh_boxplot.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Total_Charging_Cycles", y="SoH_Percent", s=18)
plt.title("Charging Cycles and State of Health")
plt.xlabel("Total charging cycles (cycles)")
plt.ylabel("State of Health (%)")
plt.tight_layout()
plt.savefig("Reports/cycles_vs_soh.png", dpi=150)
plt.close()

correlation = df.select_dtypes(include="number").corr()
plt.figure(figsize=(11, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Columns")
plt.tight_layout()
plt.savefig("Reports/correlation_heatmap.png", dpi=150)
plt.close()

status_counts = df["Battery_Status"].value_counts()
plt.figure(figsize=(7, 6))
plt.pie(
    status_counts,
    labels=status_counts.index,
    autopct="%.2f%%",
    startangle=90,
)
plt.title("Battery Status Share (%)")
plt.tight_layout()
plt.savefig("Reports/battery_status_pie.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Battery_Type")
plt.title("Number of Records by Battery Type")
plt.xlabel("Battery type (category)")
plt.ylabel("Number of records (rows)")
plt.tight_layout()
plt.savefig("Reports/battery_type_bar.png", dpi=150)
plt.close()
print("\nSaved six EDA charts in the Reports folder.")

# Step 5: Fit one simple line to charging cycles and SOH for teaching y = mx + b.
simple_X = df[["Total_Charging_Cycles"]]
simple_y = df["SoH_Percent"]
simple_model = LinearRegression()
simple_model.fit(simple_X, simple_y)
simple_prediction = simple_model.predict(simple_X)
simple_slope = simple_model.coef_[0]
simple_intercept = simple_model.intercept_
simple_r2 = r2_score(simple_y, simple_prediction)
simple_rss = np.sum((simple_y - simple_prediction) ** 2)
print("\nSimple linear regression (fitted on all cleaned records for illustration):")
print(f"Equation: SOH (%) = {simple_slope:.6f} (% per cycle) * cycles + {simple_intercept:.6f} (%)")
print(f"R-squared (unitless): {simple_r2:.6f}")
print(f"RSS (% squared): {simple_rss:.6f}")

simple_plot_data = df.sort_values("Total_Charging_Cycles")
plt.figure(figsize=(8, 5))
plt.scatter(
    simple_plot_data["Total_Charging_Cycles"],
    simple_plot_data["SoH_Percent"],
    s=14,
    alpha=0.45,
    label="Observed records",
)
plt.plot(
    simple_plot_data["Total_Charging_Cycles"],
    simple_model.predict(simple_plot_data[["Total_Charging_Cycles"]]),
    color="red",
    label="Fitted regression line",
)
plt.title("Simple Linear Regression: Cycles and SOH")
plt.xlabel("Total charging cycles (cycles)")
plt.ylabel("State of Health (%)")
plt.legend()
plt.tight_layout()
plt.savefig("Reports/simple_regression_line.png", dpi=150)
plt.close()

# Step 6: Select measurements and target, then keep 20% of records for final testing.
X = df[feature_columns]
y = df["SoH_Percent"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
print("\nInput features and units:")
print("Vehicle_Age_Months (months)")
print("Total_Charging_Cycles (cycles)")
print("Avg_Temperature_C (degrees Celsius, °C)")
print("Fast_Charge_Ratio (ratio from 0 to 1, unitless)")
print("Avg_Discharge_Rate_C (C-rate)")
print("Target: SoH_Percent (percent, %)")
print("Excluded from model inputs: Vehicle_ID, Battery_Status, Battery_Capacity_kWh,")
print("Internal_Resistance_Ohm, and text categories (leakage, identifier, or simplicity).")
print(f"Training records (rows): {len(X_train)}")
print(f"Testing records (rows): {len(X_test)}")

# Step 7: Scale features inside each fold, then compare three regression models.
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
linear_model = make_pipeline(StandardScaler(), LinearRegression())
ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000))

linear_cv_scores = cross_val_score(
    linear_model, X_train, y_train, cv=kfold, scoring="r2"
)
ridge_cv_scores = cross_val_score(
    ridge_model, X_train, y_train, cv=kfold, scoring="r2"
)
lasso_cv_scores = cross_val_score(
    lasso_model, X_train, y_train, cv=kfold, scoring="r2"
)
print("\nFive-fold cross-validation R-squared scores (unitless):")
print("Linear:", np.round(linear_cv_scores, 4))
print("Ridge (L2):", np.round(ridge_cv_scores, 4))
print("Lasso (L1):", np.round(lasso_cv_scores, 4))

# Step 8: Fit each model on training rows and compare it on the untouched test rows.
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
linear_test_prediction = linear_model.predict(X_test)
ridge_test_prediction = ridge_model.predict(X_test)
lasso_test_prediction = lasso_model.predict(X_test)

results = pd.DataFrame(
    [
        {
            "Model": "Linear",
            "Mean CV R2 (unitless)": linear_cv_scores.mean(),
            "MAE (%)": mean_absolute_error(y_test, linear_test_prediction),
            "MSE (% squared)": mean_squared_error(y_test, linear_test_prediction),
            "RMSE (%)": mean_squared_error(y_test, linear_test_prediction) ** 0.5,
            "Test R2 (unitless)": r2_score(y_test, linear_test_prediction),
        },
        {
            "Model": "Ridge (L2)",
            "Mean CV R2 (unitless)": ridge_cv_scores.mean(),
            "MAE (%)": mean_absolute_error(y_test, ridge_test_prediction),
            "MSE (% squared)": mean_squared_error(y_test, ridge_test_prediction),
            "RMSE (%)": mean_squared_error(y_test, ridge_test_prediction) ** 0.5,
            "Test R2 (unitless)": r2_score(y_test, ridge_test_prediction),
        },
        {
            "Model": "Lasso (L1)",
            "Mean CV R2 (unitless)": lasso_cv_scores.mean(),
            "MAE (%)": mean_absolute_error(y_test, lasso_test_prediction),
            "MSE (% squared)": mean_squared_error(y_test, lasso_test_prediction),
            "RMSE (%)": mean_squared_error(y_test, lasso_test_prediction) ** 0.5,
            "Test R2 (unitless)": r2_score(y_test, lasso_test_prediction),
        },
    ]
)
results = results.sort_values("Mean CV R2 (unitless)", ascending=False)
print("\nModel comparison (CV R2 is unitless; errors are in SOH percentage units):")
print(results.round(4).to_string(index=False))
best_model_name = results.iloc[0]["Model"]
model_choices = {
    "Linear": linear_model,
    "Ridge (L2)": ridge_model,
    "Lasso (L1)": lasso_model,
}
best_model = model_choices[best_model_name]
print("\nBest model by mean five-fold CV R-squared:", best_model_name)

# Step 9: Save the selected model and predict one held-out sample.
with open("Python_Code/best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

sample_input = X_test.iloc[[0]]
sample_actual = y_test.iloc[0]
sample_prediction = best_model.predict(sample_input)[0]
print("\nHeld-out sample input and its units:")
print(sample_input.to_string(index=False))
print("Age unit: months; cycles unit: cycles; temperature unit: °C;")
print("fast-charge ratio unit: ratio from 0 to 1; discharge unit: C-rate.")
print(f"Actual SOH: {sample_actual:.2f} %")
print(f"Predicted SOH: {sample_prediction:.2f} %")
print("\nSaved the best model to Python_Code/best_model.pkl")

# Create the teacher's report sections as a PDF using the measured run results.
with PdfPages("Reports/Project_Report.pdf") as report_pdf:
    # Add the cover page and leave personal details for the student to complete.
    report_figure = plt.figure(figsize=(8.27, 11.69))
    report_figure.text(
        0.5,
        0.82,
        "EV BATTERY HEALTH PREDICTION",
        ha="center",
        fontsize=22,
        weight="bold",
    )
    report_figure.text(
        0.5,
        0.76,
        "Mini Project Report",
        ha="center",
        fontsize=16,
    )
    report_figure.text(
        0.5,
        0.66,
        "BCA Sem-III | Course 303-05: Fundamentals of Machine Learning",
        ha="center",
        fontsize=11,
    )
    report_figure.text(
        0.5,
        0.55,
        "Student name: ______________________________\n"
        "Roll number: _______________________________\n"
        "Faculty guide: ______________________________\n"
        "Academic year: ______________________________",
        ha="center",
        va="center",
        fontsize=12,
        linespacing=2,
    )
    report_figure.text(
        0.5,
        0.34,
        "Dataset: Kaggle Electric Vehicle (EV) Battery Degradation and Charge\n"
        "Important: Kaggle describes the supplied 10,000 records as synthetic simulations.",
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )
    report_figure.text(0.5, 0.08, "Project report generated from the supplied CSV", ha="center", fontsize=9)
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

    # Include the certificate page without suggesting approval or signatures are complete.
    report_figure = plt.figure(figsize=(8.27, 11.69))
    report_figure.text(0.5, 0.88, "CERTIFICATE", ha="center", fontsize=18, weight="bold")
    report_figure.text(
        0.12,
        0.72,
        "This is to certify that ______________________________ (student name),\n"
        "Roll number __________________, has prepared the mini project titled\n"
        "\"EV Battery Health Prediction\" as part of BCA Sem-III, Course 303-05:\n"
        "Fundamentals of Machine Learning.\n\n"
        "The project uses a synthetic Kaggle dataset for learning and demonstration.\n"
        "Faculty approval of the topic and dataset must be confirmed before submission.",
        fontsize=12,
        va="top",
        linespacing=1.7,
    )
    report_figure.text(
        0.12,
        0.30,
        "Student signature: ____________________     Date: ______________\n\n"
        "Faculty signature: _____________________     Date: ______________\n\n"
        "Department seal: _______________________",
        fontsize=11,
        va="top",
        linespacing=1.7,
    )
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

    # State the problem, objectives, and real-world-data limitation.
    report_figure = plt.figure(figsize=(8.27, 11.69))
    report_figure.text(0.08, 0.93, "PROBLEM STATEMENT", fontsize=16, weight="bold")
    report_figure.text(
        0.08,
        0.86,
        "Estimate battery State of Health (SoH, %) from selected vehicle and\n"
        "charging measurements using supervised regression.",
        fontsize=11,
        va="top",
        linespacing=1.6,
    )
    report_figure.text(0.08, 0.73, "OBJECTIVES", fontsize=15, weight="bold")
    report_figure.text(
        0.10,
        0.68,
        "1. Inspect and clean the supplied CSV.\n"
        "2. Explore its measurements with plots and summary statistics.\n"
        "3. Fit and compare Linear, Ridge (L2), and Lasso (L1) regression.\n"
        "4. Evaluate with five-fold cross-validation and held-out test metrics.\n"
        "5. Save the selected model and demonstrate one held-out prediction.",
        fontsize=11,
        va="top",
        linespacing=1.8,
    )
    report_figure.text(0.08, 0.45, "DATASET APPROVAL AND LIMITATION", fontsize=15, weight="bold")
    report_figure.text(
        0.10,
        0.40,
        "The course instructions require a real-world dataset and faculty approval.\n"
        "The Kaggle author describes this supplied dataset as synthetic simulations.\n"
        "It therefore does not satisfy the real-world-data requirement unless the\n"
        "faculty explicitly approves it for this learning demonstration. Its scores\n"
        "must not be presented as evidence of real battery diagnostic performance.",
        fontsize=11,
        va="top",
        linespacing=1.7,
    )
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

    # Describe all supplied columns and their units or categories.
    report_figure, report_axis = plt.subplots(figsize=(8.27, 11.69))
    report_figure.suptitle("DATASET DESCRIPTION", fontsize=16, weight="bold", y=0.96)
    dataset_description = [
        ("Vehicle_ID", "Vehicle record identifier", "ID"),
        ("Car_Model", "Vehicle model name", "Category"),
        ("Battery_Type", "Battery chemistry (LFP or NMC)", "Category"),
        ("Battery_Capacity_kWh", "Listed battery capacity", "kWh"),
        ("Vehicle_Age_Months", "Vehicle age", "months"),
        ("Total_Charging_Cycles", "Cumulative charge/discharge count", "cycles"),
        ("Avg_Temperature_C", "Average operating temperature", "°C"),
        ("Fast_Charge_Ratio", "Share of fast-charging sessions", "0 to 1, unitless"),
        ("Avg_Discharge_Rate_C", "Average discharge rate", "C-rate"),
        ("Driving_Style", "Driving-style label", "Category"),
        ("Internal_Resistance_Ohm", "Battery internal resistance", "ohms (Ω)"),
        ("SoH_Percent", "State of Health; prediction target", "%"),
        ("Battery_Status", "Health-status label", "Category"),
    ]
    report_axis.axis("off")
    report_axis.table(
        cellText=dataset_description,
        colLabels=["Column", "Meaning", "Unit or kind"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.34, 0.43, 0.20],
        bbox=[0.02, 0.14, 0.96, 0.76],
        fontsize=8,
    )
    report_figure.text(
        0.06,
        0.07,
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns. "
        "Model target: SoH_Percent (%).",
        fontsize=10,
    )
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

    # Show the actual before-and-after cleaning counts from this run.
    report_figure, report_axis = plt.subplots(figsize=(8.27, 11.69))
    report_figure.suptitle("DATA PREPROCESSING", fontsize=16, weight="bold", y=0.94)
    report_axis.axis("off")
    report_axis.table(
        cellText=before_after.values.tolist(),
        colLabels=before_after.columns.tolist(),
        cellLoc="left",
        colLoc="left",
        colWidths=[0.58, 0.20, 0.20],
        bbox=[0.03, 0.58, 0.94, 0.25],
        fontsize=9,
    )
    report_figure.text(
        0.08,
        0.48,
        "Cleaning decisions",
        fontsize=13,
        weight="bold",
    )
    report_figure.text(
        0.10,
        0.43,
        "• Exact duplicate rows were removed.\n"
        "• Missing numeric values use the median; missing text values use the mode.\n"
        "• IQR limits were applied to five model-input measurements only.\n"
        "• SoH target values were not capped; all remaining records were retained.",
        fontsize=10,
        va="top",
        linespacing=1.8,
    )
    report_figure.text(
        0.08,
        0.22,
        "Selected inputs: vehicle age (months), charging cycles (cycles), average\n"
        "temperature (°C), fast-charge ratio (unitless), and discharge rate (C-rate).\n"
        "Excluded: identifier, text categories, Battery_Status, internal resistance,\n"
        "and capacity (possible target leakage or direct target association).",
        fontsize=10,
        va="top",
        linespacing=1.6,
    )
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

    # Place the six required EDA plots together in a landscape report page.
    report_figure, report_axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    report_figure.suptitle("EXPLORATORY DATA ANALYSIS", fontsize=16, weight="bold")
    chart_files = [
        ("Reports/soh_histogram.png", "SoH distribution (%)"),
        ("Reports/soh_boxplot.png", "SoH box plot (%)"),
        ("Reports/cycles_vs_soh.png", "Cycles and SoH (cycles, %)"),
        ("Reports/correlation_heatmap.png", "Numeric-column correlation (unitless)"),
        ("Reports/battery_status_pie.png", "Battery status share (%)"),
        ("Reports/battery_type_bar.png", "Records by battery type (rows)"),
    ]
    for chart_axis, (chart_file, chart_title) in zip(report_axes.flat, chart_files):
        chart_axis.imshow(plt.imread(chart_file))
        chart_axis.set_title(chart_title, fontsize=9)
        chart_axis.axis("off")
    report_figure.tight_layout(rect=[0, 0, 1, 0.94])
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

    # Explain the simple regression and show its generated graph.
    report_figure = plt.figure(figsize=(8.27, 11.69))
    report_figure.text(0.08, 0.93, "SIMPLE LINEAR REGRESSION", fontsize=16, weight="bold")
    report_figure.text(
        0.08,
        0.86,
        f"SOH (%) = {simple_slope:.6f} (% per cycle) × cycles + "
        f"{simple_intercept:.6f} (%)\n"
        f"R-squared (unitless): {simple_r2:.6f}\n"
        f"Residual sum of squares (%²): {simple_rss:.6f}",
        fontsize=11,
        va="top",
        linespacing=1.8,
    )
    report_axis = report_figure.add_axes([0.10, 0.30, 0.80, 0.45])
    report_axis.imshow(plt.imread("Reports/simple_regression_line.png"))
    report_axis.axis("off")
    report_figure.text(
        0.08,
        0.18,
        "This one-feature fit is for teaching y = mx + b. Its R-squared is calculated\n"
        "on the cleaned data used to draw the illustration, not the held-out test set.",
        fontsize=10,
        va="top",
        linespacing=1.6,
    )
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

    # Record training, validation, model comparison, and held-out test results.
    report_figure, report_axis = plt.subplots(figsize=(8.27, 11.69))
    report_figure.suptitle("MODEL BUILDING AND EVALUATION", fontsize=16, weight="bold", y=0.95)
    report_figure.text(
        0.08,
        0.86,
        f"Training rows: {len(X_train)} | Held-out test rows: {len(X_test)}\n"
        "A StandardScaler is fitted inside each training fold. Five-fold cross-validation\n"
        "compares Linear Regression, Ridge (L2), and Lasso (L1).",
        fontsize=10,
        va="top",
        linespacing=1.6,
    )
    report_axis.axis("off")
    report_axis.table(
        cellText=results.round(4).values.tolist(),
        colLabels=results.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.48, 0.98, 0.24],
        fontsize=8,
    )
    report_figure.text(
        0.08,
        0.38,
        f"Selected by mean five-fold CV R²: {best_model_name}\n"
        "MAE and RMSE are in SoH percentage points (%); MSE is in %²; R² is unitless.\n"
        "Linear and Ridge can tie at the displayed precision. Scores describe only\n"
        "performance on this synthetic dataset and split.",
        fontsize=10,
        va="top",
        linespacing=1.7,
    )
    report_figure.text(
        0.08,
        0.17,
        "Model-building code pattern:\n"
        "X = df[feature_columns]\n"
        "y = df['SoH_Percent']\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)\n"
        "model = make_pipeline(StandardScaler(), LinearRegression())",
        fontsize=9,
        family="monospace",
        va="top",
        linespacing=1.6,
    )
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

    # Summarize this run, future work, and the dataset reference.
    report_figure = plt.figure(figsize=(8.27, 11.69))
    report_figure.text(0.08, 0.93, "RESULTS, CONCLUSION, AND FUTURE SCOPE", fontsize=15, weight="bold")
    report_figure.text(
        0.08,
        0.85,
        f"Best model: {best_model_name}\n"
        f"Held-out sample actual SoH: {sample_actual:.2f} %\n"
        f"Held-out sample predicted SoH: {sample_prediction:.2f} %\n"
        "The saved estimator is Python_Code/best_model.pkl.",
        fontsize=11,
        va="top",
        linespacing=1.8,
    )
    report_figure.text(0.08, 0.66, "Conclusion", fontsize=13, weight="bold")
    report_figure.text(
        0.10,
        0.61,
        "The project demonstrates a complete beginner regression workflow: inspection,\n"
        "cleaning, visual EDA, model training, cross-validation, test evaluation, and\n"
        "saving a fitted estimator. Linear and Ridge performed similarly in this run.\n"
        "The synthetic data prevents a real-world battery-health conclusion.",
        fontsize=10,
        va="top",
        linespacing=1.7,
    )
    report_figure.text(0.08, 0.40, "Future scope", fontsize=13, weight="bold")
    report_figure.text(
        0.10,
        0.35,
        "Obtain measured battery test records and faculty approval; validate on vehicles\n"
        "or batteries not seen during training; document units and collection methods;\n"
        "check leakage and generalization; and compare predictions with a suitable\n"
        "battery-health measurement before considering practical use.",
        fontsize=10,
        va="top",
        linespacing=1.7,
    )
    report_figure.text(
        0.08,
        0.16,
        "Dataset source: Kaggle, Electric Vehicle (EV) Battery Degradation and Charge,\n"
        "https://www.kaggle.com/datasets/bertnardomariouskono/\n"
        "electric-vehicle-ev-battery-degradation-and-charge\n"
        "The Kaggle author describes the supplied data as synthetic simulations.",
        fontsize=8,
        va="top",
        linespacing=1.5,
    )
    report_pdf.savefig(report_figure)
    plt.close(report_figure)

print("Saved the course-format report to Reports/Project_Report.pdf")
