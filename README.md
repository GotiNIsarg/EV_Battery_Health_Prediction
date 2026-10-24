# Electric Vehicle (EV) Battery Degradation & Charge

## State of Health (SoH) Estimation for NMC & LFP Batteries

[![Dataset on Kaggle](https://img.shields.io/badge/Kaggle-Download%20Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge/data)

## 1. Project Overview

Estimating the State of Health (SoH) and Remaining Useful Life (RUL) of electric-vehicle batteries is important for battery monitoring, maintenance planning, and battery-management systems.

This project uses a **synthetic dataset of 10,000 records** to explore battery-health prediction and related machine-learning tasks. The dataset describes two battery chemistries: Lithium Iron Phosphate (LFP) and Nickel Manganese Cobalt (NMC).

> **Important:** The dataset is synthetic. Results from models trained on it should not be treated as validated real-world battery diagnostics without testing against independent real-world measurements.

## 2. Dataset Download

**Download the dataset:**  
[Electric Vehicle (EV) Battery Degradation & Charge — Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge/data)

Dataset details reported on the Kaggle page:

- **File:** `ev_battery_degradation_v1.csv`
- **Samples:** 10,000
- **Columns:** 13
- **Format:** CSV
- **Battery types:** LFP and NMC
- **License:** Apache 2.0 (as listed on the dataset page)

### Download steps

1. Open the [Kaggle dataset page](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge/data).
2. Sign in to Kaggle if prompted.
3. Select **Download**.
4. Extract the downloaded ZIP file if necessary.
5. Put `ev_battery_degradation_v1.csv` in your project folder.

## 3. Battery Chemistry

### LFP — Lithium Iron Phosphate

- Commonly valued for durability and thermal stability.
- Used in a range of electric vehicles and energy-storage applications.
- The dataset description mentions BYD and Wuling as examples.

### NMC — Nickel Manganese Cobalt

- Commonly used where high energy density is important.
- Its degradation depends on operating conditions, cell design, and battery-management practices.
- The dataset description mentions Tesla and Hyundai as examples.

The dataset can be used to compare the simulated patterns for LFP and NMC. These patterns should not automatically be assumed to represent every real vehicle or battery pack.

## 4. Dataset Features

The dataset page describes the following 13 columns:

| Column | Description |
|---|---|
| `Vehicle_ID` | Unique vehicle identifier. |
| `Car_Model` | Electric-vehicle model. |
| `Battery_Type` | Battery chemistry, LFP or NMC. |
| `Battery_Capacity_kWh` | Nominal battery capacity in kilowatt-hours. |
| `Vehicle_Age_Months` | Vehicle age in months. |
| `Total_Charging_Cycles` | Cumulative equivalent full charge-discharge cycles. |
| `Avg_Temperature_C` | Average operating temperature in degrees Celsius. |
| `Fast_Charge_Ratio` | Proportion of charging sessions using DC fast charging, from 0 to 1. |
| `Avg_Discharge_Rate_C` | Average discharge rate, expressed as a C-rate. |
| `Driving_Style` | Driving-style category, such as Conservative, Moderate, or Aggressive. |
| `Internal_Resistance_Ohm` | Battery internal resistance in ohms. |
| `SoH_Percent` | State of Health, expressed as a percentage; regression target. |
| `Battery_Status` | Battery status category, described as Healthy or Replace Required; classification target. |

## 5. Dataset Methodology

According to the dataset description, the records were generated using a semi-empirical degradation simulation that considers:

### Cycle aging
Degradation is modeled in relation to cumulative charge-discharge cycles.

### Temperature impact
The simulation uses the Arrhenius equation to represent faster aging at higher temperatures.

### Charging-rate impact
The simulation models the effects of DC fast charging versus AC slow charging on battery stress and internal resistance.

### Chemistry-specific effects
Different degradation coefficients are used for LFP and NMC battery chemistries.

These are descriptions of the dataset's simulation approach, not proof that every relationship in the synthetic data exactly matches real-world battery behavior.

## 6. Project Objectives

1. Explore the distribution of battery State of Health.
2. Investigate relationships between battery health, age, charging cycles, temperature, and charging behavior.
3. Compare simulated health patterns for LFP and NMC batteries.
4. Train regression models to estimate `SoH_Percent`.
5. Optionally train a classification model to predict `Battery_Status`.
6. Evaluate model performance using appropriate metrics.
7. Discuss limitations and the need for real-world validation.

## 7. Machine-Learning Tasks

### 7.1 Regression

**Goal:** Predict the continuous numerical target `SoH_Percent`.

Possible models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)

Useful evaluation metrics:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination

### 7.2 Classification

**Goal:** Predict the categorical target `Battery_Status`.

Possible models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier

Useful evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC, when appropriate

Before classification, inspect the class labels and their balance. Do not assume the labels are balanced.

## 8. Setup and Required Libraries

Use a recent Python 3 environment. Install the packages with:

```bash
python -m pip install --upgrade pandas numpy matplotlib seaborn scikit-learn
```

The examples below use common scikit-learn APIs. If your installed version differs, consult the [official scikit-learn documentation](https://scikit-learn.org/stable/).

## 9. Load and Inspect the Dataset

```python
import pandas as pd

# Load the CSV file from your project folder
df = pd.read_csv("ev_battery_degradation_v1.csv")

# Preview the first five rows
print(df.head())

# Check rows and columns
print("Dataset shape:", df.shape)

# Check column names, data types, and non-null counts
df.info()

# Summary statistics for numerical columns
print(df.describe())

# Display all column names
print(df.columns.tolist())
```

## 10. Check Data Quality

```python
# Count missing values in each column
print("Missing values:")
print(df.isnull().sum())

# Count fully duplicated rows
print("Duplicate rows:", df.duplicated().sum())

# Inspect battery types and status labels
print("\nBattery types:")
print(df["Battery_Type"].value_counts(dropna=False))

print("\nBattery status:")
print(df["Battery_Status"].value_counts(dropna=False))
```

Do not automatically remove every row with a missing value or every duplicate without first checking what they represent. Choose a cleaning strategy based on the inspection results.

## 11. Exploratory Data Analysis (EDA)

### 11.1 Battery health distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="SoH_Percent", bins=30, kde=True)
plt.title("Distribution of Battery State of Health")
plt.xlabel("State of Health (%)")
plt.ylabel("Number of records")
plt.tight_layout()
plt.show()
```

### 11.2 Charging cycles versus battery health

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Total_Charging_Cycles",
    y="SoH_Percent",
    hue="Battery_Type"
)
plt.title("Charging Cycles vs. Battery State of Health")
plt.xlabel("Total Charging Cycles")
plt.ylabel("State of Health (%)")
plt.tight_layout()
plt.show()
```

### 11.3 Compare health by battery chemistry

```python
print(
    df.groupby("Battery_Type")["SoH_Percent"]
      .agg(["count", "mean", "median", "min", "max"])
)
```

These plots and summaries describe patterns in the dataset. They do not, by themselves, establish that a feature causes battery degradation.

## 12. Understand the Key Imports, Models, and Functions

An **import** makes a class or function from a library available to your code.

| Import or function | Type | Purpose |
|---|---|---|
| `import pandas as pd` | Library import | Loads Pandas for working with tabular data. |
| `pd.read_csv()` | Function | Reads a CSV file into a DataFrame. |
| `train_test_split()` | Function | Splits data into training and testing sets. |
| `LinearRegression()` | Model class | Creates a linear regression model for numerical prediction. |
| `RandomForestRegressor()` | Model class | Creates a regression model based on an ensemble of decision trees. |
| `model.fit(X_train, y_train)` | Method | Trains the model on the training data. |
| `model.predict(X_test)` | Method | Produces predictions for the test inputs. |
| `mean_absolute_error()` | Metric function | Measures the average absolute prediction error. |
| `mean_squared_error()` | Metric function | Measures the average squared prediction error. |
| `r2_score()` | Metric function | Measures how well predictions explain variation in the target. |
| `OneHotEncoder()` | Transformer class | Converts categorical values into numerical indicator features. |
| `ColumnTransformer()` | Transformer class | Applies different preprocessing to different columns. |
| `Pipeline()` | Pipeline class | Combines preprocessing and model steps into one workflow. |

### What do `X` and `y` mean?

- **`X`** contains the input features used by the model.
- **`y`** contains the target value the model is learning to predict.
- **`X_train` and `y_train`** are used for training.
- **`X_test` and `y_test`** are held out for evaluation.
- **`y_pred`** contains the model's predictions for `X_test`.

### What do the evaluation metrics mean?

- **MAE:** Average absolute error, in the target's units. If MAE is 2, predictions are off by an average of 2 SoH percentage points.
- **MSE:** Average squared error. It penalizes large errors more strongly and is expressed in squared target units.
- **RMSE:** Square root of MSE, expressed in the target's units.
- **R²:** A score of 1 indicates perfect predictions; 0 means performance equivalent to predicting the test-set mean; a negative score means worse performance than that baseline.

## 13. Train and Compare Regression Models

This example trains both **Linear Regression** and **Random Forest Regression**. It uses a preprocessing pipeline to handle categorical columns and keeps the test set separate until evaluation.

The example deliberately excludes `Battery_Status` from the regression inputs because it may be derived from battery health. Including a target-derived field can cause data leakage. `Vehicle_ID` is also excluded because it is an identifier, not a meaningful measurement.

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load the dataset
df = pd.read_csv("ev_battery_degradation_v1.csv")

# 2. Set the regression target
target = "SoH_Percent"

# Remove rows where the target is missing
df = df.dropna(subset=[target]).copy()

# Exclude the target, identifier, and possible target-derived status
columns_to_drop = [target, "Vehicle_ID", "Battery_Status"]
X = df.drop(columns=columns_to_drop, errors="ignore")
y = df[target]

# 3. Identify numerical and categorical input columns
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 4. Preprocess numerical and categorical columns separately
numeric_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_preprocessor, numerical_features),
        ("categorical", categorical_preprocessor, categorical_features)
    ]
)

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 6. Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

# 7. Train and evaluate each model
results = []

for name, regressor in models.items():
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict the test-set target
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# 8. Display the comparison
results_df = pd.DataFrame(results).sort_values("MAE")
print(results_df.to_string(index=False))
```

### How to interpret the comparison

- Lower **MAE**, **MSE**, and **RMSE** generally indicate smaller errors on the same test set.
- Higher **R²** generally indicates a better fit to the test targets.
- Do not assume one model will always win; compare the measured results.
- For a more reliable estimate, consider cross-validation and a final independent test set.

## 14. Predict Battery Health for New Data

After choosing and fitting a model, use the same feature columns and preprocessing pipeline to make predictions for new records.

For example, add the following after fitting the `model` you want to use in Section 13:

```python
# Predict the test-set battery health
y_pred = model.predict(X_test)

# Show actual and predicted values for a few records
comparison = pd.DataFrame({
    "Actual_SoH": y_test.to_numpy(),
    "Predicted_SoH": y_pred
})

print(comparison.head(10))
```

For genuinely new records, create a DataFrame containing the same input feature columns used during training. Do not include the target column or `Battery_Status` in the regression inputs.

## 15. Project Workflow

1. **Download:** Obtain the CSV file from Kaggle.
2. **Inspect:** Review the dataset shape, columns, data types, and target values.
3. **Clean:** Check missing values, duplicates, and invalid entries.
4. **Explore:** Plot distributions and investigate feature relationships.
5. **Prepare:** Select features, handle missing values, and encode categorical data.
6. **Train:** Fit baseline and alternative regression models.
7. **Evaluate:** Compare MAE, MSE, RMSE, and R² on held-out data.
8. **Interpret:** Examine errors, model behavior, and important features.
9. **Validate:** Where possible, evaluate on independent real-world battery data.

## 16. Limitations and Good Practices

- **Synthetic data:** The records are simulated and may not capture all real battery behaviors.
- **Generalization:** Strong results on synthetic data do not guarantee strong performance on real vehicles.
- **Data leakage:** Exclude columns that directly or indirectly reveal the target.
- **Identifiers:** Usually exclude identifiers such as `Vehicle_ID` from model features unless there is a justified use.
- **Validation:** Use an appropriate split strategy. If multiple records belong to the same vehicle, consider a group-based split to reduce leakage across records.
- **Reproducibility:** Set a random seed when splitting data and training models.
- **Safety:** Do not use this model alone to make real-world battery safety or replacement decisions.

## 17. Conclusion

The Electric Vehicle (EV) Battery Degradation & Charge dataset offers 10,000 synthetic records for exploring battery health and developing initial machine-learning models. Its features include battery chemistry, vehicle age, charging cycles, temperature, charging behavior, discharge rate, internal resistance, and battery-health targets.

This project demonstrates a typical machine-learning workflow: loading data, inspecting and visualizing it, preparing features, training Linear Regression and Random Forest models, and evaluating predictions with standard regression metrics.

Because the dataset is synthetic, its results should be treated as an educational and experimental starting point. Independent real-world validation is necessary before practical deployment.

## 18. References

- **Dataset and documentation:** [Electric Vehicle (EV) Battery Degradation & Charge — Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge/data)
- **Pandas documentation:** https://pandas.pydata.org/docs/
- **scikit-learn documentation:** https://scikit-learn.org/stable/
- **Matplotlib documentation:** https://matplotlib.org/stable/
- **Seaborn documentation:** https://seaborn.pydata.org/

---

**Dataset file:** `ev_battery_degradation_v1.csv`  
**Primary regression target:** `SoH_Percent`  
**Classification target:** `Battery_Status`  
**Dataset license:** Apache 2.0, as listed on the Kaggle dataset page.
