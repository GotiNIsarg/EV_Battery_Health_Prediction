# Learning Guide: EV Battery Health Prediction

This guide explains the four course units in simple English and shows where each idea appears in the project. The examples and results below come from the supplied CSV, which Kaggle describes as synthetic simulation data. They are for learning, not real battery diagnosis.

## The teacher's project workflow

The course project instructions move from choosing a problem and dataset to inspecting and cleaning data, making EDA charts, creating an optional automated HTML report, selecting features, training a suitable model, evaluating and improving it, then presenting the code, results, conclusion, and future scope. The submission folder is `Dataset / Python_Code / Reports`.

The report follows this order: cover page, certificate, problem statement, objectives, dataset description, preprocessing, EDA, model building, evaluation, results, conclusion, and future scope. The journal also asks for at least 300 records and 5 attributes/features. This CSV has 10,000 records and 13 columns, so it meets those size counts; however, Kaggle identifies it as simulated rather than measured real-world data.

## Unit 1 — Fundamentals of AI and Machine Learning

### Learn the idea

- **Artificial Intelligence (AI)** is the broad idea of making computers do tasks that seem intelligent.
- **Machine Learning (ML)** is one way to build AI: the computer learns patterns from example data.
- **Deep Learning** is one kind of ML that uses large neural networks; this project does not need it.
- The ML life cycle is: define the question, collect data, inspect and clean it, explore it, select features, train a model, evaluate it, and then maintain or improve it.
- **Exploratory Data Analysis (EDA)** means looking at a dataset before modeling. Univariate EDA studies one column, bivariate EDA compares two columns, and multivariate EDA looks at several columns together.
- Numeric data can be **discrete** (a count such as charging cycles) or **continuous** (a measurement such as temperature). Text categories include **nominal** labels (battery type) and **ordinal** labels (a ranked category).
- Missing values, duplicate rows, and unusual values should be checked before modeling. Skewness describes whether a distribution has a longer tail on one side; kurtosis describes how heavy its tails are compared with a normal distribution.

### Where the project uses it

The program reads the CSV, prints its shape and column types, separates numeric columns from text columns, and prints summary statistics, skewness, and kurtosis. It checks missing cells and exact duplicate rows, then caps IQR outliers in the five selected input measurements. It saves a histogram, box plot, scatter plot, heatmap, pie chart, and bar chart.

### Small project example

`SoH_Percent` is numeric and continuous because it can have decimal values such as 94.60%. `Battery_Type` is a category such as NMC or LFP. The histogram studies the SoH column by itself; the scatter plot studies charging cycles (cycles) against SoH (%).

### Actual data checks

- Shape: **10,000 rows × 13 columns**.
- Numeric columns: **8**. Text columns: **5**.
- Missing cells before cleaning: **0**.
- Exact duplicate rows before cleaning: **0**.
- IQR outlier values in the five planned input measurements: **101 before capping, 0 after capping**.
- The program preserves all **10,000 rows** and does not cap the SoH target.

## Unit 2 — Automated EDA and Simple Linear Regression

### Learn the idea

- **Pandas** reads and organizes table-shaped data. **NumPy** supports numeric calculations. Matplotlib and Seaborn make charts.
- An automated EDA tool can quickly create an HTML summary. It is an extra aid: a student still needs to check units, strange values, and possible leakage.
- A **feature** is an input used to make a prediction. The **target** is the answer the model should predict.
- **Correlation** measures the direction and strength of a linear relationship; its value is between -1 and 1. Correlation is not proof that one value causes another.
- Simple Linear Regression uses one input and fits a straight line:

  ```text
  y = m x + b
  ```

  Here, `m` is the slope, `b` is the intercept, and `y` is the predicted output. A residual is actual value minus predicted value. RSS adds up the squared residuals; lower RSS means the fitted line has smaller training errors.
- R² describes how much variation the fitted model explains on the data being scored. A high training R² does not by itself prove that the model will work on new data.

### Where the project uses it

The simple regression uses `Total_Charging_Cycles` (cycles) as its one input and `SoH_Percent` (%) as its output. A separate optional ydata-profiling snippet can make `Reports/EDA_Report.html`; it is not imported by the main script.

### Small project example and measured result

The actual fitted line on all cleaned rows was:

```text
SOH (%) = -0.009001 (% per cycle) * cycles + 99.920179 (%)
Training R² (unitless): 0.762814
Training RSS (%²): 25094.825246
```

The negative slope means that the simulated records generally show lower SoH as charging cycles rise. This line and R² use the same rows for fitting and scoring, so they are an in-sample teaching example rather than final test performance.

## Unit 3 — Understanding Machine Learning

### Learn the idea

- **Supervised learning** uses example inputs and known answers. **Unsupervised learning** looks for patterns without given answers.
- **Regression** predicts a number, such as SoH (%). **Classification** predicts a category, such as Healthy or Replace Required.
- A dataset has rows (records) and columns (variables). Features are written as `X`; the target is written as `y`.
- A **training set** teaches the model. A **test set** is held back to check how well the trained model predicts unseen rows. Cross-validation checks performance over several different training/validation folds.
- **Overfitting** means a model learns details of its training examples too closely and may perform poorly on new examples. **Underfitting** means it is too simple to learn the useful pattern.
- MAE averages the absolute prediction errors. MSE averages squared errors, so large errors count more. Both are for numerical regression targets.

### Where the project uses it

The task is supervised regression because each row has battery measurements (`X`) and a known SoH value (`y`). The project uses 80% of the cleaned rows for training and 20% for a final test. The training portion is also checked using five-fold cross-validation.

### Small project example

For one row, the model receives age (months), cycles (cycles), temperature (°C), fast-charge ratio (unitless, 0–1), and discharge rate (C-rate). The label it tries to predict is SoH (%). `Battery_Status` is another category in the dataset, but it is not used as an input or predicted target in this project.

## Unit 4 — Supervised Learning Algorithms

### Learn the idea

- Linear Regression uses Ordinary Least Squares (OLS) to fit coefficients that minimize squared residuals.
- R², MAE, MSE, and RMSE help compare regression models. MAE and RMSE use the target's unit (%); MSE uses the squared target unit (%²); R² is unitless.
- Logistic Regression, sigmoid, softmax, and One-vs-Rest are for classification categories. They are not used to predict the numeric SoH target here.
- **Ridge (L2)** adds a penalty related to squared coefficients and usually shrinks them. **Lasso (L1)** penalizes absolute coefficient sizes and may shrink some coefficients to zero.
- **K-fold cross-validation** repeatedly trains on all but one fold and scores on the remaining fold. With 5 folds, every training row is held out once for scoring.

### Where the project uses it

The project compares Linear Regression, Ridge, and Lasso. Each model is placed after a StandardScaler in a scikit-learn pipeline, so scaling is fitted using only the training part of each fold. The model with the highest mean five-fold R² on the training set is selected, then measured on the held-back test set.

### Actual five-fold scores

The fold scores below are R² values and have no unit:

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.8123 | 0.8089 | 0.7803 | 0.7937 | 0.8067 | 0.8004 |
| Ridge (L2) | 0.8124 | 0.8089 | 0.7802 | 0.7937 | 0.8067 | 0.8004 |
| Lasso (L1) | 0.8117 | 0.8020 | 0.7761 | 0.7938 | 0.8046 | 0.7976 |

## The nine project steps

The complete, runnable program is in `Python_Code/project.py`. The short excerpts below show the main idea of each step; they are not replacements for the full script.

### Step 1 — Load data

```python
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
print(df.head())
```

**Explanation:** `read_csv` reads the supplied CSV into a Pandas table named `df`. `head()` shows the first five records so we can check that the file loaded.

**Actual result:** The first record is a Tesla Model 3 with NMC chemistry, a listed capacity of 75.0 kWh, age 41 months, 390 cycles, average temperature 21.5 °C, and SoH 94.60%. These are values from the supplied CSV, not a new prediction.

### Step 2 — Understand data

```python
print(df.shape)
df.info()
print(df.describe(include="all"))
print(df.select_dtypes(include="number").skew())
print(df.select_dtypes(include="number").kurt())
```

**Explanation:** `shape` gives the number of rows and columns. `info()` shows data types and non-missing counts. `describe()` summarizes values; `skew()` and `kurt()` summarize numeric distribution shape.

**Actual result:** The shape is **(10,000, 13)**. The table has 8 numeric and 5 text columns, and the original file has no missing values. The full script prints the real statistics for each numeric column.

### Step 3 — Clean data

```python
df = df.drop_duplicates().copy()
column = "Total_Charging_Cycles"
q1 = df[column].quantile(0.25)
q3 = df[column].quantile(0.75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
df[column] = df[column].clip(lower=lower_limit, upper=upper_limit)
```

**Explanation:** Exact duplicate rows are removed. The full program fills any numeric gaps with a median and text gaps with a mode. It calculates IQR limits for the selected input measurements and caps values beyond those limits. The SoH target is not capped because changing the answers would change what the model learns.

**Actual result:** There were 0 duplicate rows and 0 missing cells, so neither removal nor imputation changed the dataset. IQR flagged 101 input-feature values; capping reduced this count to 0 and retained 10,000 rows.

### Step 4 — Make EDA charts

```python
sns.histplot(data=df, x="SoH_Percent", bins=30, kde=True)
plt.xlabel("State of Health (%)")
plt.savefig("Reports/soh_histogram.png")
```

**Explanation:** The histogram shows how SoH values are distributed; `kde=True` adds a smooth curve. The complete program also saves a box plot, scatter plot, correlation heatmap, battery-status pie chart, and battery-type bar chart.

**Actual result:** Seven chart images are generated in `Reports/`, including the separate regression-line chart from Step 5.

### Step 5 — Fit a simple linear regression

```python
simple_X = df[["Total_Charging_Cycles"]]
simple_y = df["SoH_Percent"]
simple_model = LinearRegression()
simple_model.fit(simple_X, simple_y)
```

**Explanation:** This model fits a straight line between one input (cycles) and the SoH target. The complete script prints the equation `y = mx + b`, R², and RSS, then saves the regression-line chart.

**Actual result:** The slope is **-0.009001 percentage points per cycle**, the intercept is **99.920179%**, the in-sample R² is **0.762814 (unitless)**, and RSS is **25094.825246 %²**.

### Step 6 — Choose features, target, and split

```python
X = df[feature_columns]
y = df["SoH_Percent"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

**Explanation:** `X` holds the five chosen measurements and `y` holds the answer to predict. A fixed random state makes the 80/20 split repeatable.

**Actual result:** There are **8,000 training rows** and **2,000 test rows**. `Vehicle_ID`, capacity, internal resistance, text categories, and `Battery_Status` are not model inputs.

### Step 7 — Train three models with five-fold cross-validation

```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
linear_cv_scores = cross_val_score(
    linear_model, X_train, y_train, cv=kfold, scoring="r2"
)
ridge_cv_scores = cross_val_score(
    ridge_model, X_train, y_train, cv=kfold, scoring="r2"
)
lasso_cv_scores = cross_val_score(
    lasso_model, X_train, y_train, cv=kfold, scoring="r2"
)
```

**Explanation:** Each model is trained and checked on five different parts of the training rows. The mean fold score is used to choose a model; the test rows stay separate.

**Actual result:** Mean CV R² is **0.8004** for Linear, **0.8004** for Ridge, and **0.7976** for Lasso, all unitless. Linear and Ridge tie at four decimal places.

### Step 8 — Evaluate and compare

```python
y_pred = linear_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
```

**Explanation:** These metrics compare predictions with actual test-set SoH values. Lower MAE, MSE, and RMSE are better; a higher R² is generally better. MAE and RMSE are in percentage points, MSE is in squared percentage points, and R² has no unit.

**Actual test results:**

| Model | MAE (%) | MSE (%²) | RMSE (%) | R² (unitless) |
|---|---:|---:|---:|---:|
| Linear | 1.0687 | 2.2163 | 1.4887 | 0.7916 |
| Ridge (L2) | 1.0687 | 2.2163 | 1.4887 | 0.7916 |
| Lasso (L1) | 1.0623 | 2.2665 | 1.5055 | 0.7868 |

### Step 9 — Save the selected model and predict one sample

```python
with open("Python_Code/best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

sample_input = X_test.iloc[[0]]
sample_prediction = best_model.predict(sample_input)[0]
```

**Explanation:** `pickle` saves the fitted model so it can be reused. The sample is from the held-out test rows, and the full program prints both the actual and predicted SoH.

**Actual result:** For age **68 months**, **432 cycles**, temperature **24.3 °C**, fast-charge ratio **0.91**, and discharge rate **0.61 C-rate**, actual SoH was **97.31%** and predicted SoH was **95.41%**. The saved model is `Python_Code/best_model.pkl`.

## Important scientific limitation

The supplied Kaggle dataset is synthetic. A high score here only means the model learned patterns in this simulation. It does not show that the model can estimate the health of a real battery. A future project should test on measured data from different batteries or vehicles and avoid features that directly reveal the target.
