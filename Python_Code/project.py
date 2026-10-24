import pickle  # Save the model for later use.
import matplotlib.pyplot as plt  # Draw the charts.
import numpy as np  # Calculate square roots.
import pandas as pd  # Read and examine the CSV.
import seaborn as sns  # Draw simple charts.
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Make regression models.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Check predictions.
from sklearn.model_selection import cross_val_score, train_test_split  # Test models fairly.

# Step 1: Load data.
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")  # Read the supplied CSV file.
print(df.head())  # Show the first five battery records.

# Step 2: Understand data.
print("Rows and columns:", df.shape)  # Show the number of records and columns.
df.info()  # Show column names and data types.
print(df.describe())  # Show summaries of numeric columns.
print("Numeric columns:", df.select_dtypes(include="number").columns)  # Show number columns.
print("Text columns:", df.select_dtypes(include="object").columns)  # Show text columns.
print("Missing values:", df.isnull().sum().sum())  # Count missing cells.
print("Duplicate rows:", df.duplicated().sum())  # Count repeated records.

# Step 3: Clean data.
df = df.drop_duplicates()  # Remove exact duplicate records.
df = df.dropna()  # Remove rows that contain missing values.
feature_columns = ["Vehicle_Age_Months", "Total_Charging_Cycles", "Avg_Temperature_C", "Fast_Charge_Ratio", "Avg_Discharge_Rate_C"]  # Choose five inputs.
outliers = 0  # Start the outlier count at zero.
for col in feature_columns:  # Check each input measurement.
    q1 = df[col].quantile(0.25)  # Find the lower quarter.
    q3 = df[col].quantile(0.75)  # Find the upper quarter.
    low = q1 - 1.5 * (q3 - q1)  # Find the lower outlier limit.
    high = q3 + 1.5 * (q3 - q1)  # Find the upper outlier limit.
    outliers += ((df[col] < low) | (df[col] > high)).sum()  # Count unusual input values.
    df[col] = df[col].clip(low, high)  # Keep input values within the limits.
print("Outlier values capped:", outliers)  # Report the cleaned input outliers.
print("Rows kept:", len(df))  # Report how many records remain.

# Step 4: Make six simple EDA charts.
plt.figure(figsize=(12, 8))  # Start one figure for six charts.
sns.histplot(df["SoH_Percent"], ax=plt.subplot(2, 3, 1))  # Draw the SoH histogram (%).
plt.title("SoH histogram (%)")  # Label the first chart.
plt.subplot(2, 3, 2).boxplot(df["SoH_Percent"], orientation="vertical")  # Draw the SoH box plot (%).
plt.title("SoH box plot (%)")  # Label the second chart.
sns.scatterplot(x=df["Total_Charging_Cycles"], y=df["SoH_Percent"], ax=plt.subplot(2, 3, 3))  # Plot cycles and SoH.
plt.title("Cycles and SoH (cycles, %)")  # Label the third chart.
sns.heatmap(df[feature_columns + ["SoH_Percent"]].corr(), ax=plt.subplot(2, 3, 4))  # Draw unitless correlations.
plt.title("Correlation (unitless)")  # Label the fourth chart.
plt.subplot(2, 3, 5).pie(df["Battery_Status"].value_counts(), labels=df["Battery_Status"].value_counts().index, autopct="%1.1f%%")  # Show status shares (%).
plt.title("Battery status share (%)")  # Label the fifth chart.
plt.subplot(2, 3, 6).bar(df["Battery_Type"].value_counts().index, df["Battery_Type"].value_counts().values)  # Count records (rows).
plt.title("Records by battery type (rows)")  # Label the sixth chart.
plt.tight_layout()  # Fit all charts inside the figure.
plt.savefig("Reports/eda_charts.png")  # Save the six charts together.

# Step 5: Fit a simple line using charging cycles.
model = LinearRegression()  # Make a simple linear regression model.
model.fit(df[["Total_Charging_Cycles"]], df["SoH_Percent"])  # Learn the line from the data.
print("Line: SOH (%) =", model.coef_[0], "percent per cycle x cycles +", model.intercept_, "%")  # Show y = mx + b.
print("Simple model R2 (unitless):", r2_score(df["SoH_Percent"], model.predict(df[["Total_Charging_Cycles"]])))  # Show the line score.

# Step 6: Split data into training and test records.
X = df[feature_columns]  # Keep the five chosen input measurements.
y = df["SoH_Percent"]  # Keep battery health (%) as the answer.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # Keep 80% for training.
print("Training rows:", len(X_train), "| Test rows:", len(X_test))  # Show the split sizes.

# Step 7: Train each model and compare five-fold CV R2.
model1 = LinearRegression()  # Make the Linear model.
model1.fit(X_train, y_train)  # Train Linear using the training rows.
score1 = cross_val_score(model1, X_train, y_train, cv=5).mean()  # Average five Linear R2 scores.
model2 = Ridge(alpha=1.0)  # Make the Ridge (L2) model.
model2.fit(X_train, y_train)  # Train Ridge using the training rows.
score2 = cross_val_score(model2, X_train, y_train, cv=5).mean()  # Average five Ridge R2 scores.
model3 = Lasso(alpha=0.1, max_iter=10000)  # Make the Lasso (L1) model.
model3.fit(X_train, y_train)  # Train Lasso using the training rows.
score3 = cross_val_score(model3, X_train, y_train, cv=5).mean()  # Average five Lasso R2 scores.
print("Linear CV R2:", score1)  # Show Linear's unitless score.
print("Ridge CV R2:", score2)  # Show Ridge's unitless score.
print("Lasso CV R2:", score3)  # Show Lasso's unitless score.

# Step 8: Compare test predictions and errors.
pred1 = model1.predict(X_test)  # Predict test rows with Linear.
pred2 = model2.predict(X_test)  # Predict test rows with Ridge.
pred3 = model3.predict(X_test)  # Predict test rows with Lasso.
results = pd.DataFrame()  # Start a table for the three models.
results["Model"] = ["Linear", "Ridge", "Lasso"]  # Name each model.
results["MAE (%)"] = [mean_absolute_error(y_test, pred1), mean_absolute_error(y_test, pred2), mean_absolute_error(y_test, pred3)]  # Add average errors.
results["MSE (% squared)"] = [mean_squared_error(y_test, pred1), mean_squared_error(y_test, pred2), mean_squared_error(y_test, pred3)]  # Add squared errors.
results["RMSE (%)"] = [np.sqrt(mean_squared_error(y_test, pred1)), np.sqrt(mean_squared_error(y_test, pred2)), np.sqrt(mean_squared_error(y_test, pred3))]  # Add square-root errors.
results["R2 (unitless)"] = [r2_score(y_test, pred1), r2_score(y_test, pred2), r2_score(y_test, pred3)]  # Add unitless scores.
print(results.round(4))  # Show errors in % or % squared.

# Step 9: Save the best CV model and predict one test sample.
model = model1  # Start with Linear as the current best model.
best_name = "Linear"  # Start with Linear's name.
if round(score2, 4) > round(score1, 4) and round(score2, 4) >= round(score3, 4):  # Compare rounded CV scores.
    model = model2  # Select Ridge.
    best_name = "Ridge"  # Save Ridge's name.
if round(score3, 4) > round(score1, 4) and round(score3, 4) > round(score2, 4):  # Compare rounded CV scores.
    model = model3  # Select Lasso.
    best_name = "Lasso"  # Save Lasso's name.
print("Selected model:", best_name)  # Show the selected model.
with open("Python_Code/best_model.pkl", "wb") as model_file:  # Open the model file safely.
    pickle.dump(model, model_file)  # Save the selected model.
sample = X_test.iloc[[0]]  # Pick one held-out battery sample.
print("Sample inputs (months, cycles, °C, ratio, C-rate):")  # Explain the sample units.
print(sample)  # Show the five sample measurements.
print("Actual SoH (%):", y_test.iloc[0], "| Predicted SoH (%):", model.predict(sample)[0])  # Compare actual and predicted health.
