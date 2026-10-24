import pandas as pd  # Read and check the battery data.
import numpy as np  # Calculate the square root for RMSE.
import matplotlib.pyplot as plt  # Draw and save charts.
import seaborn as sns  # Draw simple data charts.

from sklearn.linear_model import LinearRegression  # Use Linear Regression only.
from sklearn.model_selection import train_test_split  # Split training and test rows.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Check prediction errors.

# Stage 1: This project predicts battery State of Health (SoH, %).

# Stage 2: Load and inspect the supplied Kaggle CSV file.
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")  # Load the supplied CSV file.
print("Problem: predict battery State of Health (SoH, %).")  # State the project goal.
print("Dataset shape (rows, columns):", df.shape)  # Show the real dataset size.
print("First five rows:")  # Label the sample records.
print(df.head())  # Show the first five records.
print("Dataset information:")  # Label the column summary.
df.info()  # Show column names and types.
print("Numeric summary:")  # Label the numeric summary.
print(df.describe())  # Show real number summaries.
data_summary = pd.DataFrame({"Check": ["Rows", "Columns", "Numeric columns", "Text columns", "Missing cells", "Duplicate rows"], "Value": [df.shape[0], df.shape[1], len(df.select_dtypes(include="number").columns), len(df.select_dtypes(include="object").columns), df.isnull().sum().sum(), df.duplicated().sum()]})  # Make a dataset result table.
plt.figure(figsize=(7, 3))  # Make a figure for the dataset summary.
plt.axis("off")  # Hide chart axes.
plt.table(cellText=data_summary.values, colLabels=data_summary.columns, loc="center")  # Draw the dataset summary.
plt.savefig("Reports/dataset_summary.png")  # Save the dataset result image.

# Stage 3: Remove duplicates and blanks, then cap unusual input values.
feature_columns = ["Vehicle_Age_Months", "Total_Charging_Cycles", "Avg_Temperature_C", "Fast_Charge_Ratio", "Avg_Discharge_Rate_C"]  # Choose five model inputs.
rows_before = len(df)  # Count rows before cleaning.
missing_before = df.isnull().sum().sum()  # Count blank cells before cleaning.
duplicates_before = df.duplicated().sum()  # Count duplicates before cleaning.
outliers_before = 0  # Start counting unusual input values.
for col in feature_columns:  # Check each model input.
    q1 = df[col].quantile(0.25)  # Find the lower quarter.
    q3 = df[col].quantile(0.75)  # Find the upper quarter.
    low = q1 - 1.5 * (q3 - q1)  # Set the lower outlier limit.
    high = q3 + 1.5 * (q3 - q1)  # Set the upper outlier limit.
    outliers_before += ((df[col] < low) | (df[col] > high)).sum()  # Count values outside the limits.
df = df.drop_duplicates()  # Remove exact duplicate rows.
df = df.dropna()  # Remove rows with missing cells.
for col in feature_columns:  # Cap unusual values in each input.
    q1 = df[col].quantile(0.25)  # Find the lower quarter.
    q3 = df[col].quantile(0.75)  # Find the upper quarter.
    low = q1 - 1.5 * (q3 - q1)  # Set the lower outlier limit.
    high = q3 + 1.5 * (q3 - q1)  # Set the upper outlier limit.
    df[col] = df[col].clip(low, high)  # Keep values inside the limits.
outliers_after = 0  # Start counting remaining outliers.
for col in feature_columns:  # Check each cleaned input.
    q1 = df[col].quantile(0.25)  # Find the cleaned lower quarter.
    q3 = df[col].quantile(0.75)  # Find the cleaned upper quarter.
    low = q1 - 1.5 * (q3 - q1)  # Find the cleaned lower limit.
    high = q3 + 1.5 * (q3 - q1)  # Find the cleaned upper limit.
    outliers_after += ((df[col] < low) | (df[col] > high)).sum()  # Count remaining unusual values.
cleaning = pd.DataFrame({"Check": ["Rows", "Missing cells", "Duplicate rows", "Input outlier values"], "Before": [rows_before, missing_before, duplicates_before, outliers_before], "After": [len(df), df.isnull().sum().sum(), df.duplicated().sum(), outliers_after]})  # Make a before/after table.
print("Cleaning results (counts):")  # Label the cleaning results.
print(cleaning)  # Show before and after counts.
plt.figure(figsize=(8, 3))  # Make a figure for the cleaning table.
plt.axis("off")  # Hide chart axes.
plt.table(cellText=cleaning.values, colLabels=cleaning.columns, loc="center")  # Draw the cleaning table.
plt.savefig("Reports/data_cleaning.png")  # Save the cleaning result image.

# Stage 4: Explore the cleaned measurements with six charts.
plt.figure()  # Start the SoH histogram.
sns.histplot(df["SoH_Percent"])  # Show the distribution of SoH (%).
plt.xlabel("State of Health (%)")  # Label the health unit.
plt.savefig("Reports/soh_histogram.png")  # Save the histogram.
plt.figure()  # Start the SoH box plot.
plt.boxplot(df["SoH_Percent"], orientation="vertical")  # Show unusual SoH values (%).
plt.ylabel("State of Health (%)")  # Label the health unit.
plt.savefig("Reports/soh_boxplot.png")  # Save the box plot.
plt.figure()  # Start the cycles scatter plot.
plt.scatter(df["Total_Charging_Cycles"], df["SoH_Percent"])  # Compare cycles and SoH (%).
plt.xlabel("Total charging cycles (cycles)")  # Label cycle counts.
plt.ylabel("State of Health (%)")  # Label health values.
plt.savefig("Reports/cycles_vs_soh.png")  # Save the scatter plot.
plt.figure(figsize=(8, 6))  # Start the correlation heatmap.
sns.heatmap(df[feature_columns + ["SoH_Percent"]].corr(), annot=True)  # Show unitless correlations.
plt.savefig("Reports/correlation_heatmap.png")  # Save the heatmap.
plt.figure()  # Start the battery-status pie chart.
plt.pie(df["Battery_Status"].value_counts(), labels=df["Battery_Status"].value_counts().index, autopct="%1.1f%%")  # Show category shares (%).
plt.savefig("Reports/battery_status_pie.png")  # Save the pie chart.
plt.figure()  # Start the battery-type bar chart.
plt.bar(df["Battery_Type"].value_counts().index, df["Battery_Type"].value_counts().values)  # Count records (rows).
plt.xlabel("Battery type (category)")  # Label battery types.
plt.ylabel("Number of records (rows)")  # Label record counts.
plt.savefig("Reports/battery_type_bar.png")  # Save the bar chart.

# Stage 5: Choose safe features and split data into training and test rows.
X = df[feature_columns]  # Use age, cycles, temperature, charge ratio, and discharge rate.
y = df["SoH_Percent"]  # Set State of Health (%) as the target.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # Use 80% train and 20% test.
print("Training rows:", len(X_train), "| Test rows:", len(X_test))  # Show the split sizes.
split_summary = pd.DataFrame({"Part": ["Training rows", "Test rows", "Input columns", "Target"], "Value": [len(X_train), len(X_test), 5, "SoH_Percent (%)"]})  # Make a split result table.
plt.figure(figsize=(7, 3))  # Make a figure for the split results.
plt.axis("off")  # Hide chart axes.
plt.table(cellText=split_summary.values, colLabels=split_summary.columns, loc="center")  # Draw the split results.
plt.savefig("Reports/feature_split.png")  # Save the split result image.

# Stage 6: Train simple and multiple Linear Regression models.
simple_X_train = X_train[["Total_Charging_Cycles"]]  # Use cycles (cycles) for the simple model.
simple_X_test = X_test[["Total_Charging_Cycles"]]  # Keep test cycles for the simple model.
model_simple = LinearRegression()  # Make the one-feature model.
model_simple.fit(simple_X_train, y_train)  # Learn the simple regression line.
model = LinearRegression()  # Make the five-feature model.
model.fit(X_train, y_train)  # Learn multiple linear regression.
print("Simple line: SoH (%) =", model_simple.coef_[0], "percent per cycle * cycles +", model_simple.intercept_, "%")  # Show y = mx + b.
simple_line = df.sort_values("Total_Charging_Cycles")  # Put cycle counts in order for the line chart.
plt.figure()  # Start the simple regression chart.
plt.scatter(df["Total_Charging_Cycles"], df["SoH_Percent"], s=8, alpha=0.3)  # Show real records (cycles, %).
plt.plot(simple_line["Total_Charging_Cycles"], model_simple.predict(simple_line[["Total_Charging_Cycles"]]), color="red", linewidth=3)  # Draw the fitted line.
plt.xlabel("Total charging cycles (cycles)")  # Label cycle counts.
plt.ylabel("State of Health (%)")  # Label health values.
plt.savefig("Reports/simple_regression_line.png")  # Save the regression result.

# Stage 7: Evaluate both Linear Regression models.
simple_prediction = model_simple.predict(simple_X_test)  # Predict test SoH with cycles only.
prediction = model.predict(X_test)  # Predict test SoH with five inputs.
results = pd.DataFrame()  # Start the model-results table.
results["Model"] = ["Simple Linear", "Multiple Linear"]  # Name the two models.
results["MAE (%)"] = [mean_absolute_error(y_test, simple_prediction), mean_absolute_error(y_test, prediction)]  # Add average errors.
results["MSE (% squared)"] = [mean_squared_error(y_test, simple_prediction), mean_squared_error(y_test, prediction)]  # Add squared errors.
results["RMSE (%)"] = [np.sqrt(mean_squared_error(y_test, simple_prediction)), np.sqrt(mean_squared_error(y_test, prediction))]  # Add root errors.
results["R2 (unitless)"] = [r2_score(y_test, simple_prediction), r2_score(y_test, prediction)]  # Add unitless scores.
print("Test results:")  # Label the evaluation results.
print(results.round(4))  # Show results to four decimal places.
plt.figure(figsize=(10, 2))  # Make a figure for the score table.
plt.axis("off")  # Hide chart axes.
plt.table(cellText=results.round(4).values, colLabels=results.columns, loc="center")  # Draw the test score table.
plt.savefig("Reports/model_results.png")  # Save the model comparison image.

# Stage 8: Predict one held-out sample and save its result image.
sample = X_test.iloc[[0]]  # Choose one test record the model did not train on.
sample_actual = y_test.iloc[0]  # Get its actual SoH (%).
sample_prediction = model.predict(sample)[0]  # Predict its SoH (%).
print("Held-out sample actual SoH (%):", sample_actual)  # Show the real test answer.
print("Held-out sample predicted SoH (%):", sample_prediction)  # Show the model answer.
plt.figure()  # Start a sample-result chart.
plt.bar(["Actual SoH (%)", "Predicted SoH (%)"], [sample_actual, sample_prediction])  # Compare actual and predicted health.
plt.ylabel("State of Health (%)")  # Label health values.
plt.savefig("Reports/sample_prediction.png")  # Save the sample result.

# Stage 9: Ask for new measurements and print a deployment prediction.
age_input = float(input("Vehicle age in months: "))  # Ask for vehicle age.
cycles_input = float(input("Total charging cycles (cycles): "))  # Ask for charging cycles.
temperature_input = float(input("Average temperature (°C): "))  # Ask for temperature.
charge_input = float(input("Fast-charge ratio (0 to 1): "))  # Ask for fast-charge share.
discharge_input = float(input("Average discharge rate (C-rate): "))  # Ask for discharge rate.
new_sample = pd.DataFrame([[age_input, cycles_input, temperature_input, charge_input, discharge_input]], columns=feature_columns)  # Arrange the entered inputs.
new_soh = model.predict(new_sample)[0]  # Predict the new State of Health (%).
print("Predicted new battery SoH (%):", new_soh)  # Show the predicted health.
plt.figure()  # Start a new-prediction chart.
plt.bar(["Predicted SoH (%)"], [new_soh])  # Show the new health estimate.
plt.ylabel("State of Health (%)")  # Label health values.
plt.savefig("Reports/deployment_prediction.png")  # Save the deployment result.

# Stage 10: Later, monitor measured SoH, errors, and unusual new inputs.
