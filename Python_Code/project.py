import pandas as pd  # Read and check the battery data.
import numpy as np  # Calculate the square root for RMSE.
import matplotlib.pyplot as plt  # Draw and save charts.
import seaborn as sns  # Draw simple data charts.

from sklearn.linear_model import LinearRegression  # Use Linear Regression only.
from sklearn.model_selection import train_test_split  # Split training and test rows.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Check errors.

# Stage 1: Predict battery State of Health (SoH, %).

# Stage 2: Load the supplied CSV and inspect the data.
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")  # Load the dataset.
print("Problem: predict battery State of Health (SoH, %).")  # Show the project goal.
print("Dataset shape (rows, columns):", df.shape)  # Show the dataset size.
print("First five rows:")  # Label the sample.
print(df.head())  # Show five battery records.
print("Dataset information:")  # Label the column details.
df.info()  # Show column names and data types.
print("Numeric summary:")  # Label the numeric summary.
print(df.describe())  # Show numeric measurements.

# Prepare a small table to show basic dataset facts.
data_summary = pd.DataFrame(  # Make a dataset summary table.
    {  # Start the table columns.
        "Check": [  # Name each check.
            "Rows",  # Count records.
            "Columns",  # Count fields.
            "Numeric columns",  # Count number fields.
            "Text columns",  # Count text fields.
            "Missing cells",  # Count empty values.
            "Duplicate rows",  # Count repeated records.
        ],  # Finish the check names.
        "Value": [  # Add the measured values.
            df.shape[0],  # Show the row count.
            df.shape[1],  # Show the column count.
            len(df.select_dtypes(include="number").columns),  # Count number fields.
            len(df.select_dtypes(include="object").columns),  # Count text fields.
            df.isnull().sum().sum(),  # Count empty cells.
            df.duplicated().sum(),  # Count duplicate rows.
        ],  # Finish the measured values.
    }  # Finish the table columns.
)  # Finish the dataset summary table.
plt.figure(figsize=(8, 3))  # Make a wide summary image.
plt.title("Dataset summary")  # Name the summary image.
plt.axis("off")  # Hide chart axes.
data_table = plt.table(  # Draw the summary table.
    cellText=data_summary.values,  # Show the measured values.
    colLabels=data_summary.columns,  # Show the column headings.
    cellLoc="left",  # Align table text for reading.
    loc="center",  # Center the table.
)  # Finish drawing the table.
data_table.auto_set_font_size(False)  # Keep the text size readable.
data_table.set_fontsize(10)  # Set table text size.
data_table.scale(1, 1.5)  # Add space between table rows.
plt.tight_layout()  # Fit the title and table.
plt.savefig("Reports/dataset_summary.png", dpi=160, bbox_inches="tight")  # Save the summary.
plt.close()  # Close this chart.

# Stage 3: Remove duplicates and blank rows, then cap input outliers.
feature_columns = [  # List five model input columns.
    "Vehicle_Age_Months",  # Age unit: months.
    "Total_Charging_Cycles",  # Cycle unit: cycles.
    "Avg_Temperature_C",  # Temperature unit: degrees Celsius (°C).
    "Fast_Charge_Ratio",  # Ratio unit: unitless, from 0 to 1.
    "Avg_Discharge_Rate_C",  # Discharge unit: C-rate.
]  # Finish the input column list.
rows_before = len(df)  # Count rows before cleaning.
missing_before = df.isnull().sum().sum()  # Count missing cells before cleaning.
duplicates_before = df.duplicated().sum()  # Count duplicate rows before cleaning.
outliers_before = 0  # Start the before-cleaning outlier count.
for col in feature_columns:  # Check every model input.
    q1 = df[col].quantile(0.25)  # Find the lower quarter.
    q3 = df[col].quantile(0.75)  # Find the upper quarter.
    low = q1 - 1.5 * (q3 - q1)  # Find the lower IQR limit.
    high = q3 + 1.5 * (q3 - q1)  # Find the upper IQR limit.
    outliers_before += int(((df[col] < low) | (df[col] > high)).sum())  # Count outliers.
df = df.drop_duplicates()  # Remove exact duplicate rows.
df = df.dropna()  # Remove rows with missing cells.
for col in feature_columns:  # Cap values in every model input.
    q1 = df[col].quantile(0.25)  # Find the lower quarter.
    q3 = df[col].quantile(0.75)  # Find the upper quarter.
    low = q1 - 1.5 * (q3 - q1)  # Find the lower IQR limit.
    high = q3 + 1.5 * (q3 - q1)  # Find the upper IQR limit.
    df[col] = df[col].clip(low, high)  # Cap values to the IQR limits.
outliers_after = 0  # Start the after-cleaning outlier count.
for col in feature_columns:  # Check every cleaned input.
    q1 = df[col].quantile(0.25)  # Find the cleaned lower quarter.
    q3 = df[col].quantile(0.75)  # Find the cleaned upper quarter.
    low = q1 - 1.5 * (q3 - q1)  # Find the cleaned lower IQR limit.
    high = q3 + 1.5 * (q3 - q1)  # Find the cleaned upper IQR limit.
    outliers_after += int(((df[col] < low) | (df[col] > high)).sum())  # Count outliers.
cleaning = pd.DataFrame(  # Make the before-and-after table.
    {  # Start the table columns.
        "Check": [  # Name each cleaning check.
            "Rows",  # Count records.
            "Missing cells",  # Count empty values.
            "Duplicate rows",  # Count repeated records.
            "Input outlier values",  # Count unusual input values.
        ],  # Finish the check names.
        "Before": [  # Add the values before cleaning.
            rows_before,  # Show original records.
            missing_before,  # Show original missing cells.
            duplicates_before,  # Show original duplicates.
            outliers_before,  # Show original input outliers.
        ],  # Finish the original values.
        "After": [  # Add the values after cleaning.
            len(df),  # Show remaining records.
            df.isnull().sum().sum(),  # Show remaining missing cells.
            df.duplicated().sum(),  # Show remaining duplicates.
            outliers_after,  # Show remaining input outliers.
        ],  # Finish the cleaned values.
    }  # Finish the table columns.
)  # Finish the cleaning table.
print("Cleaning results (counts):")  # Label the cleaning table.
print(cleaning)  # Show the real before-and-after values.
plt.figure(figsize=(9, 3))  # Make a wide cleaning image.
plt.title("Data cleaning: before and after")  # Name the cleaning image.
plt.axis("off")  # Hide chart axes.
cleaning_table = plt.table(  # Draw the cleaning table.
    cellText=cleaning.values,  # Show the measured values.
    colLabels=cleaning.columns,  # Show the column headings.
    cellLoc="center",  # Center the counts.
    loc="center",  # Center the table.
)  # Finish drawing the table.
cleaning_table.auto_set_font_size(False)  # Keep table text readable.
cleaning_table.set_fontsize(10)  # Set table text size.
cleaning_table.scale(1, 1.6)  # Add space between table rows.
plt.tight_layout()  # Fit title and table.
plt.savefig("Reports/data_cleaning.png", dpi=160, bbox_inches="tight")  # Save cleaning results.
plt.close()  # Close this chart.

# Stage 4: Explore the cleaned data with six charts.
plt.figure(figsize=(8, 5))  # Make a clear histogram.
sns.histplot(data=df, x="SoH_Percent", bins=30, color="#247ba0")  # Show SoH distribution (%).
plt.title("Battery State of Health distribution")  # Name the histogram.
plt.xlabel("State of Health (%)")  # Label the target unit.
plt.ylabel("Number of records (rows)")  # Label the record count.
plt.tight_layout()  # Fit the labels.
plt.savefig("Reports/soh_histogram.png", dpi=160, bbox_inches="tight")  # Save the histogram.
plt.close()  # Close this chart.

plt.figure(figsize=(8, 4))  # Make a clear box plot.
plt.boxplot(df["SoH_Percent"], orientation="horizontal", patch_artist=True)  # Show SoH spread (%).
plt.title("Battery State of Health box plot")  # Name the box plot.
plt.xlabel("State of Health (%)")  # Label the target unit.
plt.tight_layout()  # Fit the labels.
plt.savefig("Reports/soh_boxplot.png", dpi=160, bbox_inches="tight")  # Save the box plot.
plt.close()  # Close this chart.

plt.figure(figsize=(8, 5))  # Make a clear scatter plot.
plt.scatter(  # Draw each battery record.
    df["Total_Charging_Cycles"],  # Use charging cycles on the x-axis.
    df["SoH_Percent"],  # Use SoH on the y-axis.
    s=12,  # Use small points for many records.
    alpha=0.45,  # Make overlapping points easier to see.
    color="#247ba0",  # Use the chart's blue color.
)  # Finish the scatter plot.
plt.title("Charging cycles and battery health")  # Name the scatter plot.
plt.xlabel("Total charging cycles (cycles)")  # Label the cycle unit.
plt.ylabel("State of Health (%)")  # Label the target unit.
plt.tight_layout()  # Fit the labels.
plt.savefig("Reports/cycles_vs_soh.png", dpi=160, bbox_inches="tight")  # Save the scatter plot.
plt.close()  # Close this chart.

plt.figure(figsize=(9, 7))  # Make room for heatmap labels.
sns.heatmap(  # Draw correlations among selected measurements.
    df[feature_columns + ["SoH_Percent"]].corr(),  # Use input columns and the target.
    annot=True,  # Print the correlation values.
    fmt=".2f",  # Show two decimal places.
    cmap="YlGnBu",  # Use a clear blue-green scale.
    vmin=-1,  # Set the lowest possible correlation.
    vmax=1,  # Set the highest possible correlation.
)  # Finish the heatmap.
plt.title("Correlation between inputs and SoH (unitless)")  # Name the heatmap.
plt.tight_layout()  # Fit the labels.
plt.savefig("Reports/correlation_heatmap.png", dpi=160, bbox_inches="tight")  # Save the heatmap.
plt.close()  # Close this chart.

plt.figure(figsize=(7, 6))  # Make room for pie labels.
status_counts = df["Battery_Status"].value_counts()  # Count battery-status labels.
plt.pie(  # Draw the status share chart.
    status_counts,  # Use the category counts.
    labels=status_counts.index,  # Name each status.
    autopct="%1.1f%%",  # Show each status share (%).
    startangle=90,  # Start the first pie slice at the top.
    colors=["#2a9d8f", "#e9c46a"],  # Use clear contrasting colors.
)  # Finish the pie chart.
plt.title("Battery status share (category %)")  # Explain that shares are not SoH.
plt.tight_layout()  # Fit the labels.
plt.savefig("Reports/battery_status_pie.png", dpi=160, bbox_inches="tight")  # Save the pie.
plt.close()  # Close this chart.

plt.figure(figsize=(7, 5))  # Make a clear battery-type chart.
battery_counts = df["Battery_Type"].value_counts()  # Count battery type records.
plt.bar(  # Draw the battery-type counts.
    battery_counts.index,  # Show the battery types.
    battery_counts.values,  # Show each record count.
    color=["#247ba0", "#70c1b3"],  # Use clear colors.
)  # Finish the bar chart.
plt.title("Records by battery type")  # Name the bar chart.
plt.xlabel("Battery type (category)")  # Label the category axis.
plt.ylabel("Number of records (rows)")  # Label the record counts.
plt.tight_layout()  # Fit the labels.
plt.savefig("Reports/battery_type_bar.png", dpi=160, bbox_inches="tight")  # Save the bar chart.
plt.close()  # Close this chart.

# Stage 5: Select five inputs and split rows into train and test groups.
X = df[feature_columns]  # Select age, cycles, temperature, ratio, and discharge rate.
y = df["SoH_Percent"]  # Select battery health as the target (%).
X_train, X_test, y_train, y_test = train_test_split(  # Split records into two groups.
    X,  # Provide the input measurements.
    y,  # Provide the correct SoH answers.
    test_size=0.20,  # Set aside 20 percent of the rows.
    random_state=42,  # Keep the same split each run.
)  # Finish the train/test split.
print("Training rows:", len(X_train), "| Test rows:", len(X_test))  # Show group sizes.
split_summary = pd.DataFrame(  # Make a feature and split summary.
    {  # Start the table columns.
        "Part": ["Training rows", "Test rows", "Input columns", "Target"],  # Name the rows.
        "Value": [len(X_train), len(X_test), len(feature_columns), "SoH_Percent (%)"],  # Add values.
    }  # Finish the table columns.
)  # Finish the split summary.
plt.figure(figsize=(8, 3))  # Make a wide split image.
plt.title("Selected features and data split")  # Name the split image.
plt.axis("off")  # Hide chart axes.
split_table = plt.table(  # Draw the split summary table.
    cellText=split_summary.values,  # Show the values.
    colLabels=split_summary.columns,  # Show the headings.
    cellLoc="left",  # Align the text for easy reading.
    loc="center",  # Center the table.
)  # Finish drawing the table.
split_table.auto_set_font_size(False)  # Keep table text readable.
split_table.set_fontsize(10)  # Set the text size.
split_table.scale(1, 1.5)  # Add space between table rows.
plt.tight_layout()  # Fit title and table.
plt.savefig("Reports/feature_split.png", dpi=160, bbox_inches="tight")  # Save the split image.
plt.close()  # Close this chart.

# Stage 6: Train simple and multiple Linear Regression models.
simple_X_train = X_train[["Total_Charging_Cycles"]]  # Select cycles for simple regression.
simple_X_test = X_test[["Total_Charging_Cycles"]]  # Select test cycles for evaluation.
model_simple = LinearRegression()  # Create the one-input model.
model_simple.fit(simple_X_train, y_train)  # Learn the line using training rows.
model = LinearRegression()  # Create the five-input model.
model.fit(X_train, y_train)  # Learn from the five training measurements.
print(  # Start printing the simple regression equation.
    "Simple line: SoH (%) =",  # Label the predicted target.
    model_simple.coef_[0],  # Show the slope in percent per cycle.
    "percent per cycle * cycles +",  # Explain the slope unit.
    model_simple.intercept_,  # Show the intercept in percent.
    "%",  # Label the intercept unit.
)  # Finish printing the equation.
simple_line = df.sort_values("Total_Charging_Cycles")  # Order cycles to draw the line.
plt.figure(figsize=(8, 5))  # Make a clear regression plot.
plt.scatter(  # Draw the dataset records.
    df["Total_Charging_Cycles"],  # Use cycles on the x-axis.
    df["SoH_Percent"],  # Use SoH on the y-axis.
    s=12,  # Use small points.
    alpha=0.35,  # Make overlapping points visible.
    color="#247ba0",  # Use blue for the observations.
    label="Dataset records",  # Name the observations.
)  # Finish drawing the points.
plt.plot(  # Draw the fitted simple regression line.
    simple_line["Total_Charging_Cycles"],  # Use ordered cycle counts.
    model_simple.predict(simple_line[["Total_Charging_Cycles"]]),  # Predict SoH for the line.
    color="#e63946",  # Use red for the fitted line.
    linewidth=3,  # Make the line easy to see.
    label="Fitted Linear Regression line",  # Name the fitted line.
)  # Finish drawing the fitted line.
plt.title("Simple Linear Regression: cycles and SoH")  # Name the chart.
plt.xlabel("Total charging cycles (cycles)")  # Label the input unit.
plt.ylabel("State of Health (%)")  # Label the target unit.
plt.legend()  # Explain the points and line.
plt.tight_layout()  # Fit the labels and legend.
plt.savefig("Reports/simple_regression_line.png", dpi=160, bbox_inches="tight")  # Save the line plot.
plt.close()  # Close this chart.

# Stage 7: Evaluate both models on the test rows.
simple_prediction = model_simple.predict(simple_X_test)  # Predict using cycles only.
prediction = model.predict(X_test)  # Predict using five measurements.
results = pd.DataFrame()  # Start the model comparison table.
results["Model"] = ["Simple Linear", "Multiple Linear"]  # Name each model.
results["MAE (%)"] = [  # Add mean absolute errors.
    mean_absolute_error(y_test, simple_prediction),  # Simple model error in percentage points.
    mean_absolute_error(y_test, prediction),  # Multiple model error in percentage points.
]  # Finish MAE values.
results["MSE (% squared)"] = [  # Add mean squared errors.
    mean_squared_error(y_test, simple_prediction),  # Simple model squared error.
    mean_squared_error(y_test, prediction),  # Multiple model squared error.
]  # Finish MSE values.
results["RMSE (%)"] = [  # Add root mean squared errors.
    np.sqrt(mean_squared_error(y_test, simple_prediction)),  # Simple model error in percent.
    np.sqrt(mean_squared_error(y_test, prediction)),  # Multiple model error in percent.
]  # Finish RMSE values.
results["R2 (unitless)"] = [  # Add unitless R-squared values.
    r2_score(y_test, simple_prediction),  # Simple model test score.
    r2_score(y_test, prediction),  # Multiple model test score.
]  # Finish R-squared values.
print("Test results:")  # Label the test scores.
print(results.round(4))  # Show scores to four decimal places.
plt.figure(figsize=(10, 3))  # Make a readable results image.
plt.title("Test results: Linear Regression models")  # Name the results image.
plt.axis("off")  # Hide chart axes.
results_table = plt.table(  # Draw the results table.
    cellText=results.round(4).values,  # Show measured scores.
    colLabels=results.columns,  # Show score names and units.
    cellLoc="center",  # Center the score values.
    loc="center",  # Center the table.
)  # Finish drawing the results table.
results_table.auto_set_font_size(False)  # Keep the values readable.
results_table.set_fontsize(10)  # Set table text size.
results_table.scale(1, 1.6)  # Add space around each score.
plt.tight_layout()  # Fit title and table.
plt.savefig("Reports/model_results.png", dpi=160, bbox_inches="tight")  # Save the scores.
plt.close()  # Close this chart.

# Stage 8: Compare actual and predicted SoH for one held-out record.
sample = X_test.iloc[[0]]  # Select one test record.
sample_actual = y_test.iloc[0]  # Read the actual SoH (%).
sample_prediction = model.predict(sample)[0]  # Predict its SoH (%).
print("Held-out sample actual SoH (%):", sample_actual)  # Show the known answer.
print("Held-out sample predicted SoH (%):", sample_prediction)  # Show the prediction.
plt.figure(figsize=(7, 5))  # Make a clear sample comparison image.
plt.bar(  # Draw actual and predicted SoH bars.
    ["Actual SoH (%)", "Predicted SoH (%)"],  # Name each bar.
    [sample_actual, sample_prediction],  # Show measured values.
    color=["#2a9d8f", "#e9c46a"],  # Use different colors for comparison.
)  # Finish the sample comparison.
plt.title("One held-out battery: actual and predicted SoH")  # Name the image.
plt.ylabel("State of Health (%)")  # Label the target unit.
plt.ylim(0, 105)  # Keep both bars on the same percent scale.
plt.tight_layout()  # Fit the labels.
plt.savefig("Reports/sample_prediction.png", dpi=160, bbox_inches="tight")  # Save the sample.
plt.close()  # Close this chart.

# Stage 9: Ask for new measurements and show a new prediction.
age_input = float(input("Vehicle age in months: "))  # Ask for the age.
cycles_input = float(input("Total charging cycles (cycles): "))  # Ask for cycle count.
temperature_input = float(input("Average temperature (degrees C): "))  # Ask for temperature.
charge_input = float(input("Fast-charge ratio (0 to 1): "))  # Ask for the unitless ratio.
discharge_input = float(input("Average discharge rate (C-rate): "))  # Ask for discharge rate.
new_sample = pd.DataFrame(  # Arrange the five new measurements.
    [[age_input, cycles_input, temperature_input, charge_input, discharge_input]],  # Add the entered values.
    columns=feature_columns,  # Match the model input names.
)  # Finish the new sample table.
new_soh = model.predict(new_sample)[0]  # Predict the new SoH (%).
print("Predicted new battery SoH (%):", new_soh)  # Show the estimated health.
plt.figure(figsize=(7, 5))  # Make a clear deployment result image.
plt.bar(  # Draw the predicted SoH bar.
    ["Predicted SoH (%)"],  # Name the bar.
    [new_soh],  # Show the prediction.
    color="#247ba0",  # Use the project blue.
)  # Finish the deployment chart.
plt.title("New battery prediction")  # Name the prediction image.
plt.ylabel("State of Health (%)")  # Label the target unit.
plt.ylim(0, 105)  # Keep the result on a percent scale.
plt.tight_layout()  # Fit the labels.
plt.savefig("Reports/deployment_prediction.png", dpi=160, bbox_inches="tight")  # Save the result.
plt.close()  # Close this chart.

# Stage 10: Later, compare predictions with measured SoH and check errors.
