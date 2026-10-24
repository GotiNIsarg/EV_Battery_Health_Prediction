import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")

# Step 2: Display the first five rows
print("\nFIRST FIVE ROWS")
print(df.head())

# Step 3: Check the number of rows and columns
print("\nDATASET SHAPE")
print(df.shape)

# Step 4: Check column names and data types
print("\nDATASET INFORMATION")
df.info()

# Step 5: Calculate summary statistics
print("\nNUMERICAL SUMMARY")
print(df.describe())

# Step 6: Count missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Step 7: Count duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Step 8: Plot battery health distribution
plt.hist(df["SoH_Percent"], bins=20, edgecolor="black")
plt.title("Distribution of EV Battery State of Health")
plt.xlabel("State of Health (%)")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.show()
