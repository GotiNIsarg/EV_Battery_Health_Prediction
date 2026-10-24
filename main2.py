import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")

column = "Internal_Resistance_Ohm"

q1 = df[column].quantile(0.25)
q3 = df[column].quantile(0.75)
iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

outliers = df[
    (df[column] < lower_limit) |
    (df[column] > upper_limit)
]

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
print("Potential outliers:", len(outliers))
