import pandas as pd

df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
print(df.head())
# Count missing values in every column
print("Missing values:")
print(df.isnull().sum())

# Count all missing cells
print("\nTotal missing cells:")
print(df.isnull().sum().sum())

# Count duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())
