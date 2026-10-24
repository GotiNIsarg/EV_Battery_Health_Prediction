
import pandas as pd
import matplotlib.pyplot as plt

# Load your EV battery dataset
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")

# Check the battery health column
print(df["SoH_Percent"].describe())

# Plot a histogram
plt.hist(df["SoH_Percent"], bins=20, edgecolor="black")
plt.title("EV Battery State of Health Distribution")
plt.xlabel("State of Health (%)")
plt.ylabel("Number of Vehicles")
plt.show()

print("Mean:", df["SoH_Percent"].mean())
print("Median:", df["SoH_Percent"].median())
print("Skewness:", df["SoH_Percent"].skew())    