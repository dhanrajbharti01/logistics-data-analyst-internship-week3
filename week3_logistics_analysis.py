import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load logistics dataset
df = pd.read_csv("logistics_orders.csv")

# Step 1: Dataset Overview
print("--- Step 1: Dataset Overview ---")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

# Step 2: Exploratory Data Analysis (EDA)

print("\n--- Step 2: Exploratory Data Analysis ---")

print("\nAverage Delivery Time:")
print(df["actual_time_min"].mean())

print("\nMedian Delivery Time:")
print(df["actual_time_min"].median())

print("\nAverage Distance:")
print(df["distance_km"].mean())

print("\nAverage Parcel Count:")
print(df["parcel_count"].mean())

print("\nTraffic Level Distribution:")
print(df["traffic_level"].value_counts())

print("\nAverage Delivery Time by Traffic Level:")
print(df.groupby("traffic_level")["actual_time_min"].mean())

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Step 3: Visualizations

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Traffic Level vs Average Delivery Time
traffic_avg = df.groupby("traffic_level")["actual_time_min"].mean()

plt.figure(figsize=(7, 5))
traffic_avg.plot(kind="bar")
plt.title("Average Delivery Time by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 2. Distance vs Actual Delivery Time
plt.figure(figsize=(7, 5))
plt.scatter(df["distance_km"], df["actual_time_min"])
plt.title("Distance vs Actual Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Actual Delivery Time (minutes)")
plt.tight_layout()
plt.show()


# 3. Actual Delivery Time Distribution
plt.figure(figsize=(7, 5))
plt.hist(df["actual_time_min"], bins=10)
plt.title("Distribution of Actual Delivery Time")
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()


# 4. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. Delivery Time Distribution

plt.figure(figsize=(8, 5))
plt.hist(df["actual_time_min"], bins=10, edgecolor="black")

plt.title("Distribution of Actual Delivery Time")
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()