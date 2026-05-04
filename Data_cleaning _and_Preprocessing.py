
# Data Cleaning and Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("./Datasets/iris.csv")

print(df.head())
print(df.info())

# Check for missing data
print("check for missing values")
print(df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)

# Detect and Remove Outliers

num_df = df.select_dtypes(include=['number'])

Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)   
IQR = Q3 - Q1

# Filter rows (keep only non-outliers)
filter_mask = ~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)

df = df[filter_mask]   

# Convert categorical variables
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Normalize/Standardize Numerical Data
print("Normalize Numerical Data")
scaler = StandardScaler()

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.head())
print(df.describe())