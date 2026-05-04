# EXPLORATORY DATA ANALYSIS (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Load the dataset
df = pd.read_csv("./Datasets/housePredictiondatasets.csv", sep="\s+",header=None)
columns = [
"CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
"DIS","RAD","TAX","PTRTATIO","B","LSTAT","MEDV"
]    

df.columns = columns
df = df.apply(pd.to_numeric)

print(df.dtypes)
print(df.head())
print(df.info())

#Understand the data

print(df.shape)
print(df.columns)
print(df.describe())

# #Compute Summary Statistics

# #Mean
print(df.mean(numeric_only=True))

# #Median
print(df.median(numeric_only=True))

# #Variance
print(df.var(numeric_only=True))

# #Check Missing Value

print(df.isnull().sum)

# #Visualisation

# # 1.Histogram
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()


#Box Plot(Outliers Detection)

plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

#Scatter Plot(Relationships)

plt.scatter(df['RM'], df['MEDV'])
plt.xlabel('Average Rooms(RM)')
plt.ylabel('House Price(MEDV)')
plt.show()


#Correlation Matrix

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()




## EDA REPORT

## Summary Statistics

# 1) The average house price(MEDV) is around X
# 2) Features like RM(rooms) have strong variation

## Distribution Insights

# 1) Some features are skewed(e.g., crime rate)
# 2) House price are slightly right-skewed

## Outliers

# Features like CRIM and B contain extreme values

## Relationships

# 1) Strong positive correlation:
       # RM--->MEDV(house price)

# 2) Strong negative correlation:
       # LSTAT--->MEDV(lower status--->lower price)

## Key Drivers of Price

# 1. Number of rooms(RM)
# 2. Socioeconomic status(LSTAT)
# 3. Property tax(TAX)








