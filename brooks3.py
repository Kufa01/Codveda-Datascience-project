# PREDICTIVE MODEL REGRESSION #


#IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LOAD AND PREPARE DATA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("Datasets/housePredictiondatasets.csv",sep="\s+", header=None)

df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
"DIS","RAD","TAX","PTRTATIO","B","LSTAT","MEDV"
]    

#DIFINE FEATURES(X) AND TARGET(Y)
x = df.drop("MEDV", axis=1)
y = df["MEDV"]



# TRAIN-TEST SPLIT

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#TRAIN LINEAR REGRESSION MODEL

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred_lr = lr.predict(x_test)

#EVALUATE MODEL

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test,y_pred_lr)

print("Linear Regression:")
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)


#DECISION TREE MODEL


dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)


print("\nDecision tree:")
print("MSE:", mean_squared_error(y_test,y_pred_dt))
print("R2 Score:", r2_score(y_test, y_pred_dt))


#RANDOM FOREST MODEL

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("\nRandom Forest:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

#COMPARE MODELS

results = pd.DataFrame({
    "Model":["Linear Regression", "Decision Tree","Random Forest"],
    "MSE": [
        mse_lr,
        mean_squared_error(y_test, y_pred_dt),
        mean_squared_error(y_test, y_pred_rf)
    ],
    "R2 Score":[
       r2_lr,
       r2_score(y_test, y_pred_dt),
       r2_score(y_test, y_pred_rf) 
    ]

})

print(results)



#VISUALIZATION(ACTUAL VS PREDICTED)

plt.scatter(y_test,y_pred_rf)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest Predictions")
plt.show


#RESULTS INTERPRETATION

#MSE(Means Squared Error)
# 1)Lower = better
# 2)Measures prediction error

#R2 Score
# 1)Closer to 1 = better
# example:
     ## 0.85--->very good
     ## 0.60--->average
     ## 0.50--->weak


     ### MODEL SUMMARY:
        # Three regression models were trained: Linear Regression,Decision Tree, and Random Forest. Random Forest achieved the best performance with the lowest MSE and highest R2 score.