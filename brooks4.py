## CLASSIFICATION WITH LOGISTIC REGRESSION



##IMPORT LIBRARIES


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



from sklearn.metrics import(
    accuracy_score, precision_score,recall_score,
    confusion_matrix, classification_report,
    roc_curve,auc
)


##LOAD DATASET

iris = load_iris()
x = iris.data
y = iris.target

df = pd.DataFrame(x, columns=iris.feature_names)
df["target"] = y
print(df.head())


###TRAIN-TEST SPLIT

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

####FEATURE SCALING(important for Logistic and SVM)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#####Train Logistic Regression

lr = LogisticRegression(max_iter=200)
lr.fit(x_train,y_train)

y_pred_lr = lr.predict(x_test)

######EVALUATE MODEL

print("Accuracy:",accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average='weighted'))
print("Recal:", recall_score(y_test,y_pred_lr,average='weighted'))

print("\nClassification Report:\n")
print(classification_report(y_test,y_pred_lr))

#######CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred_lr)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


######## ROC CURVE (important Concept)
#since iris is multiclass, we convert to binary for ROC


from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_score = lr.predict_proba(x_test)

fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_score[:, 0])
roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr, label="ROC curve (area = %0.2f)" % roc_auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


#########COMPARE WITH OTHER MODELS

#DECISION TREE
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)

#RANDOM FOREST

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

#SVM

svm = SVC(probability=True)
svm.fit(x_train,  y_train)

y_pred_svm = svm.predict(x_test)

##########COMPARE RESULTS

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
    "Accuracy":[
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_svm)
    ]
})

print(results)



# REPORT SUMMARY #

#Logistic Regression was trained to classify flower species.
#The model achieved high accuracy and performed well across precision and recall metrics.
#When compared with other models,Random Forest and SVM showed competitive or improved performance due to their ability to capture complex patterns.


