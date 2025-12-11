"""
	Rhia Singh
	Train Random Forest model to predict churn
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


df = pd.read_pickle("X_train.pkl")
df_y = pd.read_pickle("y_train.pkl")

df_v = pd.read_pickle("X_validation.pkl")
df_y_v = pd.read_pickle("y_validation.pkl")


y_train = df_y.replace({'Attrited Customer': 1, 'Existing Customer': 0})
y_valid= df_y_v.replace({'Attrited Customer': 1, 'Existing Customer': 0})

#X_train = df[['Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status','Income_Category','Card_Category','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct']]

X_train = df[['Customer_Age','Dependent_count','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct']]

df_v = df_v[['Customer_Age','Dependent_count','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct']]


classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(df_v)

score = f1_score(y_valid, y_pred)

print("F1 Score: {}".format(score))