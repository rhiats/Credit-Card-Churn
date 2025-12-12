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

X_train['Avg_Trans_Amt'] = X_train['Total_Trans_Amt'] / X_train['Total_Trans_Ct'] #Captures spending intensity per swipe.

X_train['Trans_Amt_to_Limit'] = X_train['Total_Trans_Amt'] / X_train['Credit_Limit'] #utilization amount

X_train['Trans_per_Active_Month'] = X_train['Total_Trans_Ct'] / (12 - X_train['Months_Inactive_12_mon']) #Transactions per Month Active

X_train['Age_x_CreditLimit'] = X_train['Customer_Age'] * X_train['Credit_Limit'] #combinatorial properties

X_train['TransCt_x_Inactive'] = X_train['Total_Trans_Ct'] * X_train['Months_Inactive_12_mon']






df_v = df_v[['Customer_Age','Dependent_count','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct']]

df_v['Avg_Trans_Amt'] = df_v['Total_Trans_Amt'] / df_v['Total_Trans_Ct']

df_v['Trans_Amt_to_Limit'] = df_v['Total_Trans_Amt'] / df_v['Credit_Limit']

df_v['Trans_per_Active_Month'] = df_v['Total_Trans_Ct'] / (12 - df_v['Months_Inactive_12_mon'])

df_v['Age_x_CreditLimit'] = df_v['Customer_Age'] * df_v['Credit_Limit']

df_v['TransCt_x_Inactive'] = df_v['Total_Trans_Ct'] * df_v['Months_Inactive_12_mon']


classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(df_v)

score = f1_score(y_valid, y_pred)

print("F1 Score: {}".format(score)) 

#F1 Score Iteration 1: 0.7452054794520548
#F1 Score Interation 2: 0.7493112947658402 (Add Avg_Trans_Amt)
#F1 Score Interation 3: 0.7613941018766756 (Add TransCt_x_Inactive)