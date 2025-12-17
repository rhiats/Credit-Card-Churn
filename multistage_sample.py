from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = pd.read_pickle("X_train.pkl")
y = pd.read_pickle("y_train.pkl")

X["y"] = y

existing_customer_num = X["y"].value_counts()["Existing Customer"]
attrited_customer_num = X["y"].value_counts()["Attrited Customer"]

print("Number of Existing Customers are: {}".format(existing_customer_num))
print("Number of Attrited Customers are: {}".format(attrited_customer_num)) #(868 attrited customers)

#Cluster existing customers Use the elbow method to select number of clusters
X = X[['Customer_Age','Dependent_count','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct','y']]

X_exist = X[X["y"]=="Existing Customer"]

X_exist = X[['Customer_Age','Dependent_count','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_exist)




