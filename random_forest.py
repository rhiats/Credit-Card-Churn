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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.linear_model import LogisticRegression


#df = pd.read_pickle("X_train.pkl")
#df_y = pd.read_pickle("y_train.pkl")

##Undersampling
#df = pd.read_pickle("training_random_sample.pkl")
#df_y = df['y']

##Multi-stage Sampling
df = pd.read_pickle("training_multistage_sample.pkl")
df_y = df['y']



df_v = pd.read_pickle("X_validation.pkl")
df_y_v = pd.read_pickle("y_validation.pkl")


y_train = df_y.replace({'Attrited Customer': 1, 'Existing Customer': 0})
y_valid= df_y_v.replace({'Attrited Customer': 1, 'Existing Customer': 0})

#X_train = df[['Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status','Income_Category','Card_Category','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct']]

X_train = df[['Customer_Age','Dependent_count','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct']]

df_v = df_v[['Customer_Age','Dependent_count','Months_Inactive_12_mon','Credit_Limit','Total_Trans_Amt','Total_Trans_Ct']]


def feature_engineering(df):
    """
        Add engineered features to boost model performance

        @p: df: Dataframe with raw data
        @r: df: Dataframe with engineered data
    """

    df['Avg_Trans_Amt'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct'] #Captures spending intensity per swipe.

    df['Trans_Amt_to_Limit'] = df['Total_Trans_Amt'] / df['Credit_Limit'] #utilization amount

    df['Trans_per_Active_Month'] = df['Total_Trans_Ct'] / (12 - df['Months_Inactive_12_mon']) #Transactions per Month Active

    df['Age_x_CreditLimit'] = df['Customer_Age'] * df['Credit_Limit'] #combinatorial properties

    df['TransCt_x_Inactive'] = df['Total_Trans_Ct'] * df['Months_Inactive_12_mon']

    df['Low_Activity'] = (df['Total_Trans_Ct'] < df['Total_Trans_Ct'].median()).astype(int) #Low activity based on credit card activity

    return df


X_train = feature_engineering(X_train)

df_v = feature_engineering(df_v)


#Search Grid

rf = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 400, 600],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, None],
    'bootstrap': [True, False]
    #'class_weight': ['balanced', 'balanced_subsample']
}

rs = RandomizedSearchCV(
    rf, param_dist, n_iter=30,
    scoring='f1', cv=5, verbose=2, n_jobs=-1
)
rs.fit(X_train, y_train)

best_rf = rs.best_estimator_

y_pred = best_rf.predict(df_v)

score = f1_score(y_valid, y_pred)

print("F1 Score Random Forest: {}".format(score)) 

print("Best RF:",best_rf)


"""
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(df_v)

score = f1_score(y_valid, y_pred)

print("F1 Score: {}".format(score)) 
"""

#Confusion Matrix
# Display the confusion matrix
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1',"0"])
disp.plot(cmap=plt.cm.Blues) # You can customize the colormap
plt.show()

# To display a normalized confusion matrix (by 'true' labels, i.e., by row)
disp_norm = ConfusionMatrixDisplay.from_estimator(
    best_rf, df_v, y_valid,
    display_labels=['1',"0"],
    cmap=plt.cm.Blues,
    normalize='true' # 'true', 'pred', or 'all'
)
disp_norm.ax_.set_title("Normalized Confusion Matrix")
plt.show()


#ROC Curve

# 5. Calculate ROC curve metrics manually
fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
roc_auc = auc(fpr, tpr) 

# 6. Plot the ROC curve using the visualization API
display = RocCurveDisplay.from_estimator(best_rf, df_v, y_valid, name="Random Forest")
plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)') # Add the random chance line
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



#F1 Score Iteration 1: 0.7452054794520548
#F1 Score Interation 2: 0.7493112947658402 (Add Avg_Trans_Amt)
#F1 Score Interation 3: 0.7613941018766756 (Add TransCt_x_Inactive)
#f1 Score Iteration 4: 0.7643979057591623 (Random Search)
#f1 Score Iteration 5: 0.7251732101616628 (No change)
#F1 Score Iteration 6: 0.7325842696629213 (No change)
#F1 Score Random Forest 7: 0.7248322147651006 (No change)
#F1 Score Random Forest 8: 0.6863905325443787 (Random sample majority class)
#F1 Score Random Forest 9: 0.6892430278884463 (Multistage sample majority class)


#Logistic Regression
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(df_v)

score_lr = f1_score(y_valid, y_pred)

print("F1 Score Logistic Regression: {}".format(score)) 


#Confusion Matrix
# Display the confusion matrix
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1',"0"])
disp.plot(cmap=plt.cm.Blues) # You can customize the colormap
plt.show()

# To display a normalized confusion matrix (by 'true' labels, i.e., by row)
disp_norm = ConfusionMatrixDisplay.from_estimator(
    logreg, df_v, y_valid,
    display_labels=['1',"0"],
    cmap=plt.cm.Blues,
    normalize='true' # 'true', 'pred', or 'all'
)
disp_norm.ax_.set_title("Normalized Confusion Matrix Logistic Regression")
plt.show()


#ROC Curve

# 5. Calculate ROC curve metrics manually
fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
roc_auc = auc(fpr, tpr) 

# 6. Plot the ROC curve using the visualization API
display = RocCurveDisplay.from_estimator(logreg, df_v, y_valid, name="Random Forest")
plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)') # Add the random chance line
plt.title('Receiver Operating Characteristic (ROC) Curve Logistic Regression')
plt.legend(loc="lower right")
plt.show()
#F1 Score Logistic Regression: 0.7240618101545254 (Iteration 1) #Logistic regression performs just as well as random forest, so I won't stack the models

#Remove correlated features



