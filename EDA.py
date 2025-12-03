"""
	Rhia Singh
	File to perform Exploratory Data Analysis on Training dataset
""" 
from scipy import stats 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu

"""
	Distribtuion of Continuous Variables
"""

def distribution(data,title,x_label,y_label,filename):
	"""
		Plot the distribution of a continuous variable.
		data: Variable of interest
	"""

	# 'kde=True' adds a Kernel Density Estimation line
	sns.histplot(data)

	# 3. Add titles and labels
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	# 4. Show the plot
	plt.savefig(filename)
	plt.close()



def mannwhitney(batch_1,batch_2):
	"""
		Perform the mann whitney u test on distributions that are not normally distributed.
		batch1 (np.array): Group 1
		batch2 (np.array): Group 2
	"""

	stat, p_value = mannwhitneyu(batch_1, batch_2)
	print('Statistics=%.2f, p=%.2f' % (stat, p_value))
	alpha = 0.05
	if p_value < alpha:
	    print('Reject Null Hypothesis (Significant difference between two samples)')
	else:
	    print('Do not Reject Null Hypothesis (No significant difference between two samples)')


df = pd.read_pickle("X_train.pkl")
df_y = pd.read_pickle("y_train.pkl")

df["Attrition_Flag"] = df_y

#print(df["Dependent_count"].describe())

distribution(df["Customer_Age"],"Customer Age Distribution","Customer Age","Frequency","output/Customer Age Distribution")
distribution(df["Dependent_count"],"Dependent Distribution","Number of Dependents","Frequency","output/Dependent Distribution")
distribution(df["Credit_Limit"],"Credit Limit Distribution","Credit Limit","Frequency","output/Credit Limit Distribution")

existing_customer_credit_limit_df = df[df["Attrition_Flag"]=="Existing Customer"]["Credit_Limit"]

attrited_customer_credit_limit_df = df[df["Attrition_Flag"]=="Attrited Customer"]["Credit_Limit"]


mean_value_existing = existing_customer_credit_limit_df.mean()
mean_value_attrited = attrited_customer_credit_limit_df.mean()

print("The mean of Existing customer credit limit is {0}".format(mean_value_existing))
print("The mean of Attrited customer credit limit is {0}".format(mean_value_attrited))

ax = sns.histplot(data=df, x="Credit_Limit", hue="Attrition_Flag")
ax.axvline(mean_value_existing, color='blue', linestyle='dashed', linewidth=2)
ax.axvline(mean_value_attrited, color='gray', linestyle='dashed', linewidth=2)
plt.savefig("output/Distribution of Credit Credit Limit by Attrition Flag")

mannwhitney(existing_customer_credit_limit_df,attrited_customer_credit_limit_df)



