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


def exisitingCustomerdf(data, columnOfInterest):
	"""
		Helper function to return a pandas column of existing customers.
		
		@p data: Dataframe with all the training data
		@p columnOfInterest: Column of interest
		@r filter for existing customers
	"""

	df = data[data["Attrition_Flag"]=="Existing Customer"]

	return df[columnOfInterest]

def attritedCustomerdf(data, columnOfInterest):
	"""
		Helper function to return a pandas column of attrited customers.
		
		@p data: Dataframe with all the training data
		@p columnOfInterest: Column of interest
		@r filter for attrited customers
	"""

	df = data[data["Attrition_Flag"]=="Attrited Customer"]

	return df[columnOfInterest]


def calcMean(col):
	"""
		Return the mean of the column

		@p col: column
		@r: mean of the column
	"""


	return col.mean()


def distByClass(x_value,filename):
	"""
		Plot the feature by class.

		@p x_value: The feature of interest
		@p filename: The name of the output file
	"""

	ax = sns.histplot(data=df, x=x_value, hue="Attrition_Flag")
	ax.axvline(mean_value_existing, color='blue', linestyle='dashed', linewidth=2)
	ax.axvline(mean_value_attrited, color='gray', linestyle='dashed', linewidth=2)
	plt.savefig(filename)
	plt.close()


def barplot(dataframe,feature,titleGraph,filename):
	"""
		Plot a bar graph of categorical features

		@p dataframe: The dataframe will all the data
		@p feature: Feature that will be plotted on the x axis
		@p titleGraph: Title of graph
	"""

	df = dataframe[[feature,'CLIENTNUM']].groupby([feature]).count()
	df.reset_index(inplace=True)

	df.rename(columns={"CLIENTNUM":"Frequency"},inplace = True)

	df.sort_values(by='Frequency',inplace=True,ascending=False)

	plt.figure(figsize=(8, 6))
	sns.barplot(x=feature, y="Frequency", data=df)
	plt.title(titleGraph)
	plt.savefig(filename)
	plt.close()


df = pd.read_pickle("X_train.pkl")
df_y = pd.read_pickle("y_train.pkl")

df["Attrition_Flag"] = df_y

#print(df["Dependent_count"].describe())

distribution(df["Customer_Age"],"Customer Age Distribution","Customer Age","Frequency","output/Customer Age Distribution")
distribution(df["Dependent_count"],"Dependent Distribution","Number of Dependents","Frequency","output/Dependent Distribution")
distribution(df["Credit_Limit"],"Credit Limit Distribution","Credit Limit","Frequency","output/Credit Limit Distribution")


existing_customer_credit_limit_df = exisitingCustomerdf(df,"Credit_Limit")
attrited_customer_credit_limit_df = attritedCustomerdf(df,"Credit_Limit")


mean_value_existing = calcMean(existing_customer_credit_limit_df)
mean_value_attrited = calcMean(attrited_customer_credit_limit_df)

distByClass("Credit_Limit","output/Distribution of Credit Credit Limit by Attrition Flag")


mannwhitney(existing_customer_credit_limit_df,attrited_customer_credit_limit_df)

barplot(df,"Gender","Frequency of Each Gender", "output/gender_freq")
barplot(df,"Education_Level","Frequency of Each Education Level", "output/Education_Level_freq")
barplot(df,"Marital_Status","Frequency of Marital Status", "output/Marital_Status_freq")





