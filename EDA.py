"""
	Rhia Singh
	File to perform Exploratory Data Analysis on Training dataset
"""
import numpy as np  
from scipy import stats 
import pandas as pd
import numpy as np 

"""
	T-Test
""" 

def t_test(a,b):
	"""
		Perform t-test to determine if sample averages are different between groups.
		a (np.array): Group 1
		b (np.array): Group 2
	"""

	# Sample Sizes
	N1, N2 = len(a), len(b) 

	# Degrees of freedom  
	dof = min(N1,N2) - 1

	# Gaussian distributed data with mean = a and var = 1  
	x = np.random.randn(N1) + a.mean()

	# Gaussian distributed data with mean = b and var = 1  
	y = np.random.randn(N2) + b.mean()

	## Using the internal function from SciPy Package  
	t_stat, p_val = stats.ttest_ind(x, y)  
	print("t-statistic = " + str(t_stat))  
	print("p-value = " + str(p_val))