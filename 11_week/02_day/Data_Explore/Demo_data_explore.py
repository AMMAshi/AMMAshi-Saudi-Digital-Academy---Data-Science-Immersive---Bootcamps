# Nov 16, 2020
#----------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-colorblind')
# %matplotlib inline
import explore

# Read the dataset
#----------------------------------------------------------------------------------------------------
use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp','Survived']
data = pd.read_csv('titanic.csv', usecols=use_cols)
print(data.head(3))


# Get dtypes for each columns
#----------------------------------------------------------------------------------------------------
str_var_list, num_var_list, all_var_list = explore.get_dtypes(data=data)
print(str_var_list) # string type
print(num_var_list) # numeric type
print(all_var_list) # all

# General data description
#----------------------------------------------------------------------------------------------------
print(explore.describe(data=data))


# Discrete variable barplot
# draw the barplot of a discrete variable x against y(target variable).
# By default the bar shows the mean value of y.
#----------------------------------------------------------------------------------------------------
explore.discrete_var_barplot(x='Pclass',y='Survived',data=data)


# Discrete variable countplot
# draw the countplot of a discrete variable x
#----------------------------------------------------------------------------------------------------
explore.discrete_var_countplot(x='Pclass',data=data)


# Discrete variable boxplot 
# draw the boxplot of a discrete variable x against y.
#----------------------------------------------------------------------------------------------------
explore.discrete_var_boxplot(x='Pclass',y='Fare',data=data)


# Continuous variable distplot
# draw the distplot of a continuous variable x.
#----------------------------------------------------------------------------------------------------
explore.continuous_var_distplot(x=data['Fare'])


# Scatter plot
# draw the scatter-plot of two variables.
#----------------------------------------------------------------------------------------------------
explore.scatter_plot(x=data.Fare,y=data.Pclass,data=data)#,output_path='./output/')


# Correlation plot
# draw the correlation plot between variables.
#----------------------------------------------------------------------------------------------------
explore.correlation_plot(data=data)


# Heatmap
#----------------------------------------------------------------------------------------------------
flights = sns.load_dataset("flights")
print(flights.head(5))
# explore.heatmap(data=data[['Sex','Survived']])
flights = flights.pivot("month", "year", "passengers")
explore.heatmap(data=flights)


#----------------------------------------------------------------------------------------------------
plt.show()













