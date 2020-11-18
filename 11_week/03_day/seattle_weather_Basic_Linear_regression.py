# =================================================================================
# Arwa Ashi - HW3 - Week 11 - Nov 17, 2020
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Assignment
# Build a linear regression model using the last two days of weather data (use
# today and yesterday to predict tomorrow). Use the examples from the linear
# regression folder and make changes to the assignment code to accomplish this task. 

# Only use scikit-learn to train your model.  

# Build a Linear Regression Model using Scikit-learn
# =================================================================================

# Packages
# ---------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
# ---------------------------------------------------------------------------------
df = pd.read_csv('https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/module_4_ML/data/seattle_weather_1948-2017.csv')

#print(df.shape)#(25551, 5)
numrows = 25549 # can be as large as 25549

# Create an empty dataframe to hold values
regression_df = pd.DataFrame({"intercept"       :[1.0]*numrows,
                              'before_yesterday':[0.0]*numrows,
                              'yesterday'       :[0.0]*numrows,
                              "today"           :[0.0]*numrows,
                              "tomorrow"        :[0.0]*numrows})

# Sort columns for convience
seq = ['intercept','before_yesterday', 'yesterday', 'today','tomorrow']

regression_df = regression_df.reindex(columns=seq)

for i in range(0,numrows):
    tomorrow               = df.iloc[i,1]
    today                  = df.iloc[(i-1),1]
    yesterday              = df.iloc[(i-2),1]
    before_yesterday       = df.iloc[(i-3),1]
    regression_df.iat[i,4] = tomorrow
    regression_df.iat[i,3] = today
    regression_df.iat[i,2] = yesterday
    regression_df.iat[i,1] = before_yesterday

# This makes a simple dataframe with a relationship that we can now plot
print(regression_df.describe)
'''
<bound method NDFrame.describe of        intercept  before_yesterday  yesterday  today  tomorrow
0            1.0              0.00       0.00   0.00      0.47
1            1.0              0.00       0.00   0.47      0.59
2            1.0              0.00       0.47   0.59      0.42
3            1.0              0.47       0.59   0.42      0.31
4            1.0              0.59       0.42   0.31      0.17
...          ...               ...        ...    ...       ...
25544        1.0              0.00       0.00   0.00      0.00
25545        1.0              0.00       0.00   0.00      0.00
25546        1.0              0.00       0.00   0.00      0.00
25547        1.0              0.00       0.00   0.00      0.00
25548        1.0              0.00       0.00   0.00      0.00

[25549 rows x 5 columns]>
'''
sns.scatterplot(x='today',y='tomorrow',data=regression_df)
plt.show()

# regression
# ---------------------------------------------------------------------------------
from sklearn import linear_model

# df.dropna()
regression_df = regression_df.dropna()

# modify the data to work with this library
x = regression_df[['today','yesterday','before_yesterday']]
y = regression_df['tomorrow']
print(x)
print(y)

mymodel = linear_model.LinearRegression().fit(x,y)

# using the r2 (pronounced r squared) value we can get a basic measure of model quality
from sklearn.metrics import r2_score
print('r2 score = ',r2_score(y,mymodel.predict(x)))
# r2 score =  0.10174813361087187

# we can plot the difference between the prediction ans the actual values for a visual estimate
# of  performance. A perfect model would result in this being a straight line with a slop of 1.
# Notice, how the model predicts only lower values, meaning that it tends to under predict
# the actual amount of rain

plt.scatter(mymodel.predict(x),y, color="red")
plt.xlim(-0.1, 2.1) # to keep the same scale as the revious plot
plt.ylim(-0.1, 2.1) # same reason as xlim
plt.show()


