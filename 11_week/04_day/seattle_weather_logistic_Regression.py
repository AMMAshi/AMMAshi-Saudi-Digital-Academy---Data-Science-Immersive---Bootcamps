# =================================================================================
# Arwa Ashi - HW4 - Week 11 - Nov 18, 2020
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Assignment
# Use scikit-learn and build a logistic regression prediction model, with at least
# three input variables, for the Seattle Weather data or any data you choose.

# Predict if it is going to rain tomorrow (true or false). Iterate on your feature
# set until you have a performance that you are happy with.

#  Scikit-learn Logistic Regression
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
regression_df = pd.DataFrame({'before_yesterday':[0.0]*numrows,
                              'yesterday'       :[0.0]*numrows,
                              'today'           :[0.0]*numrows,
                              'tomorrow'        :[True]*numrows})

# Sort columns for convience
seq = ['before_yesterday', 'yesterday', 'today','tomorrow']

regression_df = regression_df.reindex(columns=seq)

for i in range(0,numrows):
    tomorrow               = df.iloc[i,1]
    today                  = df.iloc[(i-1),1]
    yesterday              = df.iloc[(i-2),1]
    before_yesterday       = df.iloc[(i-3),1]
    regression_df.iat[i,3] = tomorrow
    regression_df.iat[i,2] = today
    regression_df.iat[i,1] = yesterday
    regression_df.iat[i,0] = before_yesterday
# print(regression_df.head(20))

# regression
# ---------------------------------------------------------------------------------
from sklearn import linear_model

# df.dropna()
regression_df = regression_df.dropna()

# modify the data to work with this library
x = regression_df[['today','yesterday','before_yesterday']]
y = regression_df['tomorrow']
# print(x)
# print(y)

# note that we did not need to reshape the y values as we did with linear regression
clf = linear_model.LogisticRegression(solver='lbfgs').fit(x, y)

# we can calculate the accuarcy using the score method
score = clf.score(x,y)
print('score',score) # score 0.6758671991230131

# prediction
# ---------------------------------------------------------------------------------
from sklearn import metrics
# we can make a simple sonfusion matrix
predictions = clf.predict(x)
cm          = metrics.confusion_matrix(y,predictions)
print(cm)

# Here is a bit nicer matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="Blues_r")
plt.ylabel('Actual Label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()
