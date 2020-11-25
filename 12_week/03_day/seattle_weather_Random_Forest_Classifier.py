# =================================================================================
# Arwa Ashi - HW3 - Week 12 - Nov 24, 2020
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Assignment
# Build a classification model of the Seattle Weather dataset with multiple features
# using scikit-learn's random forest classifier. 

# Make sure you separate training and test sets and evaluate the models performance
# separately on these subsets. 

# Use the same feature set you developed for the logistic regression assignment.
# How did the two perform against each other?

#  Build a random forest with multiple features by training and testing on seperate data
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
randomforest_df = pd.DataFrame({'before_yesterday':[0.0]*numrows,
                              'yesterday'         :[0.0]*numrows,
                              'today'             :[0.0]*numrows,
                              'tomorrow'          :[True]*numrows})

# Sort columns for convience
seq = ['before_yesterday', 'yesterday', 'today','tomorrow']

randomforest_df = randomforest_df.reindex(columns=seq)

for i in range(0,numrows):
    tomorrow                 = df.iloc[i,1]
    today                    = df.iloc[(i-1),1]
    yesterday                = df.iloc[(i-2),1]
    before_yesterday         = df.iloc[(i-3),1]
    randomforest_df.iat[i,3] = tomorrow
    randomforest_df.iat[i,2] = today
    randomforest_df.iat[i,1] = yesterday
    randomforest_df.iat[i,0] = before_yesterday
# print(regression_df.head(20))

# random forest
# ---------------------------------------------------------------------------------
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics

# df.dropna()
randomforest_df = randomforest_df.dropna()

# modify the data to work with this library
x = randomforest_df[['today','yesterday','before_yesterday']]
y = randomforest_df['tomorrow']
# print(x)
# print(y)

# note that we did not need to reshape the y values as we did with linear regression
clf_RF = ensemble.RandomForestClassifier(n_estimators=10).fit(x, y)
clf_LR = linear_model.LogisticRegression(solver='lbfgs').fit(x, y)

# we can calculate the accuarcy using the score method
score_RF = clf_RF.score(x,y)
print('score',score_RF) # score 0.8157152924594785 <<<<<<< Random Forest


score_LR = clf_LR.score(x,y)
print('score',score_LR) # score 0.6758671991230131 <<<<<<< logistic Regression

# prediction
# ---------------------------------------------------------------------------------
from sklearn import metrics
# we can make a simple sonfusion matrix
predictions = clf_RF.predict(x)
cm          = metrics.confusion_matrix(y,predictions)
print(cm)

# Here is a bit nicer matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="Blues_r")
plt.ylabel('Actual Label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score_RF)
plt.title(all_sample_title, size=15)
plt.show()
