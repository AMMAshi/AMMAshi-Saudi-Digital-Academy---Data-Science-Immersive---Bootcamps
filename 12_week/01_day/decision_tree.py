# =================================================================================
# Arwa Ashi - HW1 - Week 12 - Nov 22, 2020
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Assignment
# Work through the Decision Tree Example
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

# print(df.head())

#print(df.shape)#(25551, 5)
numrows = 25549 # can be as large as 25549

# Create an empty dataframe to hold values
decision_tree_df = pd.DataFrame({'before_yesterday':[0.0]*numrows,
                                 'yesterday'       :[0.0]*numrows,
                                 'today'           :[0.0]*numrows,
                                 'tomorrow'        :[True]*numrows})

# Sort columns for convience
seq = ['before_yesterday', 'yesterday', 'today','tomorrow']

decision_tree_df = decision_tree_df.reindex(columns=seq)

for i in range(0,numrows):
    tomorrow               = df.iloc[i,1]
    today                  = df.iloc[(i-1),1]
    yesterday              = df.iloc[(i-2),1]
    before_yesterday       = df.iloc[(i-3),1]
    decision_tree_df.iat[i,3] = tomorrow
    decision_tree_df.iat[i,2] = today
    decision_tree_df.iat[i,1] = yesterday
    decision_tree_df.iat[i,0] = before_yesterday
# print(decision_tree_df.head(20))

# Decision Tree
# ---------------------------------------------------------------------------------
from sklearn import tree

# df.dropna()
decision_tree_df = decision_tree_df.dropna()

# modify the datat to work with this model
x = decision_tree_df[['today','yesterday','before_yesterday']]
y = decision_tree_df['tomorrow']

clf = tree.DecisionTreeClassifier(criterion = 'entropy').fit(x,y)

# we can calculate the accuarcy using score
score = clf.score(x,y)
print(score)

from sklearn import metrics
y_pred = clf.predict(x)
cm     = metrics.confusion_matrix(y,y_pred)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt='.3f', linewidths=0.5, square=True,cmap='Blues_r')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

dotfile = open('dt.dot','w')
tree.export_graphviz(clf, out_file=dotfile)
dotfile.close()











