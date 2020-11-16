# Nov 11, 2020
# =================================================================================
# Arwa Ashi - HW4 - Week 10 - Nov 11, 2020
# Saudi Digital Academy
# Machine Learning
# =================================================================================

# Task 1: Introduction and Import Libraries
# -------------------------------------------------------------------
from __future__ import print_function
# %matplotlib inline
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd
import pandas_profiling
# plt.style.use("ggplot")
# warnings.simplefilter("ignore")

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg
from pandas_profiling import ProfileReport


# Task 2: Exploratory Data Analysis
# -------------------------------------------------------------------
hr = pd.read_csv('https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/daily_materials/tree_forest/data/employee_data.csv')
# print(hr.head())

# hr.profile_report(title="Employee Data")
profile = ProfileReport(hr, title='Employee Data', minimal=True)
# profile.to_file("output_01.html")

'''
pd.crosstab(hr.salary, hr.quit).plot(kind="bar")
pd.title=("Frequency of Turnover based on Salary")
pd.xlabel = ('Salary')
pd.ylabel = ('Freq of Turnover')
plt.show()

# low and mid salary turnover more or quit more
pd.crosstab(hr.department, hr.quit).plot(kind="bar")
pd.title=("Frequency of Turnover based on Dep")
pd.xlabel = ('Dep')
pd.ylabel = ('Freq of Turnover')
plt.show()
'''


# Task 3: Encode Categorical Features
# -------------------------------------------------------------------
cat_vars = ['department','salary']
for var in cat_vars:
    cat_list = pd.get_dummies(hr[var],prefix=var)
    hr = hr.join(cat_list)

#print(cat_vars)
#print(hr.head())
#print(hr.info())

# removing extra columns
hr.drop(columns = ['department','salary'], axis=1, inplace=True)
#print(hr.head())
#print(hr.info())


# Task 4: Visualize Class Imbalance
# -------------------------------------------------------------------
'''
from yellowbrick.target import ClassBalance
plt.style.use("ggplot")
#plt.rcParams['figure.figsize'] = (12,9)

visualizer = ClassBalance(labels = ('stayed','quit')).fit(hr.quit)
visualizer.show()
'''
# sample grapping from the same class
# breaking out that data
# class stayed and quit. 


# Task 5: Create Training and Validation Sets
# -------------------------------------------------------------------
X = hr.loc[:,hr.columns != 'quit']
y = hr.quit

# breaking the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2,stratify=y)


# Tasks 6 & 7: Build a Decision Tree Classifier with Interactive Controls
# -------------------------------------------------------------------
# using the decision tree simplfy the code for the audince
# recroseive remailing analytic (mathematics !!!)
# low baise and high variance
# we will solve the high variance in problem by using random forest classifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg

# False when i relase the maase pottton untill we let go this bottom allow to save time and 
                 
# @interact
def plot_tree(crit=['gini','entropy'],
              bootstrap=['best','random'],
              depth=IntSlider(min=1,max=30,value=2, continuous_update=False),
              min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),
              min_leaf=IntSlider(min=1,max=5,value=2, continuous_update=False)):
    estimator = DecisionTreeClassifier(random_state = 0,
                                       #out_file = None,
                                       criterion=crit,
                                       splitter=split,
                                       max_depth = depth,
                                       min_samples_split = min_split     ,
                                       min_samples_leaf = min_leaf
                                       
        )
    estimator.fit(X_train, y_train)
    print('{:.3f}'.format(accuracy_score(y_train, estimator.predict(X_train))))
    print('{:.3f}'.format(accuracy_score(y_train, estimator.predict(X_train))))

    graph = Source(tree.export_graphviz(estimator,
                                       out_file = None,
                                       feature_names = X_train.columns,
                                       class_names   = ['stayed','quit'],
                                       filled=True))

    # pipe this data into a piture
    # display(Image(data=graph.pipe(foremat='pdf')))

# Task 8: Build a Random Forest Classifier with Interactive Controls
# -------------------------------------------------------------------

print('\n# -------------------------------------------------------------------#\n')

# @interact
def plot_tree_rf(crit=['gini','entropy'],
                 bootstrap= ['True','False'],
                 depth=IntSlider(min=1,max=7,value=2, continuous_update=False), # False when i relase the maase pottton untill we let go this bottom allow to save time and 
                 forests=IntSlider(min=1,max=200,value=100,continuous_update=False),
                 min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),
                 min_leaf=IntSlider(min=1,max=5,value=2, continuous_update=False)):
    estimator = RandomForestClassifier(random_state      = 0,
                                       criterion         = crit,
                                       bootstrap         = bootstrap,
                                       max_depth         = depth,
                                       min_samples_split = min_split     ,
                                       min_samples_leaf  = min_leaf,
                                       n_jobs            = 1,
                                       verbose           = False ).fit(X_train,y_train)
    print('Random Forest Traing Acc: {:.3f}'.format(accuracy_score(y_train, estimator.predict(X_train))))
    print('Random Forest Test Acc: {:.3f}'.format(accuracy_score(y_train, estimator.predict(X_train))))

    num_tree = estimator.estimators_[1] # grab one tree and we can change the number
    print("Visuallising Tree", 0 )

    graph = Source(tree.export_graphviz(#estimator,
                                       num_tree,
                                       out_file      = None,
                                       feature_names = X_train.columns,
                                       class_names   = ['stayed','quit'],
                                       filled        = True))

    # pipe this data into a piture
    # display(Image(data=graph.pipe(foremat='png')))
    
# Task 9: Feature Importance Plots and Evaluation Metrics
# -------------------------------------------------------------------
from yellowbrick.model_selection import FeatureImportances
plt.rcParams['figure.figsize'] = (12,8)
plt.style.use("ggplot")

rf = RandomForestClassifier(bootstrap='True', 
                            max_depth=2, 
                            n_jobs=1, 
                            random_state=0,
                            verbose=False)
viz = FeatureImportances(rf)
viz.fit(X_train, y_train)
viz.show();

dt = DecisionTreeClassifier(max_depth=5, random_state=0)
viz = FeatureImportances(dt)
viz.fit(X_train, y_train)
viz.show()




