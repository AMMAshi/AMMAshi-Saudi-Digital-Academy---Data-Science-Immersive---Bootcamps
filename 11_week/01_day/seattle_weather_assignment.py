# =================================================================================
# Arwa Ashi - HW3 - Week 10 - Nov 10, 2020
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Assignment ML3
# Use the same project from the previous assignment (the heuristic modeling) and
# build a function that takes a vector of predictions using your heuristic and
# a vector of realizations (the correct values) from the data set and calculate:
# 1- Precision
# 2- Recall  
# 3- SSE Cost of your prediction
# SSE is the sum of squared error (adding up the difference in your prediction and
# the actual value after you have squared each individual difference), you can find
# more about how to calculate it here.

# 1- Break the dataset into two parts, training and testing.
# 2- Use the first 80% of the dataset for training and the last 20% for testing.
# 3- Evaluate both sets of data using your function.
# 4- What difference do you see in the calculated values (Precision and Recall)?
# 5- Submit your notebook.
# =================================================================================

# Packages
# ---------------------------------------------------------------------------------
import pandas as pd

# Data
# ---------------------------------------------------------------------------------
df = pd.read_csv('https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/module_4_ML/data/seattle_weather_1948-2017.csv')
# Fix date and set it as index
df['DATE'] = pd.to_datetime(df['DATE']).dt.date

# print(df.head())
# print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25551 entries, 0 to 25550
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   DATE    25551 non-null  object 
 1   PRCP    25548 non-null  float64
 2   TMAX    25551 non-null  int64  
 3   TMIN    25551 non-null  int64  
 4   RAIN    25548 non-null  object 
dtypes: float64(1), int64(2), object(2)
memory usage: 998.2+ KB
None
'''
numrows = 25549 # can be as large as 25549

# Create an empty dataframe to hold 100 values
# ---------------------------------------------------------------------------------
heuristic_df = pd.DataFrame({'yesterday'    :[0.0]*numrows,
                             'today'        :[0.0]*numrows,
                             'tomorrow'     :[0.0]*numrows,
                             'guess'        :[False]*numrows,  #logical guess
                             'rain_tomorrow':[False]*numrows,  #historical observation
                             'correct'      :[False]*numrows}) #TRUE if your guess matches the historical observation

# Sort columns for convience
# ---------------------------------------------------------------------------------
seq = ['yesterday'    ,
       'today'        ,
       'tomorrow'     ,
       'guess'        ,
       'rain_tomorrow',
       'correct']
heuristic_df = heuristic_df.reindex(columns=seq)

# print(df.head())
# print(heuristic_df.head())

# Build a loop to add your heuristic model guesses as a column to this dataframe
# ---------------------------------------------------------------------------------
# here is an example loop that populates the dataframe created earlier
# with the total percip from yesterday and today then the guess is set
# to true if rained both yesterday and today
# ---------------------------------------------------------------------------------
for z in range(numrows):
    #start at time 2 in the data frame
    i = z + 2
    #pull values from the dataframe
    yesterday = df.iloc[(i-2),1]
    today = df.iloc[(i-1),1]
    tomorrow = df.iloc[i,1]
    rain_tomorrow = df.iloc[(i),1]
    
    heuristic_df.iat[z,0] = yesterday
    heuristic_df.iat[z,1] = today
    heuristic_df.iat[z,2] = tomorrow
    heuristic_df.iat[z,3] = False # set guess default to False
    heuristic_df.iat[z,4] = rain_tomorrow
    
    ######### uncomment and create your heuristic guess ################
    #if ##### your conditions here #########:
    #    heuristic_df.iat[z,3] = True 
    ####################################################################
    if heuristic_df.iat[z,0]>0 and heuristic_df.iat[z,1]>0:
        heuristic_df.iat[z,3] = True
    
    if heuristic_df.iat[z,3] == heuristic_df.iat[z,4]:
        heuristic_df.iat[z,5] = True
    else:
        heuristic_df.iat[z,5] = False

# Evaluate the performance of the Heuristic model
# ---------------------------------------------------------------------------------
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import statistics as stat
import numpy as np 
import warnings

# The accuracy of your predicitions
# ---------------------------------------------------------------------------------
# we used this simple approach in the first part to see what percent of the time we where correct 
# calculated as (true positive + true negative)/ number of guesses
print(heuristic_df['correct'].value_counts()/numrows)

# The sum of squared error (SSE) of your predictions
# ---------------------------------------------------------------------------------
# SSE = sum((actual - mean(predicted))^2)
# https://www.wikihow.com/Calculate-the-Sum-of-Squares-for-Error-(SSE)
SSE = sum(np.square(heuristic_df['correct']-np.mean(heuristic_df['correct'])))
print('-------------------------------\nSSE=',SSE)


#  Regression
# --------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# df.dropna()
heuristic_df = heuristic_df.dropna()

X = heuristic_df[['yesterday','today','tomorrow','guess','rain_tomorrow']]
y = heuristic_df['correct']

print(X.shape)
print(y.shape)

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)


print('Coefficients: \n', reg.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
