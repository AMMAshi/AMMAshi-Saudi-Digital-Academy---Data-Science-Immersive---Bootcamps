# =================================================================================
# Arwa Ashi - HW1 - Week 10
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Heuristic Models (Cost Function Extension)
# Look at the Seattle weather in the data folder. Come up with a heuristic model
# to predict if it will rain today. Keep in mind this is a time series, which means
# that you only know what happened historically (before a given date). One example
# of a heuristic model is: It will rain tomorrow if it rained more than 1 inch (>1.0 PRCP)
# today. Describe your heuristic model in the next cell.
# =================================================================================
# your model here
# Examples:
# If rained yesterday it will rain today.
# If it rained yesterday or the day before it will rain today.
# here is an example of how to build and populate a hurestic model
# =================================================================================

# Packages
# ---------------------------------------------------------------------------------
import pandas as pd

# Data
# ---------------------------------------------------------------------------------
df = pd.read_csv('https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/module_4_ML/data/seattle_weather_1948-2017.csv')

# Fix date and set it as index
df['DATE'] = pd.to_datetime(df['DATE']).dt.date

# print(df)
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

# can be as large as 25549
numrows = 25549 


# Create an empty dataframe to hold 100 values
# ---------------------------------------------------------------------------------
heuristic_df = pd.DataFrame({'yesterday'     :[0.0]*numrows,
                             'today'         :[0.0]*numrows,
                             'tomorrow'      :[0.0]*numrows,
                             'guess'         :[False]*numrows,  #logical guess
                             'rain_tomorrow' :[False]*numrows,  #historical observation
                             'correct'       :[False]*numrows,  #TRUE if your guess matches the historical observation
                             'true_positive' :[False]*numrows,  #TRUE If you said it would rain and it did
                             'false_positive':[False]*numrows,  #TRUE If you sait id would rain and it didn't
                             'true_negative' :[False]*numrows,  #TRUE if you said it wouldn't rain and it didn't
                             'false_negative':[False]*numrows}) #TRUE if you said it wouldn't raing and it did
# print(heuristic_df)


# Sort columns for convience
# ---------------------------------------------------------------------------------
seq = ['yesterday'     ,
       'today'         ,
       'tomorrow'      ,
       'guess'         ,
       'rain_tomorrow' ,
       'correct'       ,
       'true_positive' ,
       'false_positive',
       'true_negative' ,
       'false_negative']

heuristic_df = heuristic_df.reindex(columns=seq)

# print(df.head())
# print(heuristic_df.head())


# Build a loop to add your heuristic model guesses as a column to this dataframe
# ---------------------------------------------------------------------------------
# Here is an example loop that populates the dataframe created earlier
# with the total percip from yesterday and today
# then the guess is set to true if rained both yesterday and today
# ---------------------------------------------------------------------------------
for z in range(numrows):
    # start at time 2 in the data frame
    i = z + 2
    # pull values from the dataframe
    yesterday     = df.iloc[(i-2),1]
    today         = df.iloc[(i-1),1]
    tomorrow      = df.iloc[i,1]
    rain_tomorrow = df.iloc[(i),1]
    
    heuristic_df.iat[z,0] = yesterday
    heuristic_df.iat[z,1] = today
    heuristic_df.iat[z,2] = tomorrow
    heuristic_df.iat[z,3] = False         # set guess default to False
    heuristic_df.iat[z,4] = rain_tomorrow
    
    # example hueristic
    if today > 0.0 and yesterday > 0.0:
        heuristic_df.iat[z,3] = True
        
    if heuristic_df.iat[z,3] == heuristic_df.iat[z,4]:
        heuristic_df.iat[z,5] = True
        if heuristic_df.iat[z,3] == True:
            heuristic_df.iat[z,6] = True #true positive
        else:
            heuristic_df.iat[z,8] = True #true negative
    else:
        heuristic_df.iat[z,5] = False
        if heuristic_df.iat[z,3] == True:
            heuristic_df.iat[z,7] = True #false positive
        else:
            heuristic_df.iat[z,9] = True #false negative

# print(heuristic_df.head())
print(heuristic_df.info())

# Evaluate the performance of the Heuristic model
# ---------------------------------------------------------------------------------
# split data into training and testing
# ---------------------------------------------------------------------------------
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import statistics as stat
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# df.dropna()
heuristic_df = heuristic_df.dropna()

# enter split function here to make h_train and h_test subsets of the data
h_train, h_test = train_test_split(heuristic_df,test_size = 0.31)

print(h_train)
print(h_test)

print(h_train.shape) 
print(h_test.shape)  

# The accuracy of your predicitions
# ---------------------------------------------------------------------------------
# we used this simple approach in the first part to see what percent of the time we where correct 
# calculated as (true positive + true negative)/ number of guesses
print(heuristic_df['correct'].value_counts()/numrows)
print((sum(heuristic_df.true_positive)+sum(heuristic_df.true_negative))/len(heuristic_df))

# The precision of your predicitions
# ---------------------------------------------------------------------------------
# precision is the percent of your postive prediction which are correct
# more specifically it is calculated (num true positive)/(num tru positive + num false positive)
print(sum(heuristic_df.true_positive)/(sum(heuristic_df.true_positive)+sum(heuristic_df.false_positive)))

# the recall of your predicitions
# ---------------------------------------------------------------------------------
# recall the percent of the time you are correct when you predict positive
# more specifically it is calculated (num true positive)/(num tru positive + num false negative)
print(sum(heuristic_df.true_positive)/(sum(heuristic_df.true_positive)+sum(heuristic_df.true_negative)))

# The sum of squared error (SSE) of your predictions
# ---------------------------------------------------------------------------------
# SSE = sum(actual - predicted^2)
SSE = sum(heuristic_df['correct'])-np.square(sum(heuristic_df['guess']))
print(SSE)

SST = sum(heuristic_df.guess) - np.square(sum(heuristic_df.guess))
SSR = sum(heuristic_df.correct) - np.square(sum(heuristic_df.guess))
SSE = SST - SSR 
print(SSE)


