# =================================================================================
# Arwa Ashi - HW1 - Week 10
# Saudi Digital Academy
# Machine Learning
# =================================================================================
# Heuristic Models
# Look at the Seattle weather in the data folder. Come up with a heuristic model
# to predict if it will rain today. Keep in mind this is a time series, which means
# that you only know what happened historically (before a given date). One example of
# a heuristic model is: It will rain tomorrow if it rained more than 1 inch (>1.0 PRCP)
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
# with the total percip from yesterday and today
# then the guess is set to true if rained both yesterday and today
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
# the accuracy of your predicitions
print(heuristic_df['correct'].value_counts()/numrows)

