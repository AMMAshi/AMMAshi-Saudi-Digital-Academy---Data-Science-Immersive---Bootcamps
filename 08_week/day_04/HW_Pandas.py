# ======================================================================================
# Arwa Ashi - HW 4 - Week 8 - Oct 28, 2020
# ======================================================================================
# In each cell complete the task using pandas
# data libraries
import pandas as pd
import numpy as np


# Q1 - Read in the titanic.csv file in the ~/data directory as a pandas dataframe called df
# --------------------------------------------------------------------------------------
# Answer
url_01     = 'https://raw.githubusercontent.com/daniel-dc-cd/data_science/master/module_3_Python/data/titanic.csv'
titanic_df = pd.read_csv(url_01)
# print(titanic_df)
# print(titanic_df.loc[titanic_df.Survived.isnull()].count())
# print(titanic_df.loc[titanic_df.Survived.notnull()].count())


# Q2 - Display the head of the dataframe
# --------------------------------------------------------------------------------------
# Answer
# print(titanic_df.head())


# Q3 - What is the percentage of people who survived? (hint find the mean of the survival column)
# --------------------------------------------------------------------------------------
# Answer
# print(titanic_df.info())
# print(titanic_df['Survived'].count())                   # total passenger = 891
# print(titanic_df.loc[titanic_df.Survived == 1].count()) # total survival  = 342
# print(342/891)                                          # 0.3838383838383838
# The percentage of people who survived is 38.38% 


# Q4 - How many women and how many men survived?
# --------------------------------------------------------------------------------------
# Answer
# print(titanic_df.loc[titanic_df.Survived == 1].groupby(['Sex','Survived']).Survived.agg([len]))
'''
                 len
Sex    Survived     
female 1         233
male   1         109
'''


# Q5 - What is the percentage of people that survied who paid a fare less than 10?
# --------------------------------------------------------------------------------------
# Answer
# print(titanic_df['Survived'].count())                                                 # total passenger = 891
# print(titanic_df.loc[titanic_df.Fare < 10].groupby(['Survived']).Survived.agg([len])) # total survival  = 67
# print(67/891)                                                                         # 0.07519640852974187
# The percentage of people that survied who paid a fare less than 10, is 7.51%


# Q6 - What is the average age of those who didn't survive?
# --------------------------------------------------------------------------------------
# Answer
# print(titanic_df.loc[titanic_df.Survived == 0].Age.dropna().mean())
# The average age of those who didn't survive, is 30.62 years old.


# Q7 - What is the average age of those who did survive?
# --------------------------------------------------------------------------------------
# Answer
# print(titanic_df.loc[titanic_df.Survived == 1].Age.dropna().mean())
# The average age of those who didn't survive, is 28.34 years old.


# Q8 - What is the average age of those who did and didn't survive grouped by gender?
# --------------------------------------------------------------------------------------
# Answer
# print(titanic_df.groupby(['Sex','Survived']).Age.mean())
'''
Sex     Survived
female  0           25.046875
        1           28.847716
male    0           31.618056
        1           27.276022
Name: Age, dtype: float64
'''


# Q9 - Tidy GDP
# Manipulate the GDP.csv file and make it tidy, the result should be a pandas dataframe with
# the following columns:
# Country Name
# Country Code
# Year
# GDP
# --------------------------------------------------------------------------------------
# Answer
GDP_df = pd.read_csv("GDP.csv")
years  = [str(i) for i in range(1960,2018,1)]
# print(years)
# print(pd.melt(GDP_df,id_vars=['Country Name','Country Code'],
#              value_vars= years ,var_name='Year',
#              value_name='GDP'))
'''
       Country Name Country Code  Year           GDP
0             Aruba          ABW  1960           NaN
1       Afghanistan          AFG  1960  5.377778e+08
2            Angola          AGO  1960           NaN
3           Albania          ALB  1960           NaN
4           Andorra          AND  1960           NaN
...             ...          ...   ...           ...
15307        Kosovo          XKX  2017  7.128691e+09
15308   Yemen, Rep.          YEM  2017           NaN
15309  South Africa          ZAF  2017  3.494190e+11
15310        Zambia          ZMB  2017  2.580867e+10
15311      Zimbabwe          ZWE  2017  1.784582e+10

[15312 rows x 4 columns]
'''
