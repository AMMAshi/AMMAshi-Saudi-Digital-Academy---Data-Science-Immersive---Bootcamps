# ====================================================
# Arwa Ashi - HomeWork 2 - Week 9 - Nov 2, 2020
# ====================================================
# Saudi Digital Academy
# Creating my own project visualization!
# ====================================================
# Packages:
# ----------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project Guideline: 
# ----------------------------------------------------
# Data:
# COVID-19's Impact on Airport Traffic
# https://www.kaggle.com/terenceshin/covid19s-impact-on-airport-traffic

# Metadata:
# #   Column              Non-Null Count  Dtype  
# --- -----------------   --------------  -----  
# 0   AggregationMethod   5936 non-null   object 
# 1   Date                5936 non-null   object 
# 2   Version             5936 non-null   float64
# 3   AirportName         5936 non-null   object 
# 4   PercentOfBaseline   5936 non-null   int64  
# 5   Centroid            5936 non-null   object 
# 6   City                5936 non-null   object 
# 7   State               5936 non-null   object 
# 8   ISO_3166_2          5936 non-null   object 
# 9   Country             5936 non-null   object 
# 10  Geography           5936 non-null   object

# Objective:
# 1- undrestading COVID-19's Impact on Airport Traffic over time.
# 2- undrestading the relation between the variables. 

# Code:
# ----------------------------------------------------
# 1-  undrestanding the data
df = pd.read_csv('covid_impact_on_airport_traffic.csv')#,index_col="Date",parse_dates=True)
# print(df.head())
# print(df.info())
# print(df.describe())
'''
       Version  PercentOfBaseline
count   5936.0        5936.000000
mean       1.0          65.581705
std        0.0          21.985070
min        1.0           0.000000
25%        1.0          52.000000
50%        1.0          66.000000
75%        1.0          82.000000
max        1.0         100.000000
'''
# print(df.shape)

# Fix date and set it as index
df['Date'] = pd.to_datetime(df['Date']).dt.date
df         = df.set_index('Date')
# print(df.head())
# print(df.set_index('Date'))

# 2- visualization
# line plot
plt.figure(figsize=(16,6))
plt.title('Monthly Percent of Baseline in 2020')
sns.lineplot(data=df['PercentOfBaseline'])
plt.show()

# scatter plot
plt.figure(figsize=(16,6))
plt.title('Monthly Percent of Baseline in 2020')
sns.scatterplot(x = df.index, y = df['PercentOfBaseline'])
plt.show()

# Color-Coded Scatter Plots by Country
plt.figure(figsize=(16,6))
plt.title('Monthly Percent of Baseline in 2020 by Country')
sns.scatterplot(x = df.index, y = df['PercentOfBaseline'], hue=df['Country'])
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# Color-Coded Scatter Plots by AirportName
plt.figure(figsize=(16,6))
plt.title('Monthly Percent of Baseline in 2020 by AirportName by AirportName')
sns.scatterplot(x = df.index, y = df['PercentOfBaseline'], hue=df['AirportName'])
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# Color-Coded Scatter Plots by City
plt.figure(figsize=(16,6))
plt.title('Monthly Percent of flight Baseline in 2020 by City')
sns.scatterplot(x = df.index, y = df['PercentOfBaseline'], hue=df['City'])
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# Color-Coded Scatter Plots by State
plt.figure(figsize=(16,6))
plt.title('Monthly Percent of Baseline in 2020 by State')
sns.scatterplot(x = df.index, y = df['PercentOfBaseline'], hue=df['State'])
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# Swarm Plots by Country
plt.figure(figsize=(16,6))
plt.title('Percent of Baseline in 2020 by Country')
sns.swarmplot(x = df['Country'], y = df['PercentOfBaseline'])
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# Swarm Plots by AirportName
plt.figure(figsize=(35,6))
plt.title('Percent of Baseline in 2020 by AirportName')
sns.swarmplot(x = df['AirportName'], y = df['PercentOfBaseline'])
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# Histograms
sns.displot(df['PercentOfBaseline'],kde=False)
plt.title('Percent of Baseline in 2020')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# Density plots
sns.kdeplot(df['PercentOfBaseline'],shade=True)
plt.title('Percent of Baseline in 2020')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# Color-coded plots
# creating df for variables 
Australia_data = df[df.Country == 'Australia']
Canada_data    = df[df.Country == 'Canada']
Chile_data     = df[df.Country == 'Chile']
USA_data       = df[df.Country == 'United States of America (the)']

# Histograms for each species
sns.distplot(a=Australia_data['PercentOfBaseline'], label='Australia', kde=False)
sns.distplot(a=Canada_data['PercentOfBaseline']   , label='Canada'   , kde=False)
sns.distplot(a=Chile_data['PercentOfBaseline']    , label='Chile'    , kde=False)
sns.distplot(a=USA_data['PercentOfBaseline']      , label='United States of America', kde=False)
# Add title
plt.title("Histogram of Percent of Baseline, by Country")
# Force legend to appear
plt.legend()
plt.show()

# Kde plot for each species
sns.kdeplot(data=Australia_data['PercentOfBaseline'], label='Australia', shade=True)
sns.kdeplot(data=Canada_data['PercentOfBaseline']   , label='Canada'   , shade=True)
sns.kdeplot(data=Chile_data['PercentOfBaseline']    , label='Chile'    , shade=True)
sns.kdeplot(data=USA_data['PercentOfBaseline']      , label='United States of America', shade=True)
# Add title
plt.title("Histogram of Percent of Baseline, by Country")
plt.legend()
plt.show()



