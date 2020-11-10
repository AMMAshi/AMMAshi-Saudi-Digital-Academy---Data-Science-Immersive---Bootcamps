# ==================================================================================================
# Arwa Ashi - HW2 - Week 10
# Saudi Digital Academy
# Machine Learning
# ==================================================================================================
# Assignment ML2
# Work through the notebook in the ML section of the Data Science Handbook with Python.
# 05.01-What-Is-Machine-Learning.ipynb
# https://github.com/daniel-dc-cd/data_science/blob/master/daily_materials/ml_kmeans_nb_regression/Assignment_11_9_2020.ipynb
# Then complete the stocks project.
# ==================================================================================================
# In this notebook we will be looking at financial forecasting with machine learning

# 1. Preparing our tools.
# --------------------------------------------------------------------------------------------------

### Packages
# --------------------------------------------------------------------------------------------------
# 1- numpy for rapid numerical calculations with fast vectorized C implementations
# 2- pandas for processing data
# 3- matplotlib and Seaborn for visualizing charts
# 4- scikit-learn (imported as sklearn) is the de facto standard machine learning library
# in the pydata ecosystem.
# 5- pandas_profiling which is a newer convenience package that helps by putting together
# much of our initial boilerplate exploratory data analysis code.

# Bring our tools in:
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory

from pandas_profiling import ProfileReport
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit

### Importing and describing the data
# --------------------------------------------------------------------------------------------------
# my data
stocks = pd.read_csv('stocks.csv')

# Kaggle data
# stocks = pd.read_csv('prices.csv')

# print(stocks.info())
# print(stocks.head())

# Fix date and set it as index
stocks['date'] = pd.to_datetime(stocks['date']).dt.date
# print(stocks.info())
# print(stocks.head())

# Fix volume as floats64
stocks['volume']=stocks['volume'].astype(float)
# print(stocks.info())
# print(stocks.head())

# Set Date as index
stocks.set_index('date',inplace=True)
# print(stocks.info())
# print(stocks.head())
# The string column symbol is the trading symbol, also known in finance as the ticker.
'''
<class 'pandas.core.frame.DataFrame'>
Index: 13792 entries, 2020-08-31 to 2009-08-01
Data columns (total 7 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   symbol  13792 non-null  object 
 1   open    13792 non-null  float64
 2   close   13792 non-null  float64
 3   low     13792 non-null  float64
 4   high    13792 non-null  float64
 5   volume  13792 non-null  float64
 6   change  13792 non-null  float64
dtypes: float64(6), object(1)
memory usage: 862.0+ KB
None
'''
# print(stocks.describe())


### Pandas Profiling
# --------------------------------------------------------------------------------------------------
# Minimal avoids expensive calculations that won't have much insight for us and are slow
profile = ProfileReport(stocks, title='Stocks Profiling Report', minimal=True)
# profile.to_widgets()
# profile.to_file("output_01.html")
profile_02 = stocks.profile_report(title='Pandas Profiling Report', plot={'histogram': {'bins': 8}})
# profile_02.to_file("output_02.html")


# 2. Exploring, cleaning and visualizing the data
# --------------------------------------------------------------------------------------------------

plt.figure(figsize=(12,6))
sns.lineplot(data=stocks, x=stocks.index, y="close",hue="symbol")
plt.title('Historical stock Prices')
plt.legend(loc=2, bbox_to_anchor=(1,1))
plt.show()

'''
# PCLN is stocks for prices.csv file
def plot_tick(data, symbol):
    df  = data[data["symbol"]==symbol].reset_index(drop=True)
    fig = plotly.figure_factory.create_candlestick(df.open, df.high, df.low, 
                                                   df.close, dates=df.index)
    fig.show()

plot_tick(stocks, 'PCLN')
'''

# STC is Saudi stocks for stocks.csv file
def plot_tick(data, symbol):
    df  = data[data["symbol"]==symbol].reset_index(drop=True)
    fig = plotly.figure_factory.create_candlestick(df.open, df.high, df.low, 
                                                   df.close, dates=df.index)
    fig.show()

plot_tick(stocks, 'STC')


# 3. Feature engineering (transform returns with logarithms based on financial research)
# --------------------------------------------------------------------------------------------------
# - For modelling data with machine learning, it is helpful to transform the data into a
# form that is closer to the theoretical expectations where the ML models should perform well.

# - Let's transform the data into returns and generate other features. We will transform returns
# with logarithms based on financial research that log returns are closer to normally distributed
# and (statistically) stable.

# - The function below is just a sample of feature transformations. Taking the logarithms can help
# deal with skewed data as we saw we have in the pandas-profile report.

# - To be honest, which variables you use and how you transform them is largely dependent on domain
# expertise and traditions of the field. It can also be a matter of trial and error, although that can
# lead to overfitting. We will discuss overfitting a little bit later.

def feature_target_generation(df):
    """
    df: a pandas dataframe containing numerical columns
    num_days_ahead: an integer that can be used to shift the prediction value from the future into a prior row.
    """

    # The following line ensures the data is in date order    
    features = pd.DataFrame(index=df.index).sort_index() 
    features['f01'] = np.log(df.close / df.open)          # intra-day log return
    features['f02'] = np.log(df.open / df.close.shift(1)) # overnight log return

    features['f03'] = df.volume                           # try both regular and log volume
    features['f04'] = np.log(df.volume) 
    features['f05'] = df.volume.diff()                    # 1-day absolute change in volume
    features['f06'] = df.volume.pct_change()              # 1-day relative change in volume

    # The following are rolling averages of different periods
    features['f07'] = df.volume.rolling(5, min_periods=1).mean().apply(np.log)
    features['f08'] = df.volume.rolling(10, min_periods=1).mean().apply(np.log)
    features['f09'] = df.volume.rolling(30, min_periods=1).mean().apply(np.log)

    # More of our original data: low, high and close
    features['f10'] = df.low 
    features['f11'] = df.high
    features['f12'] = df.close

    # The Intraday trading spread measures how far apart the high and low are
    features['f13'] = df.high - df.low 

    # These are log returns over different time periods 
    features['f14'] = np.log(df.close / df.close.shift(1))  # 1 day log return
    features['f15'] = np.log(df.close / df.close.shift(5))  # 5 day log return
    features['f16'] = np.log(df.close / df.close.shift(10)) # 10 day log return

    return features


# 4. Features Dataframe
# ---------------------------------------------------------------------------------
# Let's generate a list of tickers so we can easily select them
ticker_list = stocks.symbol.unique()

# MSFT is stocks for prices.csv file
# These are hyperparameters you can play with or tune
# prediction_horizon = -5      # this is a negative number by convention
# ticker             = 'MSFT'  # choose any ticker
# n_splits           = 5

# STC is Saudi stocks for stocks.csv file
prediction_horizon = -5     # this is a negative number by convention
ticker             = 'STC'  # choose any ticker
n_splits           = 3

# Make an individual model for each ticker/symbol
features = feature_target_generation(stocks[stocks.symbol==ticker])
# print(features.head())
# print(features.info())

# 5. Preparing and splitting our data
# --------------------------------------------------------------------------------------------------
# - We are trying to predict the price prediction_horizon days in the future.  So we take the future
# value and move it prediction_horizon into the past to line up our data in the Scikit-learn format.  
y = features.f12.shift(prediction_horizon)

# The latest (prediction_horizon) rows will have nans because we have no future data, so let's drop them.
shifted = ~np.isnan(y)
X       = features[y.notna()] # Remove the rows that do not have valid target values
y       = y[shifted]          # Remove the rows that do not have valid target values

# Split the history into different backtesting regimes
tscv = TimeSeriesSplit(n_splits=n_splits)
# print(tscv)

# Review the features
# print(features.info())


# 6. Building our first model
# --------------------------------------------------------------------------------------------------
def model_ts_report(model, tscv, X, y, impute=False):
    """
    Fit the model and then run time series backtests.
    """
    # Loop through the backtests
    for train_ind, test_ind in tscv.split(X): 
        # Report on the time periods
        print(f'Train is from {X.iloc[train_ind].index.min()} to {X.iloc[train_ind].index.max()}. ')
        print(f'Test is from {X.iloc[test_ind].index.min()} to {X.iloc[test_ind].index.max()}. ')
        # Generate training and testing features and target for each fold.
        X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
        y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

        if impute==True:
            # Since linear regression cannot deal with NaN, we need to impute.  There may be the better choices.
            X_train.fillna(0, inplace=True)
            X_test.fillna(0, inplace=True)
        
        # Fit the model
        model.fit(X_train, y_train)

        # Predict and measure on the training data
        y_pred_train = model.predict(X_train) 
        print("Training results:")
        print("RMSE:", mean_squared_error(y_train, y_pred_train, squared=False))

        # Predict and measure on the testing data
        y_pred_test = model.predict(X_test)
        print("Test results:")
        print("RMSE:", mean_squared_error(y_test, y_pred_test, squared=False))
        print("")

from sklearn.linear_model import LinearRegression
# Fit and report on a linear model
lm = LinearRegression()
model_ts_report(lm, tscv, X, y, impute=True)


# Ensemble Model
# Initiate a Random Forest
rf = RandomForestRegressor()
model_ts_report(rf, tscv, X, y, impute=True) # Report on the random forest


# Support Vector Regressor
from sklearn.svm import SVR
svr = SVR()
model_ts_report(svr, tscv, X, y, impute=True)


# Extra Trees Regressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
extra_tree = ExtraTreeRegressor(random_state=0)
ETR        = BaggingRegressor(extra_tree, random_state=0)
model_ts_report(ETR, tscv, X, y, impute=True)


# ElasticNet
from sklearn.linear_model import ElasticNet
EN = ElasticNet(random_state=0)
model_ts_report(EN, tscv, X, y, impute=True)


# --------------------------------------------------------------------------------------------------
# My Feature engineering (transform returns with logarithms based on financial research)
# Adding change in % 
# --------------------------------------------------------------------------------------------------
def new_feature_target_generation(df):
    """
    df: a pandas dataframe containing numerical columns
    num_days_ahead: an integer that can be used to shift the prediction value from the future into a prior row.
    """

    # The following line ensures the data is in date order    
    features = pd.DataFrame(index=df.index).sort_index() 
    features['f01'] = np.log(df.close / df.open)          # intra-day log return
    features['f02'] = np.log(df.open / df.close.shift(1)) # overnight log return

    features['f03'] = df.volume                           # try both regular and log volume
    features['f04'] = np.log(df.volume) 
    features['f05'] = df.volume.diff()                    # 1-day absolute change in volume
    features['f06'] = df.volume.pct_change()              # 1-day relative change in volume

    # The following are rolling averages of different periods
    features['f07'] = df.volume.rolling(5, min_periods=1).mean().apply(np.log)
    features['f08'] = df.volume.rolling(10, min_periods=1).mean().apply(np.log)
    features['f09'] = df.volume.rolling(30, min_periods=1).mean().apply(np.log)

    # More of our original data: low, high and close
    features['f10'] = df.low 
    features['f11'] = df.high
    features['f12'] = df.close

    # The Intraday trading spread measures how far apart the high and low are
    features['f13'] = df.high - df.low 

    # These are log returns over different time periods 
    features['f14'] = np.log(df.close / df.close.shift(1))  # 1 day log return
    features['f15'] = np.log(df.close / df.close.shift(5))  # 5 day log return
    features['f16'] = np.log(df.close / df.close.shift(10)) # 10 day log return

    # change in percent
    features['f17'] = df.change 

    return features

# Make an individual model for each ticker/symbol
features = new_feature_target_generation(stocks[stocks.symbol==ticker])

# Preparing and splitting our data  
y = features.f12.shift(prediction_horizon)

# The latest (prediction_horizon) rows will have nans because we have no future data, so let's drop them.
shifted = ~np.isnan(y)
X       = features[y.notna()] # Remove the rows that do not have valid target values
y       = y[shifted]          # Remove the rows that do not have valid target values

# Split the history into different backtesting regimes
tscv = TimeSeriesSplit(n_splits=n_splits)
# print(tscv)

# Fit and report on a linear model
lm = LinearRegression()
model_ts_report(lm, tscv, X, y, impute=True)

# Ensemble Model
# Initiate a Random Forest
rf = RandomForestRegressor()
model_ts_report(rf, tscv, X, y, impute=True) # Report on the random forest














