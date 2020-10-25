# ============================================
# Arwa Ashi - HW 1 - Week 8 - Oct 25, 2020
# ============================================
import pandas as pd

dist = {"country"   :["Barzil"  ,"Russia","India"    ,"China"  ,"South Africa"],
        "capital"   :["Brasilia","Moscow","New Dehli","Beijing","Pretoria"],
        "area"      :[8.516     , 17.10  , 3.286     , 9.597   , 1.221],
        "population":[200.4     , 143.5  , 1252      , 1357    , 52.98]}
# print(dist)

brics = pd.DataFrame(dist)
#print(brics)

# Set the index for brics
brics.index = ["BR","RU","IN","CH","SA"]

# Print out brics with new index values
# print(brics)


# Another way to create a DataFrame is by importing a csv file using Pandas.
cars = pd.read_csv('cars.csv', index_col = 0)
# print(cars)

# Print out country column as Pandas Series
# print(cars['cars_per_cap'])

# Print out country column as Pandas DataFrame
# print(cars[['cars_per_cap']])

# Print out DataFrame with country and drives_right columns
# print(cars[['cars_per_cap','country']])

# Print out first 4 observations
# print(cars[0:4])

# Print out fifth and sixth observations
# print(cars[4:6])

# Print out observation for Japan
# print(cars.iloc[2])

# Print out observations for Australia and Egypt
# print(cars.loc[['AUS','EG']])


