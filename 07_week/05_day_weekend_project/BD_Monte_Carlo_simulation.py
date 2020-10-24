# ==================================================================================
# Arwa Ashi - Weekend Project - Week 7 - Oct 22, 2020
# Saudi Digital Academy 2020
# ==================================================================================
# Project Question:
# There is a famous problem in statistics that concerns
# a room full of people: Same Birthday!

# An instructor offers a prize of $20.00 to anyone who
# thinks that two people in the room have the same birthday.

# Your assignment is to build a Monte Carlo simulation to tell
# the instructor how many people need to be in the room to give
# him/her a better than 50% chance of winning the $20.

# That is to say how many people need to be in a room in order
# for the probability of two of them having the same birthday is
# greater than 50%. Do the same for 95% and 99%.

# Build a solution to the birthday problem using Monte Carlo.

# ----------------------------------------------------------------------------------
# Answer:
# Assuming two people are having the same Birthday BD but not the same age, the year
# will be ignored.

# Libraries:
import numpy as np
from scipy import stats          # undrestand the statistic
import matplotlib.pyplot as plt  # plot

# Mathematically -------------------------------------------------------------------
# P (at least two people in a group of n people share a birthday) =
# P = (365/365) * ((365-1)/365) * ((365-2)/365) * ... * ((365-n)/365)
n                = 100
x                = (365/365)
prob_graph_array = np.array([])
prob_graph_array = np.append(prob_graph_array,1-x)
for i in range(1,n+1,1):
    x                = x * ((365-i)/365)
    prob_graph_array = np.append(prob_graph_array,1-x)
print(f"P (at least two people in a group of {n} people share a birthday) =", 1 - x)
# print(prob_graph_array)

# Plotting the probability to understand the changes with increasing the people inside a room
z = np.linspace(1, n+1, n+1)
# print(z)

# Plotting the data
from scipy import interpolate

def refline(x, **kwargs):
        y = f(x)
        plt.plot([x, x, 0], [0, y, y], **kwargs)
        
f = interpolate.interp1d(z,prob_graph_array)
plt.scatter(z, prob_graph_array,color="c",s=5)
plt.plot(z, prob_graph_array,color="m")
refline(23, color="r", lw=0.5, dashes=[2, 2])
refline(60, color="r", lw=0.5, dashes=[2, 2])
refline(67, color="g", lw=0.5, dashes=[2, 2])
plt.xlabel('Total n people in a room')
plt.ylabel('Probability')
plt.title('The Probability of at least two people in a group \n of n people share a birthday')
plt.show()

# As a result: from the plot:
# if there are 23 people in a room, the probability of two of them having the same birthday is 50%.
# if there are 60 people in a room, the probability of two of them having the same birthday is 95%.
# if there are 67 people in a room, the probability of two of them having the same birthday is 99%. 

# Using Monte Carlo ----------------------------------------------------------------
def BD_Monte_carlo(n = int,trials = int):
    n       = n            # Total people
    trials  = trials       # Total trials
    match   = 0            # Total match between to people BD
    results = np.array([]) # Create array to hold simulation results
    for i in range(trials):
        track = {}
        for j in range(n):
            BD = np.random.randint(0,365)
            #print(BD)
            if BD in track:
                #print(BD)
                #print(track)
                match  += 1
                break
            track[BD]   = 1
    for i in range(match):
            results = np.append(results,1)
    for i in range(trials-match):
            results = np.append(results,0)
    print("match",match)
    #print(results)
    #print(len(results))
    print(stats.describe(results))

#  if there are 23 people in a room, the probability of two of them having the same birthday is 50%.
BD_Monte_carlo(n = 23,trials = 23)
'''
match 14
DescribeResult(nobs=23, minmax=(0.0, 1.0), mean=0.6086956521739131, variance=0.24901185770750986, skewness=-0.44543540318737423, kurtosis=-1.8015873015873007)
'''

# if there are 60 people in a room, the probability of two of them having the same birthday is 95%.
BD_Monte_carlo(n = 60,trials = 60)
'''
match 59
DescribeResult(nobs=60, minmax=(0.0, 1.0), mean=0.9833333333333333, variance=0.016666666666666666, skewness=-7.550956836887783, kurtosis=55.01694915254235)
'''

# if there are 67 people in a room, the probability of two of them having the same birthday is 99%. 
BD_Monte_carlo(n = 67,trials = 67)
'''
match 67
DescribeResult(nobs=67, minmax=(1.0, 1.0), mean=1.0, variance=0.0, skewness=0.0, kurtosis=-3.0)
'''
