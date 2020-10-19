#============================================
# Arwa Ashi - HW 2 - Week 7 - Oct 19, 2020
#============================================

# random.random() returns a random floating number between 0.000 and 1.000
# random.random() * 50 returns a random floating number between 0.000 and 50.000, i.e. scaling the range of random numbers.
# random.random() * 25 + 10 returns a random floating number between 10.000 and 35.000,
# round(num) returns the rounded integer value of num
# ==================================================
# 1- Complete the randInt function
import random

def randInt(min= 0 ,max= 0):
    if max == 0 and min == 0:
        num = random.random() * 100
    elif max != 0 and min == 0:
        if max > 0:
            num = random.random() * max
        else:
            return "ERROR: please enter a +ve value !"
    elif min != 0 and max == 0:
        if min > 0:
            num = random.random() * (100 - min) + min
        else:
            return "ERROR: please enter a +ve value !" 
    elif min !=0 and max !=0:
        if max > min:
            num = random.random() * (max - min) + min
        else:
            return "ERROR: min is greater than max, please reorder them !"
    return round(num)

# If no arguments are provided, the function should return a random integer between 0 and 100.
test_randInt_01 = randInt()
print(test_randInt_01) 		    

# If only a max number is provided,
# the function should return a random integer between 0 and the max number.
test_randInt_02 = randInt(max=50)
print(test_randInt_02) 	            

# If only a min number is provided,
# the function should return a random integer between the min number and 100
test_randInt_03 = randInt(min=50)
print(test_randInt_03) 	            

# If both a min and max number are provided,
# the function should return a random integer between those 2 values
test_randInt_04 = randInt(min=50, max=500)
print(test_randInt_04)     

# 2- Account for any edge cases (e.g. min is greater than max, max is less than 0)
# min is greater than max
test_randInt_edge_01 = randInt(min=10, max=5)
print(test_randInt_edge_01)

# max is less than 0
test_randInt_edge_02 = randInt(max=-9)
print(test_randInt_edge_02)

test_randInt_edge_03 = randInt(min=-9)
print(test_randInt_edge_03)



