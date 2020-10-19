#============================================
# Arwa Ashi - HW 2 - Week 7 - Oct 19, 2020
#============================================
from random import randint

# 1
# Biggie Size - Given a list, write a function that changes all positive numbers in the list to "big".
# Example: biggie_size([-1, 3, 5, -5]) returns that same list, but whose values are now [-1, "big", "big", -5]
def biggie_size(func_list):
    func_list = func_list
    for i in range(0,len(func_list),1):
        if func_list[i] > 0:
            func_list[i] = "big"
    return func_list
test_BSZ = biggie_size([-1, 3, 5, -5])
print(test_BSZ)

# 2
# Count Positives - Given a list of numbers, create a function to replace the last value with the number of positive values.
# (Note that zero is not considered to be a positive number).
# Example: count_positives([-1,1,1,1]) changes the original list to [-1,1,1,3] and returns it
# Example: count_positives([1,6,-4,-2,-7,-2]) changes the list to [1,6,-4,-2,-7,2] and returns it
def count_positives(func_list):
    func_list = func_list
    x         = randint(1,10)
    func_list.pop(int(len(func_list))-1)
    func_list.append(x)
    return func_list

test_CP_01 = count_positives([-1,1,1,1])
print(test_CP_01)

test_CP_02 = count_positives([1,6,-4,-2,-7,-2])
print(test_CP_02)      

# 3
# Sum Total - Create a function that takes a list and returns the sum of all the values in the array.
# Example: sum_total([1,2,3,4]) should return 10
# Example: sum_total([6,3,-2]) should return 7
def sum_total(func_list):
    func_list = func_list
    sumVal    = 0
    for i in range(0,len(func_list),1):
        sumVal = sumVal + func_list[i]
    return sumVal

test_ST_01 = sum_total([1,2,3,4])
print (test_ST_01)

test_ST_02 = sum_total([6,3,-2])
print (test_ST_02)
       
# 4
# Average - Create a function that takes a list and returns the average of all the values.
# Example: average([1,2,3,4]) should return 2.5
def average(func_list):
    func_list = func_list
    Ave       = 0
    sumVal    = 0
    for i in range(0,len(func_list),1):
        sumVal = sumVal + func_list[i]
    Ave = sumVal / int(len(func_list))
    return Ave

test_ave = average([1,2,3,4])
print(test_ave)

# 5
# Length - Create a function that takes a list and returns the length of the list.
# Example: length([37,2,1,-9]) should return 4
# Example: length([]) should return 0
def length(func_list):
    func_list = func_list
    return len(func_list)

test_list_length_01 = length([37,2,1,-9])
print(test_list_length_01)

test_list_length_02 = length([])
print(test_list_length_02)

# 6
# Minimum - Create a function that takes a list of numbers and returns the minimum value in the list.
# (Optional) If the list is empty, have the function return False.
# Example: minimum([37,2,1,-9]) should return -9
# (Optional) Example: minimum([]) should return False
def minimum(func_list):
    func_list = func_list
    if len(func_list) > 0:
        minVal    = func_list[0]
        for i in range(0,len(func_list),1):
            if func_list[i] < minVal:
                minVal = func_list[i]
        return minVal
    else:
        return False 

test_minVal_01 = minimum([37,2,1,-9])
print(test_minVal_01)

test_minVal_02 = minimum([])
print(test_minVal_02)

# 7
# Maximum - Create a function that takes a list and returns the maximum value in the array.
# (Optional) If the list is empty, have the function return False.
# Example: maximum([37,2,1,-9]) should return 37
# (Optional) Example: maximum([]) should return False
def maximum(func_list):
    func_list = func_list
    if len(func_list) > 0:
        maxVal    = func_list[0]
        for i in range(0,len(func_list),1):
            if func_list[i] > maxVal:
                maxVal = func_list[i]
        return maxVal
    else:
        return False 

test_maxVal_01 = maximum([37,2,1,-9])
print(test_maxVal_01)

test_maxVal_02 = maximum([])
print(test_maxVal_02)

# 8
# Ultimate Analysis (Optional) - Create a function that takes a list
# and returns a dictionary that has the sumTotal, average, minimum, maximum and length of the list.
# Example: ultimate_analysis([37,2,1,-9]) should return {'sumTotal': 31, 'average': 7.75, 'minimum': -9, 'maximum': 37, 'length': 4 }
def ultimate_analysis(func_list):
    func_list = func_list
    func_dict = {}
    Ave       = 0
    sumVal    = 0
    if len(func_list) > 0:
        maxVal    = func_list[0]
        minVal    = func_list[0]
        for i in range(0,len(func_list),1):
            sumVal = sumVal + func_list[i]
            if func_list[i] > maxVal:
                maxVal = func_list[i]
            elif func_list[i] < minVal:
                minVal = func_list[i]
    else:
        return False
    Ave = sumVal / int(len(func_list))
    
    func_dict['sumTotal'] = sumVal
    func_dict['average']  = Ave
    func_dict['minimum']  = minVal
    func_dict['maximum']  = maxVal
    func_dict['length']   = len(func_list)

    return func_dict

test_UA = ultimate_analysis([37,2,1,-9])
print(test_UA) 

# 9
# Reverse List (Optional) - Create a function that takes a list and return that list with values reversed.
# Do this without creating a second list. (This challenge is known to appear during basic technical interviews.)
# Example: reverse_list([37,2,1,-9]) should return [-9,1,2,37]
def reverse_list(func_list):
    func_list = func_list
    x         = len(func_list)
    for i in range(1,len(func_list)+1,1):
        func_list.append(func_list[x-i])
    for i in range(0,x,1):
        func_list.pop(0)  
    return func_list

test_RL_01 = reverse_list([37,2,1,-9])
print(test_RL_01)

test_RL_02 = reverse_list([0,1,2,3,4,5,6,7,8,9,10])
print(test_RL_02)







