#============================================
# Arwa Ashi - HW 2 - Week 7 - Oct 19, 2020
#============================================

# 1
# Countdown - Create a function that accepts a number as an input.
# Return a new list that counts down by one, from the number (as the 0th element) down to 0 (as the last element).
# Example: countdown(5) should return the list: [5,4,3,2,1,0]
def CountDown(a):
    count_list = []
    count_list.append(a)
    for i in range(0,a,1):
        a = a - 1
        count_list.append(a)
    return count_list

test_count_down = CountDown(5)
print(test_count_down)
       
# 2
#Print and Return - Create a function that will receive a list with two numbers.
#Print the first value and return the second.
#Example: print_and_return([1,2]) should print 1 and return 2
def print_and_return(func_list): 
    func_list = func_list 
    print(func_list[0])
    return func_list[1]

print_and_return([1,2])           # the function print the first value
test_PR = print_and_return([7,9]) # the function print the first value
print(test_PR)                    # the function return the second

# 3
# First Plus Length - Create a function that accepts a list 
# and returns the sum of the first value in the list plus the list's length.
# Example: first_plus_length([1,2,3,4,5]) should return 6 (first value: 1 + length: 5)
def first_plus_length(func_list):
    func_list = func_list
    sumVal = func_list[0] + int(len(func_list))
    return sumVal #," which is the sum of the first value:", func_list[0], "+ lenght :", len(func_list)  

test_FPL = first_plus_length([1,2,3,4,5])
print(test_FPL)
 
# 4
# This Length, That Value- Write a function that accepts two integers as parameters: size and value.
# The function should create and return a list whose length is equal to the given size,
# and whose values are all the given value.
# Example: length_and_value(4,7) should return [7,7,7,7]
# Example: length_and_value(6,2) should return [2,2,2,2,2,2]
def length_and_value(size,value):
    func_list = []
    for i in range(0,size,1):
        func_list.append(value)
    return func_list

test_LV_01 = length_and_value(4,7)
print(test_LV_01)

test_LV_02 = length_and_value(6,2)
print(test_LV_02)

# 5
# Values Greater than Second (Optional) -
# Write a function that accepts a list and creates a new list
# containing only the values from the original list that are greater than its 2nd value.
# Print how many values this is and then return the new list.
# If the list has less than 2 elements, have the function return False
# Example: values_greater_than_second([5,2,3,2,1,4]) should print 3 and return [5,3,4]
# Example: values_greater_than_second([3]) should return False
def values_greater_than_second(func_list):
    func_list   = func_list
    second_list = []
    if len(func_list) > 2:
        for i in range(0,len(func_list),1):
            if func_list[i] > func_list[1]:
                second_list.append(func_list[i])
        print(len(second_list))        
        return second_list
    else:
        return False
    
test_VGS_01 = values_greater_than_second([5,2,3,2,1,4])
print(test_VGS_01)

test_VGS_02 = values_greater_than_second([3])
print(test_VGS_02)

test_VGS_03 = values_greater_than_second([10,7,3,2,9,4])
print(test_VGS_03)









