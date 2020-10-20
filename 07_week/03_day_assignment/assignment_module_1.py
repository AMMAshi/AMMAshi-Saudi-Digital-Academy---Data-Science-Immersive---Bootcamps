#===================================================
# Arwa Ashi - HW 3 - Week 8 - Oct 19, 2020
#===================================================

#==================================================================
# In each cell complete the task using basic Python functions
#==================================================================

# 1- print the numbers between 1 and 100
for i in range(100):
    print(i+1)

# 2- print the numbers between 1 and 100 divisible by 8
for i in range(100):
    if((i+1)%8 == 0):
        print(i+1)
    else:
        next

# 3- Use a while loop to find the first 20 numbers divisible by 5
start = 1
count = 0
while count<20:
    if (start%5==0):
        print(start)
        count+=1
    start +=1

# 4- Here is an example function that adds two numbers
def simple_adding_function(a,b):
    return a + b
print(simple_adding_function(3,4))

# 5- Create a function that evaluates if a number is prime (you can not use a list of known primes). Only allow values between 0 and 100.
def prime(a):
    for i in range(2,a):
        if((a%(i) == 0)&(a!=(i))):
            return(a," is not Prime")
    return(a," is a prime number")
print(prime(13))

