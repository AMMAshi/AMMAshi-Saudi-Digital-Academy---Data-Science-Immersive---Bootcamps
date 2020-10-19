#============================================
# Arwa Ashi - HW 2 - Week 7 - Oct 19, 2020
#============================================
# 1 =========================================
def a():
    return 5
print(a())    # The result is 5

# 2 =========================================
def a():
    return 5
print(a()+a())
# The result is 10

# 3 =========================================
def a():
    return 5
    return 10
print(a())    # The result is 5 

# 4 =========================================
def a():
    return 5
    print(10)
print(a())    # The result is 5 

# 5 =========================================
def a():
    print(5)
x = a()       # The result is 5
print(x)      # The result is None no return in the function

# 6 =========================================
def a(b,c):
    print(b+c)
print(a(1,2) + a(2,3))
# The result is 3 and 5 + an error
# The sum of Nonetype equation created an error

# 7 =========================================
def a(b,c):
    return str(b)+str(c)
print(a(2,5))
# The result is 2 and 5 as strings to be 25

# 8 =========================================
def a():
    b = 100
    print(b)
    if b < 10:
        return 5
    else:
        return 10
    return 7
print(a())
# The result is 100 and 10

# 9 =========================================
def a(b,c):
    if b<c:
        return 7
    else:
        return 14
    return 3
print(a(2,3))          # The result is 7
print(a(5,3))          # The result is 14
print(a(2,3) + a(5,3)) # The result is the sum of 7 and 14 = 21

# 10 ========================================
def a(b,c):
    return b+c
    return 10
print(a(3,5)) # The result is 8

# 11 ========================================
b = 500
print(b)      # The result is 500
def a():
    b = 300
    print(b)
print(b)      # The result is 500
a()           # The result is 300    
print(b)      # The result is 500                

# 12 ========================================
b = 500
print(b)      # The result is 500
def a():
    b = 300
    print(b)
    return b
print(b)      # The result is 500
a()           # The result is 300
print(b)      # The result is 500

# 13 ========================================
b = 500
print(b)      # The result is 500
def a():
    b = 300
    print(b)
    return b
print(b)      # The result is 500
b=a()         # The result is 300
print(b)      # The result is 300

# 14 ========================================
def a():
    print(1)
    b()
    print(2)
def b():
    print(3)
a()           # The result is 1, 3, 2

# 15 ========================================
def a():
    print(1)
    x = b()
    print(x)
    return 10
def b():
    print(3)
    return 5
y = a()       # The result is 1, 3, 5
print(y)      # The result is 10      

