#============================================
# Arwa Ashi - HW 2 - Week 7 - Oct 19, 2020
#============================================

#Basic - Print all integers from 0 to 150. Hint:use a for loop and range
for i in range(0,151,1):
    print(i)

#Multiples of Five - Print all the multiples of 5 from 5 to 1,000
for x in range(5,1001,1):
    new_x = 5 * x
    print(x,"multiply by 5 = ", new_x)
   
#Counting, the Dojo Way - Print integers 1 to 100. If divisible by 5, print "Coding" instead.
#If divisible by 10, print "Coding Dojo".
for x in range(1,101,1):
    if x % 5 == 0 and x % 10 != 0:
        x = "Coding"
    elif x % 10 == 0:
        x = "Coding Dojo"
    print(x)

#Whoa. That Sucker's Huge - Add odd integers from 0 to 500,000, and print the final sum.
count = 0
for i in range(0,500001,1):
    if i % 2 != 0:
        count = count + i
print(count)

#Countdown by Fours - Print positive numbers starting at 2018, counting down by fours.
count_2 = 2022
while count_2 < 2023:
    count_2 = count_2 - 4
    if count_2 < 0:
        break
    print(count_2)
else:
    print("No -ve, +ve numbers ONLY ! :D")

#Flexible Counter (optional) - Set three variables: lowNum, highNum, mult.
#Starting at lowNum and going through highNum, print only the integers that are a multiple of mult.
#For example, if lowNum=2, highNum=9, and mult=3, the loop should print 3, 6, 9 (on successive lines)
lowNum  = 2
highNum = 9
mult    = 3
empty_list = []
for x in range(lowNum,highNum+1,1):
    if x % mult == 0:
        empty_list.append(x)
print(empty_list)
    








