# ===================================================
# Arwa Ashi - HW 4 - Week 8 - Oct 21, 2020
# ===================================================

# ===================================================
# Linear Algebra Assignment
# ===================================================

# Use your Jupyterlab environment and numpy from your MatrixDS project
# to solve the attached systems of equations. What values do you get
# for x and y for each?

# Use Numpy to Solve the Systems of Equations

import numpy as np

# Solve : -------------------------------------------
# 3x + 5y = 6
# 7x - 5y = 9
A1 = np.array([[3,5],[7,-5]])
b1 = np.array([[6],[9]])
x1 = np.dot(np.linalg.inv(A1), b1)
print(A1,"\n",b1,"\n",x1)
# x = 1.5 , y = 0.3


# Solve : -------------------------------------------
# 2x + 5y - z = 27
#  x +  y + z = 6
#       y + z = -4
A2 = np.array([[2,5,-1],[1,1,1],[0,1,1]])
b2 = np.array([[27],[6],[-4]])
x2 = np.dot(np.linalg.inv(A2), b2)
print(A2,"\n",b2,"\n",x2)
# x = 10.0 , y= 0.5, z= -4.5


# Solve : -------------------------------------------
# x + y = 4
# 2x + 2y = 8
A3 = np.array([[1,1],[2,2]])
b3 = np.array([[4],[8]])
x3 = np.dot(np.linalg.inv(A3), b3)
# print(A3,"\n",b3,"\n",x3)

# What happens with the last solution? Why?
# I got an error for having a Singular matrix
'''
LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
'''
