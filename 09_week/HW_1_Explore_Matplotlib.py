# ================================================================================================================
# Arwa Ashi - HomeWork 1 Week 9 - Nov 1, 2020 || Saudi Digital Academy 
# ================================================================================================================
# Question
# In the previous section is a link to the matplotlib documentation.
# Explore the example plots and reproduce a least three of them in
# your own Jupyterlab environment. Create one of each of these types:
# 1- line plot
# 2- scatter plot
# 3- bar plot
# ----------------------------------------------------------------------------------------------------------------
# Answer

from math import *
from random import *
import numpy as np
from numpy import linalg as LA
from numpy import linalg as LA
import scipy.linalg as sl
import scipy.optimize as opt
from scipy import interpolate
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns

# For more fun !!
# I am solving a problem and using matplotlib and seaborn for visualization !!

# The Problem:
# The path an ant takes while walking across a sidewalk is tracked.
# The following data points when ﬁt with a natural cubic spline function
# (d^2 S(x)/ dx^2 = 0 on the ends)
# depict the path of the ant.

# Methodology:
# 1
# The coefficients for natural cubic spline found by solving nonlinear system.

# 2
# The two point Gaussian quadrature rule used to find the lenght of the ant's path across the side walk.
# The reason is the basis of the two point Gaussian quadrature is {1 , x, x^2, x^3}
# so we can integrate any polynomial of the third degree exactly.

# ----------------------------------------------------------------------------------------------------------------
xData = np.array([ 0.0 , 2.0, 4.0, 6.0, 8.0])
yData = np.array([-9.0 , 8.0, 5.0, 4.0, 7.0])
plt.scatter(xData, yData, alpha=0.5)
plt.show()

# The coefficients for natural cubic spline found by solving nonlinear system.
# ----------------------------------------------------------------------------------------------------------------
def nonlinear_system(variables,xData=xData,yData=yData):
    
    
    (a0,b0,c0,d0,a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3) = variables

    #-----
    
    eq1 = a0 + b0*xData[0] + c0*xData[0]**2.0+ d0*xData[0]**3.0 - yData[0]  

    eq2 = a0 + b0*xData[1] + c0*xData[1]**2.0+ d0*xData[1]**3.0 - yData[1]

    eq3 = b0 + 2.0*c0*xData[1] + 3*d0*xData[1]**2.0 - b1 - 2.0*c1*xData[1] - 3.0*d1*xData[1]**2.0

    eq4 = c0 + 3.0*d0*xData[1] - c1 - d1*3.0*xData[1]
    
    #-----
    
    eq5 = a1 + b1*xData[1] + c1*xData[1]**2.0+ d1*xData[1]**3.0 - yData[1]  

    eq6 = a1 + b1*xData[2] + c1*xData[2]**2.0+ d1*xData[2]**3.0 - yData[2]

    eq7 = b1 + 2.0*c1*xData[2]+ 3*d1*xData[2]**2.0 - b2 - 2.0*c2*xData[2] - 3.0*d2*xData[2]**2.0

    eq8 = c1 + 3.0*d1*xData[2] - c2 - d2*3.0*xData[2]

    #----- 
    
    eq9  = a2 + b2*xData[2] + c2*xData[2]**2.0+ d2*xData[2]**3.0 - yData[2]  

    eq10 = a2 + b2*xData[3] + c2*xData[3]**2.0+ d2*xData[3]**3.0 - yData[3]

    eq11 = b2 + 2.0*c2*xData[3]+ 3*d2*xData[3]**2.0 - b3 - 2.0*c3*xData[3] - 3.0*d3*xData[3]**2.0

    eq12 = c2 + 3.0*d2*xData[3] - c3 - d3*3.0*xData[3]
    
    #----- 
    
    eq13 = a3 + b3*xData[3] + c3*xData[3]**2.0+ d3*xData[3]**3.0 - yData[3]  

    eq14 = a3 + b3*xData[4] + c3*xData[4]**2.0+ d3*xData[4]**3.0 - yData[4]

    eq15 = 2.0*c0 + 6.0*d0*xData[0] 

    eq16 = 2.0*c3 + 6.0*d3*xData[4]

    return[eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8,eq9,eq10,eq11,eq12,eq13,eq14,eq15,eq16]      

solution    = opt.fsolve(nonlinear_system,(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)) 
a0,b0,c0,d0,a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3 = opt.fsolve(nonlinear_system,(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)) 

# print ("The solution is",solution)
# print ("The solution is a0=",a0,"and b0=",b0,"and c0=",c0,"and d0=",d0)
# print ("The solution is a1=",a1,"and b1=",b1,"and c1=",c1,"and d1=",d1)
# print ("The solution is a2=",a2,"and b2=",b2,"and c2=",c2,"and d2=",d2)
# print ("The solution is a3=",a3,"and b3=",b3,"and c3=",c3,"and d3=",d3)


# Cubic-spline interpolation
# ----------------------------------------------------------------------------------------------------------------
def Q0(x):
    return a0 + (b0*x) + (c0*x**2.0) + (d0*x**3.0)

def Q1(x):
    return a1 + (b1*x) + (c1*x**2.0) + (d1*x**3.0)

def Q2(x):
    return a2 + (b2*x) + (c2*x**2.0) + (d2*x**3.0)

def Q3(x):
    return a3 + (b3*x) + (c3*x**2.0) + (d3*x**3.0)

tck  = interpolate.splrep(xData, yData, s=0)
xnew = np.arange(0,2*np.pi,np.pi/50)
ynew = interpolate.splev(xnew, tck, der=0)

plt.figure()
plt.plot(xData, yData, 'x', xnew, ynew,'--', xData, yData, 'b')

q0x = np.linspace(xData[0],xData[1],100)
q0y = Q0(q0x)
plt.plot(q0x,q0y,'r')

q1x = np.linspace(xData[1],xData[2],100)
q1y = Q1(q1x)
plt.plot(q1x,q1y,'r')

q2x = np.linspace(xData[2],xData[3],100)
q2y = Q2(q2x)
plt.plot(q2x,q2y,'r')

q3x = np.linspace(xData[3],xData[4],100)
q3y = Q3(q3x)
plt.plot(q3x,q3y,'r')

plt.legend(['Linear', 'Cubic Spline','stright lines','Qi(x) Cubic Spline'],loc=4)
plt.title('Cubic-spline interpolation')
plt.show()


# test
def nonlinear_system(variables,a=-1,b=1):
    
    (w1,w2,x1,x2) = variables
    
    eq1 = w1 + w2 - (b - a)
    eq2 = w1*x1 + w2*x2 - (b**2.0 - a**2.0)/2.0
    eq3 = w1*x1**2.0 + w2*x2**2.0 - (b**3.0 - a**3.0)/3.0
    eq4 = w1*x1**3.0 + w2*x2**3.0 - (b**4.0 - a**4.0)/4
    return [eq1,eq2,eq3,eq4]

solution    = opt.fsolve(nonlinear_system,(1,1,1,1)) 
w1,w2,x1,x2 = opt.fsolve(nonlinear_system,(1,1,1,1)) 
# print ("The solution is",solution)
# print ("The solution is w1=",w1,"and w2=",w2,"and x1=",x1,"and x2=",x2)

# first interval from 0 to 2
a = xData[0]
b = xData[1]

def nonlinear_system(variables,a=a,b=b):
    
    (w1,w2,x1,x2) = variables
    
    eq1 = w1 + w2 - (b - a)
    eq2 = w1*x1 + w2*x2 - (b**2.0 - a**2.0)/2.0
    eq3 = w1*x1**2.0 + w2*x2**2.0 - (b**3.0 - a**3.0)/3.0
    eq4 = w1*x1**3.0 + w2*x2**3.0 - (b**4.0 - a**4.0)/4
    return [eq1,eq2,eq3,eq4]

solution    = opt.fsolve(nonlinear_system,(1,1,1,1)) 
w1,w2,x1,x2 = opt.fsolve(nonlinear_system,(1,1,1,1)) 
# print ("The solution is",solution)
# print ("The solution is w1=",w1,"and w2=",w2,"and x1=",x1,"and x2=",x2)

def mapp(x,a=a,b=b):
    return ((b-a)/2.0)*(x+1.0) + a

def TwoPtGauss0(f,a=a,b=b,w1=w1,w2=w2,x1=x1,x2=x2):
    return ((b-a)/2.0)*(w1*f(mapp(x1,a,b)) + w2*f(mapp(x2,a,b)))

def CompositQuad0(f,A,B,m):
    xpts = np.linspace(A,B,m);
    qsum = 0.0; 
    for i in range(m-1):
        qsum += TwoPtGauss0(f,xpts[i],xpts[i+1])
    
    return qsum

a = xData[1]
b = xData[2]

def nonlinear_system(variables,a=a,b=b):
    
    (w1,w2,x1,x2) = variables
    
    eq1 = w1 + w2 - (b - a)
    eq2 = w1*x1 + w2*x2 - (b**2.0 - a**2.0)/2.0
    eq3 = w1*x1**2.0 + w2*x2**2.0 - (b**3.0 - a**3.0)/3.0
    eq4 = w1*x1**3.0 + w2*x2**3.0 - (b**4.0 - a**4.0)/4
    return [eq1,eq2,eq3,eq4]

solution    = opt.fsolve(nonlinear_system,(1,1,1,1)) 
w1,w2,x1,x2 = opt.fsolve(nonlinear_system,(1,1,1,1)) 
# print ("The solution is",solution)
# print ("The solution is w1=",w1,"and w2=",w2,"and x1=",x1,"and x2=",x2)

def mapp(x,a=a,b=b):
    return ((b-a)/2.0)*(x+1.0) + a

def TwoPtGauss1(f,a=a,b=b,w1=w1,w2=w2,x1=x1,x2=x2):
    return ((b-a)/2.0)*(w1*f(mapp(x1,a,b)) + w2*f(mapp(x2,a,b)))

def CompositQuad1(f,A,B,m):
    xpts = np.linspace(A,B,m);
    qsum = 0.0; 
    for i in range(m-1):
        qsum += TwoPtGauss1(f,xpts[i],xpts[i+1])
    
    return qsum

a = xData[2]
b = xData[3]

def nonlinear_system(variables,a=a,b=b):
    
    (w1,w2,x1,x2) = variables
    
    eq1 = w1 + w2 - (b - a)
    eq2 = w1*x1 + w2*x2 - (b**2.0 - a**2.0)/2.0
    eq3 = w1*x1**2.0 + w2*x2**2.0 - (b**3.0 - a**3.0)/3.0
    eq4 = w1*x1**3.0 + w2*x2**3.0 - (b**4.0 - a**4.0)/4
    return [eq1,eq2,eq3,eq4]

solution    = opt.fsolve(nonlinear_system,(1,1,1,1)) 
w1,w2,x1,x2 = opt.fsolve(nonlinear_system,(1,1,1,1)) 
# print ("The solution is",solution)
# print ("The solution is w1=",w1,"and w2=",w2,"and x1=",x1,"and x2=",x2)

def mapp(x,a=a,b=b):
    return ((b-a)/2.0)*(x+1.0) + a

def TwoPtGauss2(f,a=a,b=b,w1=w1,w2=w2,x1=x1,x2=x2):
    return ((b-a)/2.0)*(w1*f(mapp(x1,a,b)) + w2*f(mapp(x2,a,b)))

def CompositQuad2(f,A,B,m):
    xpts = np.linspace(A,B,m);
    qsum = 0.0; 
    for i in range(m-1):
        qsum += TwoPtGauss2(f,xpts[i],xpts[i+1])
    
    return qsum

a = xData[3]
b = xData[4]

def nonlinear_system(variables,a=a,b=b):
    
    (w1,w2,x1,x2) = variables
    
    eq1 = w1 + w2 - (b - a)
    eq2 = w1*x1 + w2*x2 - (b**2.0 - a**2.0)/2.0
    eq3 = w1*x1**2.0 + w2*x2**2.0 - (b**3.0 - a**3.0)/3.0
    eq4 = w1*x1**3.0 + w2*x2**3.0 - (b**4.0 - a**4.0)/4
    return [eq1,eq2,eq3,eq4]

solution    = opt.fsolve(nonlinear_system,(1,1,1,1)) 
w1,w2,x1,x2 = opt.fsolve(nonlinear_system,(1,1,1,1)) 
# print ("The solution is",solution)
# print ("The solution is w1=",w1,"and w2=",w2,"and x1=",x1,"and x2=",x2)

def mapp(x,a=a,b=b):
    return ((b-a)/2.0)*(x+1.0) + a

def TwoPtGauss3(f,a=a,b=b,w1=w1,w2=w2,x1=x1,x2=x2):
    return ((b-a)/2.0)*(w1*f(mapp(x1,a,b)) + w2*f(mapp(x2,a,b)))

def CompositQuad3(f,A,B,m):
    xpts = np.linspace(A,B,m);
    qsum = 0.0; 
    for i in range(m-1):
        qsum += TwoPtGauss3(f,xpts[i],xpts[i+1])
    
    return qsum

# Since the integration is finding the area under the curve. we need to find a lenght
# of the cure inorder of finding the lenght of ant's path.
# Using https://en.wikipedia.org/wiki/Arc_length as a resorce to find the lenght of a curve
# ----------------------------------------------------------------------------------------------------------------
# 1 taking the derivative of Qi(x) inorder to find the lenght
def dQ0(x):
    return (1 + (3*(d0)*x**2.0 + 2*(c0)*x + (b0))**2.0)**(1/2)

def dQ1(x):
    return (1 + (3*(d1)*x**2.0 + 2*(c1)*x + (b1))**2.0)**(1/2)

def dQ2(x):
    return (1 + (3*(d2)*x**2.0 + 2*(c2)*x + (b2))**2.0)**(1/2)

def dQ3(x):
    return (1 + (3*(d3)*x**2.0 + 2*(c3)*x + (b3))**2.0)**(1/2)

ans0 = CompositQuad0(dQ0,xData[0],xData[1],200)
ans1 = CompositQuad1(dQ1,xData[1],xData[2],200)
ans2 = CompositQuad2(dQ2,xData[2],xData[3],200)
ans3 = CompositQuad3(dQ3,xData[3],xData[4],200)
# print (ans0)
# print (ans1)
# print (ans2)
# print (ans3)
approximation_lenght= ans0 + ans1 + ans2 + ans3
# print ("approximation lenght=", ans0 + ans1 + ans2 +ans3)

# -------
from scipy.integrate import quad

def integrandd0(x):
    return dQ0(x)

def integrandd1(x):
    return dQ1(x)

def integrandd2(x):
    return dQ2(x)

def integrandd3(x):
    return dQ3(x)

ans0 , err = quad(integrandd0, xData[0], xData[1])
ans1 , err = quad(integrandd1, xData[1], xData[2])
ans2 , err = quad(integrandd2, xData[2], xData[3])
ans3 , err = quad(integrandd3, xData[3], xData[4])

# print (ans0)
# print (ans1)
# print (ans2)
# print (ans3)
Exact_lenght = ans0 + ans1 + ans2 + ans3
# print ("Exact lenght = ", ans0 + ans1 + ans2 + ans3)

# Using an appropriate Gaussian quadrature rule to ﬁnd the length of the ant’s path across the side walk. 
# ----------------------------------------------------------------------------------------------------------------
# integrat exact by using wolframalpha
Exact_dQ0 = 17.1313
Exact_dQ1 = 4.84638
Exact_dQ2 = 2.849
Exact_dQ3 = 3.61787
Exact_lenght = Exact_dQ0 + Exact_dQ1 + Exact_dQ2 + Exact_dQ3
# print ("Exact lenght = ", Exact_dQ0 + Exact_dQ1 + Exact_dQ2 + Exact_dQ3)

# Test 
ans0 = CompositQuad0(Q0,xData[0],xData[1],200)
ans1 = CompositQuad1(Q1,xData[1],xData[2],200)
ans2 = CompositQuad2(Q2,xData[2],xData[3],200)
ans3 = CompositQuad3(Q3,xData[3],xData[4],200)
# print (ans0)
# print (ans1)
# print (ans2)
# print (ans3)
labels             = ['Area','Lenght']
approximation_Area = [round(ans0,2)+round(ans1,2)+round(ans2,2)+round(ans3,2), round(approximation_lenght,2)]
# print ("approximation Area =", ans0 + ans1 + ans2 +ans3)

from scipy.integrate import quad

def integrand0(x):
    return Q0(x)

def integrand1(x):
    return Q1(x)

def integrand2(x):
    return Q2(x)

def integrand3(x):
    return Q3(x)

ans0 , err = quad(integrand0, xData[0], xData[1])
ans1 , err = quad(integrand1, xData[1], xData[2])
ans2 , err = quad(integrand2, xData[2], xData[3])
ans3 , err = quad(integrand3, xData[3], xData[4])

# print (ans0)
# print (ans1)
# print (ans2)
# print (ans3)
Exact_Area = [round(ans0,2)+round(ans1,2)+round(ans2,2)+round(ans3,2), round(Exact_lenght,2)]
# print ("Exact Area = ", ans0 + ans1 + ans2 + ans3)
# So, the approximate Area of the ant’s path across the side walk is 35.1907267321 and the exact Area is 35.1428571429.

x     = np.arange(len(labels))  # the label locations
width = 0.35                    # the width of the bars

fig, ax = plt.subplots()
s1      = ax.bar(x - width/2, approximation_Area, width, label='Approximate solution')
s2      = ax.bar(x + width/2, Exact_Area, width, label='Exact solution')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Approximate and Exact Area and Lenght')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=4)

def labelFunction(Ss):
    for s in Ss:
        height = s.get_height()
        ax.annotate('{}'.format(height),
                    xy=(s.get_x() + s.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

labelFunction(s1)
labelFunction(s2)

fig.tight_layout()

plt.show()
