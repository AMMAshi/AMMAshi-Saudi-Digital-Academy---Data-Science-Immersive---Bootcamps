# ==============================================================
# Arwa Ashi - HW 2 week 6 - Oct 13, 2020
# ============================================================== 
# Now you get to expand and modify the models for LP Brewery and Memoryless Bar.

# LP Brewery is coming up with a hot new brew called The Barrel-Aged Dantzig. Because it is a special craft beer, 
# it has a margin of $30, It uses 10 pounds of corn, 4 oz of hops and 15 pounds of malt. With the addition of this 
# new brew, LP Brewery has to now think about allocating its personnel to the different manufacturing processes. 
# Each of the kegs takes 5 (Hopatronic), 10 (All American), and 20 (Dantzig) hours of labor to make and we have only 
# 5 employees full-time. If this is the production planning for a month of brewing, 
# what is the optimal amount of each beer that must be produced to maximize profit

# Memoryless bar, on the other hand, is tired of running out of inventory and missing a lot of potential sales. They 
# have hired you to help them figure out what they need to do to have a probability of 60% or higher of having 2 kegs 
# on hand in the long run. There are several things that you can change to attempt this task: changing the order rules,
# expanding the chain to allow for more inventory to accumulate. Experiment with a few of these and make a recommendation 
# to the bar owners, your recommendation should include a diagram, a transition matrix, and the steady-state values.

# -------------------------------------------------------------- 
# Make improvements to LP Brewery's manufacturing schedule
# -------------------------------------------------------------- 

# Answer:
# ----------------------|----------|-------|---------------|-----------|------|------|------  
# Products              | variable | Price | Hour of Labor | employees | Hops | Corn | Malt
# ----------------------|----------|-------|---------------|-----------|------|------|------ 
# Hopatronic            | x_1      | $ 13  | 5  hours      | 1         | 4    | 5    | 35
# American Kolsch style | x_2      | $ 23  | 10 hours      | 1         | 4    | 15   | 20
# Barrel-Aged Dantzig   | x_3      | $ 30  | 20 hours      | 1         | 4    | 10   | 15
# ----------------------|----------|-------|---------------|-----------|------|------|------ 

# First  : Assuming month = 29 days = 696 hours
# Second : objective Max z = 13 x_1 + 23  x_2 + 30  x_3)
# Third  : Subject to:       5h x_1 + 10h x_2 + 20h x_3 <= 696    (Hours = 1 month)
#                               x_1 +     x_2 +     x_3 <= 5      (Empolyees      )
#                            4  x_1 + 4   x_2 + 4   x_3 <= 150    (Hops           )
#                            5  x_1 + 15  x_2 + 10  x_3 <= 4800   (Corn           )
#                            35 x_1 + 20  x_2 + 15  x_3 <= 1190   (Malt           )
#                               x_1, x_2, x_3 >= 0
# Fourth : build a matrix fucntion:

library(lpSolve)

f.obj <- c(13,23,30)

f.col <- matrix(c(5,10,20,      # Hours
                  1, 1, 1,      # Employee
                  4, 4, 4,      # Hops
                  5,15,10,      # Corn
                 35,20,15),     # Malt  
                nrow  = 5, 
                byrow = T)

f.dir <- c("<=",
           "<=",
           "<=",
           "<=",
           "<=")

f.rhs<-c(696,
         5  ,
         160,
         4800,
         1190)

# Fifth: solve the LP system
sol<-lp("max",f.obj,f.col,f.dir,f.rhs,compute.sens = T)
sol$objval
sol$solution

# Understanding the dual help you understand the shadow prices
sol$duals

# -------------------------------------------------------------- 
# Make improvements to Memoryless Bar's supply chain decisions
# -------------------------------------------------------------- 

# Answer:
#-----------------------
# A diagram code :
library('heemod')
library('diagram')
supply_chain_diagram <- define_transition(
  state_names = 
   c('0', '1', '2', '3'),
    0.20, 0.15, 0.65, 0.00,
    0.15, 0.20, 0.65, 0.00,
    0.25, 0.05, 0.65, 0.05,
    0.20, 0.15, 0.60, 0.05);

curves <- matrix(nrow = 4, ncol = 4, 0.04)

plot(supply_chain_diagram, 
     curve=curves, 
     self.shiftx = c(0.1,-0.1,0), 
     self.shifty = c(-0.1,-0.1,0.15), 
     self.arrpos = c(1,2.1,1))

# A transition matrix 
library(pgirmess)
library(expm)
# Creating transition matrix
T <- matrix(c(0.20, 0.15, 0.65, 0.00,
              0.15, 0.20, 0.65, 0.00,
              0.25, 0.05, 0.65, 0.05,
              0.20, 0.15, 0.60, 0.05), nrow = 4, byrow = TRUE)
# creating column and row names
colnames(T) = c(0,1,2,3)
rownames(T) = c(0,1,2,3)
T
# The steady-state values
T%^%20













