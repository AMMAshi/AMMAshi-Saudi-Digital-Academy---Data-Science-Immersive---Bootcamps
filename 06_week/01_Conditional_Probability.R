# ========================================
# Arwa Ashi HW 1 week 6 Oct 12, 2020
# ========================================
# Joint Probability      : Probability of events A intersection B.
# Marginal Probability   : Probability of event A.
# Conditional Probability: Probability of event A given event B.

# Library
library(tidyverse)

# -------------------------------------------------
# Question 1: 
# Write a function that takes marginals and conditionals as inputs and calculates the all the conditionals, marginals and joints
# -------------------------------------------------
# Given: 
# Marginal Probability   : Probability of event A. 
# Conditional Probability: Probability of event B given event A & Probability of event Bnot given event A.

prob_cal <- function(prob_A= 0.1, prob_B_given_A = 0.9, prob_B_given_Anot = 0.1){
  #  calculates the all the conditionals, marginals and joints
  Prob_cal <- tribble(
    ~prob               ,    ~value,
    #-------------------|-------------
    # Marginal Probability
    "prop_Anot" , 1 - prop_A,
    
    # Conditional Probability
    "prob_Bnot_given_A"   ,     1 - prob_B_given_A,
    "prob_Bnot_given_Anot",  1 - prob_B_given_Anot,
    
    # Joint Probability
    "prob_AinB"       , prob_B_given_A * prop_A,
    "prob_AinBnot"    , prob_B_given_Anot * prop_A,
    "prob_AnotinB"    , prob_B_given_Anot * prop_Anot,
    "prob_AnotinBnot" , prob_Bnot_given_Anot * prop_Anot
    
  )
  return(Prob_cal)
}
prob_cal(prob_A= 0.1, prob_B_given_A = 0.9, prob_B_given_Anot = 0.1)

# -------------------------------------------------
# Write a function that solves a decision tree
# -------------------------------------------------
# Condition Type:
# 1- Outdoor
# 2- Porch
# 3- Indoor

# Weather Type:
# Sun
# Rain

Decision_tree <- function(condition="Outdoors", weather="sun"){
  if(condition=="Outdoors"){
    if(weather=="sun"){
      pay    = 100
      prob   = 0.4
    }else{
      pay  = 0
      prob = 0.6
    }
    x = pay * prob
  }
  
  if(condition=="Porch"){
    if(weather=="sun"){
      pay  = 90
      prob = 0.4
    }else{
      pay  = 20
      prob = 0.6
    }
    x = pay * prob
  }

  if(condition=="Indoor"){
    if(weather=="sun"){
      pay  = 40
      prob = 0.4
    }else{
      pay  = 50 
      prob = 0.6
    }
    x = pay * prob
  }
  return(x)
}
Decision_tree(condition="Outdoors", weather="sun")+Decision_tree(condition="Outdoors", weather="Rain")

(Decision_tree_table <- tribble(
  ~condition           ,   ~Expected_value,
  #-----------------------|-------------
  "Outdoors"              , Decision_tree(condition="Outdoors", weather="sun")+Decision_tree(condition="Outdoors", weather="Rain"),
  "Porch"                 , Decision_tree(condition="Porch", weather="sun")+Decision_tree(condition="Porch", weather="Rain"),
  "Indoor"                , Decision_tree(condition="Indoor", weather="sun")+Decision_tree(condition="Indoor", weather="Rain")
))
# As a result the best expected value is "Porch"  = $48.






