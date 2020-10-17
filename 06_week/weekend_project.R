# =======================================================
# Arwa Ashi  - Weekend project - week 6 - Oct 17, 2020
# =======================================================

# Weekend Project - Optional
# You have been Hired by Napa Wine inc. to help build a decision model. Mr. Johnson the winemaker 
# at Napa Wine believes that a storm is coming and he needs to decide whether or not to harvest 
# the grapes today or wait until after the storm. 

# -------------------------------------------------
# Write a function that solves a decision tree
# -------------------------------------------------
# Condition Type:
# 1- harvest
# 2- not harvest
# 

# Weather Type:
# 1- sun
# 2- storm

# -------------------------------------------------
# First considering assumptions : 
# -------------------------------------------------
# 1- Assuming 40 % of the past predicted storm happen and 60% not.

# 2- If Mr. Johnson harvest today and weather is sun tomorrow, he earns $8000. 
#    If he harvest today and weather is storm tomorrow , he earns $8000.

# 3- If Mr. Johnson not harvest today and weather is sun tomorrow, he earnd $8000. 
#    If he not harvest today and weather is storm tomorrow, he earns $0.


# -------------------------------------------------
# Second building the decision tree
# -------------------------------------------------
Decision_tree <- function(condition="harvest", weather="sun"){
  if(condition=="harvest"){
    if(weather=="sun"){
      earn   = 8000
      prob   = 0.6
    }else{
      earn  = 8000
      prob  = 0.4
    }
    x = earn * prob
  }
  
  if(condition=="not harvest"){
    if(weather=="sun"){
      earn  = 8000
      prob  = 0.6
    }else{
      earn  = 0
      prob  = 0.4
    }
    x = earn * prob
  }
  
  return(x)
}
Decision_tree(condition="not harvest", weather="sun")+Decision_tree(condition="not harvest", weather="storm")

(Decision_tree_table <- tribble(
  ~condition     ,   ~Expected_value,
  #--------------|-------------
  "harvest"      , Decision_tree(condition="harvest", weather="sun")+Decision_tree(condition="harvest", weather="storm"),
  "not harvest"  , Decision_tree(condition="not harvest", weather="sun")+Decision_tree(condition="not harvest", weather="storm")
))
# As a result the best expected value is "harvest" Today  = $8000.
