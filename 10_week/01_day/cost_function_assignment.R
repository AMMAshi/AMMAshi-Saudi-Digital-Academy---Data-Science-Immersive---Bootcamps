# =======================================================================
# Arwa Ashi - HW1 - Week 10
# Saudi Digital Academy
# Machine Learning
# =======================================================================

# Heuristic Models (Cost Function Extension)
# =======================================================================

# Look at the seattle weather in the data folder.
# Come up with a heuristic model to predict if
# it will rain today. Keep in mind this is a time series,
# which means that you only know what happened 
# historically (before a given date). One example of a
# heuristic model is: it will rain tomorrow if it rained
# more than 1 inch (>1.0 PRCP) today. Describe your heuristic
# model here

#############################################################
##################### Your model Here #######################
#############################################################

# Examples:
# if it rained yesterday it will rain today
# if it rained yesterday or the day before, it will rain today
# Here is an example of how to build and populate 

# =======================================================================
# A heuristic model
# =======================================================================
library(tidyverse)

df <- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/10_week/01_day/seattle_weather_1948-2017.csv")
print(df)
print(names(df))
numrow = 25549

heuristic_df <- data.frame("Yesterday" = 0,
                           "Today" = 0,
                           "Tomorrow" = 0,
                           "Guess" = FALSE,
                           "Rain Tomorrow" = FALSE,
                           "Correct" = FALSE,
                           "True Positive" = FALSE,
                           "False Positive" = FALSE,
                           "True Negative" = FALSE,
                           "False Negative" = FALSE)
print(heuristic_df)
print(names(heuristic_df))

# Now let's populate our heuristic model guessess
df$PRCP = ifelse(is.na(df$PRCP),
                 ave(df$PRCP, FUN = function(x) mean(x, na.rm = TRUE)),df$PRCP)
print(df$PRCP)

for (z in 1:numrow){
  i = z + 2
  
  yesterday    = df[i-2,2]
  today        = df[i-1,2]
  tomorrow     = df[i,2]
  
  if (tomorrow == 0){
    rain_tomorrow = FALSE
  }else{
    rain_tomorrow = TRUE
  }
  
  heuristic_df[z,1]  = yesterday
  heuristic_df[z,2]  = today
  heuristic_df[z,3]  = tomorrow
  heuristic_df[z,4]  = FALSE          # Label all guesses as false
  heuristic_df[z,5]  = rain_tomorrow
  heuristic_df[z,7]  = FALSE
  heuristic_df[z,8]  = FALSE
  heuristic_df[z,9]  = FALSE
  heuristic_df[z,10] = FALSE
  
  if ((today > 0) & (yesterday > 0)){
    heuristic_df[z,4] = TRUE
  }
  if (heuristic_df[z,4] == heuristic_df[z,5]){
    heuristic_df[z,6] = TRUE
    if (heuristic_df[z,4] == TRUE){
      heuristic_df[z,7] = TRUE #true positive
    }else{
      heuristic_df[z,9] = TRUE #True negative
    }
  }else{
    heuristic_df[z,6] = FALSE
    if (heuristic_df[z,4] == TRUE){
      heuristic_df[z,7] = TRUE #false positive
    }else{
      heuristic_df[z,9] = TRUE #false negative
    }
  }
}

# Split data into training and testing
# enter split function here to make h_train and h_test subsets of the data

# split 0.3131238
h_test   <- heuristic_df[1:8000,]
h_train  <- heuristic_df[8001:25549,]

# Calculate the accuracy of your predictions
# we used this simple approach in the first part to see what percent of the time we where correct 
# calculated as (true positive + true negative)/ number of guesses
print(sum(heuristic_df$Correct)/nrow(heuristic_df))
print((sum(heuristic_df$True.Positive)+sum(heuristic_df$True.Negative))/nrow(heuristic_df))
print((sum(h_test$True.Positive)+sum(h_test$True.Negative))/nrow(h_test))
print(sum(h_train$True.Positive,h_train$True.Negative)/nrow(h_train))

# Calculate the precision of your prediction
# precision is the percent of your postive prediction which are correct
# more specifically it is calculated (num true positive)/(num tru positive + num false positive)
print(sum(heuristic_df$True.Positive)/sum(heuristic_df$True.Positive,heuristic_df$False.Positive))
print(sum(h_test$True.Positive)/sum(h_test$True.Positive,h_test$False.Positive))
print(sum(h_train$True.Positive)/sum(h_train$True.Positive,h_train$False.Positive))

# Calculate the recall of your predictions
# recall the percent of the time you are correct when you predict positive
# more specifically it is calculated (num true positive)/(num tru positive + num false negative)
print(sum(heuristic_df$True.Positive)/sum(heuristic_df$True.Positive,heuristic_df$False.Negative))
print(sum(h_test$True.Positive)/sum(h_test$True.Positive,h_test$False.Negative))
print(sum(h_train$True.Positive)/sum(h_train$True.Positive,h_train$False.Negative))

# The sum of squared error (SSE) of your predictions
# SSE = sum(actual - predicted^2)
SST <- sum((heuristic_df$Guess - mean(heuristic_df$Guess))^2)
SSR <- sum((heuristic_df$Correct - mean(heuristic_df$Guess))^2)
SSE <- SST - SSR 
print(SSE)

SST <- sum((h_train$Guess - mean(h_train$Guess))^2)
SSR <- sum((h_train$Correct - mean(h_train$Guess))^2)
SSE <- SST - SSR 
print(SSE)

SST <- sum((h_test$Guess - mean(h_test$Guess))^2)
SSR <- sum((h_test$Correct - mean(h_test$Guess))^2)
SSE <- SST - SSR 
print(SSE)
