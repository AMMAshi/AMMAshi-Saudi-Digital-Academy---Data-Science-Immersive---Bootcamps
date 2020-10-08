#==================================================
# Arwa Ashi - Homework 3 week 5 - Oct 7, 2020
#==================================================
# Complete the exercises on chapter 16 of R4DS
library(tidyverse)
library(lubridate)
library(nycflights13)

#--------------------------------------------------
# 16.2.4 Exercises
#--------------------------------------------------
# Q1- What happens if you parse a string that contains invalid dates?
ymd(c("2010-10-10", "bananas"))
# Answer: Warning message: 1 failed to parse.

# Q2- What does the tzone argument to today() do? Why is it important?
# Answer
today()             # "2020-10-08"
today(tz = "UTC")   # "2020-10-07"
# tzone control the day and time inside the today function.
# It is important if I am using two datasets from different time zone.

# Q3- Use the appropriate lubridate function to parse each of the following dates:
d1 <- "January 1, 2010"
d2 <- "2015-Mar-07"
d3 <- "06-Jun-2017"
d4 <- c("August 19 (2015)", "July 1 (2015)")
d5 <- "12/30/14" # Dec 30, 2014
# Answer
d1 <- mdy("January 1, 2010")
d2 <- ymd("2015-Mar-07")
d3 <- dmy("06-Jun-2017")
d4 <- mdy(c("August 19 (2015)", "July 1 (2015)"))
d5 <- mdy("12/30/14") # Dec 30, 2014

#--------------------------------------------------
# 16.3.4 Exercises
#--------------------------------------------------
# Q1- How does the distribution of flight times within a day change over the course of the year?
# Answer 
flights_dt %>% 
  mutate(month = month(dep_time), distribution_flight_times = hour(dep_time)*300) %>% 
  ggplot(aes(distribution_flight_times)) +
  geom_freqpoly(aes(group = month, color = month),binwidth = 300)

# Q2- Compare dep_time, sched_dep_time and dep_delay. Are they consistent? Explain your findings.
# Answer: No
flights_dt %>%
  mutate(minute = minute(dep_time) - minute(sched_dep_time)) %>% 
  group_by(dep_delay) %>% 
  summarise(
    avg_delay = mean(arr_delay, na.rm = TRUE),
    n = n()) %>% 
  ggplot(aes(dep_delay, avg_delay)) +
  geom_line()

flights_dt %>%
  mutate(minute = minute(dep_time) - minute(sched_dep_time)) %>% 
  group_by(minute) %>% 
  summarise(
    avg_delay = mean(arr_delay, na.rm = TRUE),
    n = n()) %>% 
  ggplot(aes(minute, avg_delay)) +
  geom_line()

flights_dt %>%
  mutate(minute = minute(dep_time) - minute(sched_dep_time)) %>% 
  select( origin, dest, minute, dep_delay)
#  A tibble: 328,063 x 4
#  origin dest  minute dep_delay
#  <chr>  <chr>  <int>     <dbl>
# 1 EWR    IAH        2         2
# 2 LGA    IAH        4         4
# 3 JFK    MIA        2         2
# 4 JFK    BQN       -1        -1
# 5 LGA    ATL       54        -6
# 6 EWR    ORD       -4        -4
# 7 EWR    FLL       55        -5
# 8 LGA    IAD       57        -3
# 9 JFK    MCO       57        -3
# 10 LGA    ORD       58        -2
# … with 328,053 more rows

# Q3- Compare air_time with the duration between the departure and arrival. Explain your findings. (Hint: consider the location of the airport.)
# Answer
flights_dt %>%
  mutate(ymd_hms = ymd_hms(arr_time) - ymd_hms(dep_time)) %>% 
  select(origin, dest, air_time, ymd_hms)

# Q4- How does the average delay time change over the course of a day? Should you use dep_time or sched_dep_time? Why?
# Answer  
flights_dt %>% 
  mutate(hour= hour(dep_time)) %>% 
  group_by(hour) %>% 
  summarise(
    avg_delay = mean(arr_delay, na.rm = TRUE),
    n = n()) %>% 
  ggplot(aes(hour, avg_delay)) +
  geom_line()

flights_dt %>% 
  mutate(hour= hour(sched_dep_time)) %>% 
  group_by(hour) %>% 
  summarise(
    avg_delay = mean(arr_delay, na.rm = TRUE),
    n = n()) %>% 
  ggplot(aes(hour, avg_delay)) +
  geom_line()
# sched_dep_time should be used to get the whole picture of the day.

# Q5- On what day of the week should you leave if you want to minimise the chance of a delay?
# Answer
flights_dt %>% 
  mutate(wday = wday(dep_time, label = TRUE)) %>% 
  ggplot(aes(x = wday)) +
  geom_bar()
# Saturday, less flight!

# Q6- What makes the distribution of diamonds$carat and flights$sched_dep_time similar?
# Answer 
diamonds%>%
  #filter(carat<3)%>%
  ggplot(aes(x = carat))+
  geom_histogram(binwidth=0.1)

flights_dt %>% 
  #filter(hour(sched_dep_time)>6)%>%
  ggplot(aes(hour(sched_dep_time))) + 
  geom_histogram(binwidth = 0.1) 
# Both are right skewed
  
# Q7- Confirm my hypothesis that the early departures of flights in minutes 20-30 and 50-60 are caused by scheduled flights that leave early. Hint: create a binary variable that tells you whether or not a flight was delayed.
# Answer
flights_dt %>% 
  mutate(minute = minute(dep_time)) %>% 
  group_by(minute) %>% 
  summarise(
    avg_delay = mean(arr_delay, na.rm = TRUE),
    n = n()) %>% 
  ggplot(aes(minute, avg_delay)) +
  geom_line()
# It looks like flights leaving in minutes 20-30 and 50-60 have a much lower delay than the rest of the hour regarding what time scheduled flights is.

#--------------------------------------------------
# 16.4.5 Exercises
#--------------------------------------------------
# Q1- Why is there months() but no dmonths()?
# Answer
months(1)  #[1] "1m 0d 0H 0M 0S"
dmonths(1) #[1] "2629800s (~4.35 weeks)"
# dmonths() The function exists, but the result will be present in weeks.

# Q2- Explain days(overnight * 1) to someone who has just started learning R. How does it work?
# Answer
# In these are overnight flights, We used the same date information for both the departure and the arrival times, 
# but these flights arrived on the following day.
# We can fix this by adding days(1) to the arrival time of each overnight flight.

# Q3- Create a vector of dates giving the first day of every month in 2015. Create a vector of dates giving the first day of every month in the current year.
# Answer
(DM2015 <- mdy("January 1st, 2015"))
(listM2015 <- DM2015 + months(0:11))

(DM2020 <- mdy("January 1st, 2020"))
(listM2020 <- DM2020 + months(0:11))

# Q4- Write a function that given your birthday (as a date), returns how old you are in years.
# Answer
AgeFunctionCalcu<- function(DOB){
  age<-today() - ymd(DOB)
  return(as.duration(age))
}
AgeFunctionCalcu(19860110)

# Q5- Why can’t (today() %--% (today() + years(1))) / months(1) work?
# Answer
# Well, if the year was 2015 it should return 365 = 12 month, but if it was 2016, it should return 366 = 12 month! 
# There’s not quite enough information for lubridate to give a single clear answer. 
(today() %--% (today() + years(1))) / dmonths(1)
(today() %--% (today() + years(1))) / months(1)

