#==================================================
# Arwa Ashi - Homework 2 week 5 - Oct 5, 2020
#==================================================
# Work out the exercises on Ch 12 and 13 on R for Data Science (R4DS)
# https://r4ds.had.co.nz/data-visualisation.html

#==================================================
# Finish Exercises in Ch 12 of R4DS
#==================================================

#--------------------------------------------------
# 12.2.1 Exercises
#--------------------------------------------------
# Q1- Using prose, describe how the variables and observations are organised in each of the sample tables.
# Answer:
# table1  : tidy data, Each variable has its own column, Each observation has its own row, Each value has its own cell.
# table2  : Each variable does not have  its own column.
# table3  : Each variable does not have  its own column.
# table4a : Each variable does not have  its own column and a column names are values of variables.
# table4b : Each variable does not have  its own column and a column names are values of variables.

# Q2- Compute the rate for table2, and table4a + table4b. You will need to perform four operations:
# Q2- 1- Extract the number of TB cases per country per year.
# Q2- 2- Extract the matching population per country per year.
# Q2- 3- Divide cases by population, and multiply by 10000.
# Q2- 4- Store back in the appropriate place.
# Q- Which representation is easiest to work with? Which is hardest? Why? 
# Answer: 
# All are the same, I need to write one line of code to get tidy data table.
# for table2
(tidy_table2<- table2 %>%
   pivot_wider(names_from = type, values_from = count)%>%
   mutate(cases=as.numeric(cases))%>%
   mutate(population=as.numeric(population))%>%
   mutate(rate=cases*10000/population)%>%
   unite("rate_", cases, population, sep="*10000/", remove = FALSE))
# for table3
(tidy_table3<- table3 %>%
    separate(rate, into = c("cases","population"),sep="/")%>%
    mutate(cases=as.numeric(cases))%>%
    mutate(population=as.numeric(population))%>%
    mutate(rate=cases*10000/population))
# for table4a
(tidy_table4a<- table4a %>%
  pivot_longer(c('1999','2000'),names_to = "year", values_to = "cases"))
# for table4b
(tidy_table4b<- table4b %>%
    pivot_longer(c('1999','2000'),names_to = "year", values_to = "population"))  
# for table4a and table4b
(tidy_table4ab <- left_join(tidy_table4a,tidy_table4b)%>%
    mutate(cases=as.numeric(cases))%>%
    mutate(population=as.numeric(population))%>%
    mutate(rate=cases*10000/population))

# Q3- Recreate the plot showing change in cases over time using table2 instead of table1. What do you need to do first?
# Answer:
# I need to do pivot_wider first.
ggplot(tidy_table2,aes(year,cases))+
  geom_line(aes(group=country),color="grey50")+
  geom_point(aes(color=country))

#--------------------------------------------------
# 12.3.3 Exercises
#--------------------------------------------------
# Q1- Why are pivot_longer() and pivot_wider() not perfectly symmetrical?
#  Carefully consider the following example:
(stocks <- tibble(
    year   = c(2015, 2015, 2016, 2016),
    half   = c(   1,    2,     1,    2),
    return = c(1.88, 0.59, 0.92, 0.17)
))
  (stocks %>% 
    pivot_wider(names_from = year, values_from = return) %>% 
    pivot_longer(`2015`:`2016`, names_to = "year", values_to = "return"))
#(Hint: look at the variable types and think about column names.)
#pivot_longer() has a names_ptypes argument, e.g.  names_ptypes = list(year = double()). What does it do?
# Answer:
# pivot_wider take names and values from two different column and transefer them into new columns with match values.
# pivot_longer take names of selected columns and transfer them into values inside one column.
# pivot_longer take values of selected columns and transfer them into one raw that match the selected name.

# Q2- Why does this code fail?
    table4a %>% 
    pivot_longer(c(1999, 2000), names_to = "year", values_to = "cases")
  #> Error: Can't subset columns that don't exist.
  #> ✖ Locations 1999 and 2000 don't exist.
  #> ℹ There are only 3 columns.
# Answer: 
# the code fail because c(1999, 2000) is supposed to be `1999`:`2000`
(table4a %>% 
    pivot_longer(`1999`:`2000`, names_to = "year", values_to = "cases"))    

# Q3- What would happen if you widen this table? Why? How could you add a new column to uniquely identify each value?
   (people <- tribble(
      ~name,             ~names,  ~values,
      #-----------------|--------|------
      "Phillip Woods",   "age",       45,
      "Phillip Woods",   "height",   186,
      "Phillip Woods",   "age",       50,
      "Jessica Cordero", "age",       37,
      "Jessica Cordero", "height",   156
    ))
# Answer
(people %>% 
    group_by(name,names)%>%
    mutate(row=row_number())%>%
    pivot_wider(id_cols = c(row, names),names_from  = name, values_from = values)%>%
    select(-row))
# The question here is "Phillip Woods" update his age? or there is two "Phillip Woods"???
    
# Q4- Tidy the simple tibble below. Do you need to make it wider or longer? What are the variables?
(preg <- tribble(
      ~pregnant, ~male, ~female,
      "yes",     NA,    10,
      "no",      20,    12
))
# Answer
# pivot_longer is used to bring gender in one column:
(preg %>% 
      pivot_longer(`male`:`female`,names_to ="gender", values_to ="total",values_drop_na = TRUE))

#--------------------------------------------------
# 12.4.3 Exercises
#--------------------------------------------------
# Q1- What do the extra and fill arguments do in separate()? Experiment with the various options for the following two toy datasets.
tibble(x = c("a,b,c", "d,e,f,g", "h,i,j")) %>% 
  separate(x, c("one", "two", "three"))
tibble(x = c("a,b,c", "d,e", "f,g,i")) %>% 
  separate(x, c("one", "two", "three"))
# Answer
?separate
# extra:
tibble(x = c("a,b,c", "d,e,f,g", "h,i,j")) %>% 
  separate(x, c("one", "two", "three"), extra = "drop")
#If sep is a character vector, this controls what happens when there are too many pieces. There are three valid options:
#"warn" (the default): emit a warning and drop extra values.
#"drop": drop any extra values without a warning.
#"merge": only splits at most length(into) times
# fill
tibble(x = c("a,b,c", "d,e,f,g", "h,i,j")) %>% 
  separate(x, c("one", "two", "three","four"), fill ="left",remove = FALSE)
#If sep is a character vector, this controls what happens when there are not enough pieces. There are three valid options:
#"warn" (the default): emit a warning and fill from the right
#"right": fill with missing values on the right
#"left": fill with missing values on the left

# Q2- Both unite() and separate() have a remove argument. What does it do? Why would you set it to FALSE?
# Answer: False mean do not remove the original columns.

# Q3- Compare and contrast separate() and extract(). Why are there three variations of separation (by position, by separator, and with groups), but only one unite?
# Answer
# ------------------------------------------------
#       separate()      
# ------------------------------------------------
table3 %>%
  separate(rate,into = c("cases","population"),sep = "/")
table3 %>%
  separate(rate,into = c("cases","population"))
# ------------------------------------------------
#       extract()
# ------------------------------------------------
?extract
table3 %>%
  extract(rate,into=c("cases","population"),regex ="([[:alnum:]]+)/([[:alnum:]]+)")
tibble(variable = c("X", "X", "Y", "Y"), id = c(1, 2, 1, 2))
# separate() & extract() can be used to do the same job as the above code showed
# however,  extract() need an additional arguments into regex as "([[:alnum:]]+)/([[:alnum:]]+)" that helps the function undrestand the method 
# ------------------------------------------------
#       unite()
# ------------------------------------------------
(table3 %>%
  separate(rate,into = c("cases","population"))%>%
  unite("rate_", cases, population, sep="/", remove = FALSE))
# unite() combaind the values into one column
# separate() & extract() distribute the values into several columns.

#--------------------------------------------------
# 12.5.1 Exercises
#--------------------------------------------------
# Q1- Compare and contrast the fill arguments to pivot_wider() and complete().
# Answer:
(stocks <- tibble(
  year   = c(2015, 2015, 2015, 2015, 2016, 2016, 2016),
  qtr    = c(   1,    2,    3,    4,    2,    3,    4),
  return = c(1.88, 0.59, 0.35,   NA, 0.92, 0.17, 2.66)
))
# pivot_wider()
?pivot_wider
stocks %>% 
  pivot_wider(names_from = year, values_from = return, values_fill= 0 )
# complete()
?complete
stocks %>% 
  complete(year, nesting(qtr),fill = list(return = 0))
# complete update Explicitly and Implicitly missing values.
# pivot_wider update an Implicitly missing value.

# Q2- What does the direction argument to fill() do?
# Answer: replacing NA by the most recent non-missing value.
# direction determines NA should replaced by previous or next non-missing value.

#--------------------------------------------------
# 12.6.1 Exercises
#--------------------------------------------------
(df <- who)
#--------------------1- pivot_longer
(Edit_00_df <- df %>% pivot_longer(
    cols = new_sp_m014:newrel_f65, 
    names_to = "key", 
    values_to = "cases", 
    values_drop_na = TRUE
  ))
Edit_00_df %>% count(key)
#--------------------2- mutate 
(Edit_01_df <- Edit_00_df %>% 
  mutate(key = stringr::str_replace(key, "newrel", "new_rel")))
#-------------------- 3- separate key
(Edit_02_df <- Edit_01_df %>% 
  separate(key, c("new", "type", "sexage"), sep = "_"))
Edit_02_df %>% count(new)
#--------------------4-
(Edit_03_df  <- Edit_02_df  %>% 
  select(-new, -iso2, -iso3))
#--------------------5- separate sexage into sex and age
(Edit_04_df <- Edit_03_df %>% 
  separate(sexage, c("sex", "age"), sep = 1))

# Q1- In this case study I set values_drop_na = TRUE just to make it easier to check that we had the correct values. Is this reasonable? Think about how missing values are represented in this dataset. Are there implicit missing values? What’s the difference between an NA and zero?
# Answer: No it is not resnable because we can not determain NA mean a person did not have TB or missing data!!
Edit_00_df %>%
  filter(cases == 0) %>%
  nrow()
# since there is zero in TB result, so NA is for missing info.
pivot_longer(who, c(new_sp_m014:newrel_f65), names_to = "key", values_to = "cases") %>%
  group_by(country, year) %>%
  mutate(prop_missing = sum(is.na(cases)) / n()) %>%
  filter(prop_missing > 0, prop_missing < 1)
nrow(who)
who %>% 
  complete(country, year) %>%
  nrow()

anti_join(complete(who, country, year), who, by = c("country", "year")) %>% 
  select(country, year) %>% 
  group_by(country) %>% 
  # so I can make better sense of the years
  summarise(min_year = min(year), max_year = max(year))
            
# Q2- What happens if you neglect the mutate() step? (mutate(names_from = stringr::str_replace(key, "newrel", "new_rel")))
# Answer: then the separate function will not count rel, stands for cases of relapse, as a type of TB

# Q3- I claimed that iso2 and iso3 were redundant with country. Confirm this claim.
# Answer: iso2 and iso3 contian the country initial letters.

# Q4- For each country, year, and sex compute the total number of cases of TB. Make an informative visualisation of the data.
(Edit_04_df %>% 
    group_by(country,year,sex) %>% 
    pivot_wider(names_from = age, values_from = cases)%>%
    mutate(`014`=as.numeric(`014`))%>%
    mutate(`1524`=as.numeric(`1524`))%>%
    mutate(`2534`=as.numeric(`2534`))%>%
    mutate(`3544`=as.numeric(`3544`))%>%
    mutate(`4554`=as.numeric(`4554`))%>%
    mutate(`5564`=as.numeric(`5564`))%>%
    mutate(`65`=as.numeric(`65`))%>%
    mutate(total_cases = `014`+`1524`+`2534`+`3544`+`4554`+`5564`+`65`)%>%
    select(-`014`,-`1524`,-`2534`,-`3544`,-`4554`,-`5564`,-`65`,-type)%>%
    unite(country_gender, country, sex, remove = FALSE)%>%
    filter(year > 1995) %>%
    filter(country == "Afghanistan") %>%
    ggplot(aes(x = year, y = total_cases, group = country_gender, colour = sex))+
    geom_line())

#==================================================
# Finish Exercises in Ch 13 of R4DS
#==================================================
library("tidyverse")
library("nycflights13")
library("maps")
library("ggplot2")

View(flights)
View(airlines)
View(airports)
planes
View(weather)
#--------------------------------------------------
# 13.2.1 Exercises
#--------------------------------------------------
# Q1- Imagine you wanted to draw (approximately) the route each plane flies from its origin to its destination. What variables would you need? What tables would you need to combine?
# Answer:
(O_table<- airports %>% 
    select(origin = faa, O_Latitude=lat, O_Longitude=lon) %>%
    inner_join(flights, by="origin"))

(D_table<- airports %>% 
    select(dest = faa, D_Latitude=lat, D_Longitude=lon) %>%
    inner_join(flights, by="dest"))

(O_D_table<- left_join(O_table,D_table)%>%
    select(tailnum,origin,O_Latitude,O_Longitude,dest,D_Latitude,D_Longitude))

O_D_table %>%
  slice(1:100) %>%
  ggplot(aes(
    x = O_Longitude, xend = D_Longitude,
    y = O_Latitude,  yend = D_Latitude
  )) +
  borders("state") +
  geom_segment(arrow = arrow(length = unit(0.1, "cm"))) +
  coord_quickmap() +
  labs(y = "Latitude", x = "Longitude")

# Q2- I forgot to draw the relationship between weather and airports. What is the relationship and how should it appear in the diagram?
# Answer:
(weather_O_table<- airports %>% 
    select(origin = faa, O_Latitude=lat, O_Longitude=lon) %>%
    inner_join(weather, by="origin"))


# Q3- weather only contains information for the origin (NYC) airports. If it contained weather records for all airports in the USA, what additional relation would it define with flights?
# Answer:
# year, month, day, hour, dest 

# Q4- We know that some days of the year are “special”, and fewer people than usual fly on them. How might you represent that data as a data frame? What would be the primary keys of that table? How would it connect to the existing tables?
# Answer: 
# 1- adding a column with speacial date into flights table 
# 2- or creating seperte table and join it to flights table by date

#--------------------------------------------------
# 13.3.1 Exercises
#--------------------------------------------------
# Q1- Add a surrogate key to flights.
# Answer: a surrogate key
# 1- add one with mutate() 
# 2- add row_number(). 
(flights %>% 
    group_by(year, month, day, tailnum)%>%
    mutate(row=row_number())%>%
    select(year, month, day, tailnum, row)%>%
    filter(row > 3))

# Q2- Identify the keys in the following datasets
# Q2- 1- Lahman::Batting,
# Q2- 2- babynames::babynames
# Q2- 3- nasaweather::atmos
# Q2- 4- fueleconomy::vehicles
# Q2- 5- ggplot2::diamonds
#(You might need to install some packages and read some documentation.)
# Answer:
library(Lahman)
Batting
(Batting %>% 
    count(playerID, yearID, stint) %>% 
    filter(n > 1))
library(babynames)
babynames
(babynames %>% 
    count(year,sex,name) %>% 
    filter(n > 1))
library(nasaweather)
atmos
(atmos %>% 
    count(lat,long,year,month) %>% 
    filter(n > 1))
library(fueleconomy)
vehicles
(vehicles %>% 
    count(id) %>% 
    filter(n > 1))
library(ggplot2)
diamonds
(diamonds %>% 
    group_by(carat,cut,color,clarity,depth)%>%
    mutate(row=row_number())%>%
    select(carat,cut,color,clarity,depth,row)%>%
    filter(row >= 3))

# Q3- Draw a diagram illustrating the connections between the Batting, People, and Salaries tables in the Lahman package. Draw another diagram that shows the relationship between People, Managers, AwardsManagers.
# How would you characterise the relationship between the Batting, Pitching, and Fielding tables?
# Answer:
library(Lahman)
View(Batting)        # playerID, yearID, teamID, IgID
View(People)         # playerID
View(Salaries)       # playerID, yearID, teamID, IgID

View(People)         # playerID
View(Managers)       # playerID, yearID, teamID, IgID
View(AwardsManagers) # playerID, yearID, IgID

View(Batting)        # playerID, yearID, teamID, IgID
View(Pitching)       # playerID, yearID, teamID, IgID, stint
View(Fielding)       # playerID, yearID, teamID, IgID, stint

#--------------------------------------------------
# 13.4.6 Exercises
#--------------------------------------------------
# Q1- Compute the average delay by destination, then join on the airports data frame so you can show the spatial distribution of delays. Here’s an easy way to draw a map of the United States:
  airports %>%
  semi_join(flights, c("faa" = "dest")) %>%
  ggplot(aes(lon, lat)) +
  borders("state") +
  geom_point() +
  coord_quickmap()
#(Don’t worry if you don’t understand what semi_join() does — you’ll learn about it next.)
# You might want to use the size or colour of the points to display the average delay for each airport.
# Answer:

# Q2- Add the location of the origin and destination (i.e. the lat and lon) to flights.
# Answer:

# Q3- Is there a relationship between the age of a plane and its delays?
# Answer:

# Q4- What weather conditions make it more likely to see a delay?
# Answer:

# Q5- What happened on June 13 2013? Display the spatial pattern of delays, and then use Google to cross-reference with the weather.
# Answer:

#--------------------------------------------------
# 13.5.1 Exercises
#--------------------------------------------------
# Q1- What does it mean for a flight to have a missing tailnum? What do the tail numbers that don’t have a matching record in planes have in common? (Hint: one variable explains ~90% of the problems.)
# Answer:

# Q2- Filter flights to only show flights with planes that have flown at least 100 flights.
# Answer:

# Q3- Combine fueleconomy::vehicles and fueleconomy::common to find only the records for the most common models.
# Answer:

# Q4- Find the 48 hours (over the course of the whole year) that have the worst delays. Cross-reference it with the weather data. Can you see any patterns?
# Answer:

# Q5- What does anti_join(flights, airports, by = c("dest" = "faa")) tell you? What does anti_join(airports, flights, by = c("faa" = "dest")) tell you?
# Answer:

# Q6- You might expect that there’s an implicit relationship between plane and airline, because each plane is flown by a single airline. Confirm or reject this hypothesis using the tools you’ve learned above.
# Answer:











