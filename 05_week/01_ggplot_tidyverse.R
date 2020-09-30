#==================================================
# Arwa Ashi - Homework 1 week 5 - Sep 29, 2020
#==================================================
# Questions and requirements:
# Read through Chapter 3 of R for Data Science https://r4ds.had.co.nz/data-visualisation.html
# Do the Exercises that appear in sections 3.2.4, 3.3.1, 3.5.1, 3.6.1, 3.7.1 and 3.8.1.
#==================================================

library(tidyverse)
df<-mpg
#==================================================
# 3.2.4 Exercises
#==================================================
# Q1- Run ggplot(data = mpg). What do you see?
ggplot(data = mpg)  
# The previous function created an empty graph.

# Q2- How many rows are in mpg? How many columns?
# There are 11 columns and 234 rows in the mpg data.

# Q3- What does the drv variable describe? Read the help for ?mpg to find out.
?mpg
#drv discribe the type of drive train, where f = front-wheel drive, 
#                                            r = rear wheel drive, 
#                                            4 = 4wd

# Q4- Make a scatterplot of hwy vs cyl.
ggplot(data = mpg)+
  geom_point(mapping = aes(x=hwy,y=cyl,color=class))

# Q5- What happens if you make a scatterplot of class vs drv? Why is the plot not useful?
ggplot(data = mpg)+
  geom_point(mapping = aes(x=class,y=drv,color=class))
# The plot is not useful because it did not display a clear relationship between the two variables.

#==================================================
# 3.3.1 Exercises
#==================================================
# Q1- What’s gone wrong with this code? Why are the points not blue?
  ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, color = "blue"))
# the wrong thing is the color in calling inside the mapping 
# the following is the correct code that makes all of the points in the plot blue:
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy), color = "blue")

# Q2- Which variables in mpg are categorical? Which variables are continuous? (Hint: type ?mpg to read the documentation for the dataset). How can you see this information when you run mpg?
# categorical variables are class and drv, and continuous variables are displ, hwy, year, cyl, cty.
# I can see this information when I run mpg by using the following code
ggplot(data= mpg)+
  geom_smooth(mapping=aes(x=displ,y=hwy, linetype=drv, color= drv))

# Q3- Map a continuous variable to color, size, and shape. How do these aesthetics behave differently for categorical vs. continuous variables?
ggplot(data= mpg)+
  geom_smooth(mapping=aes(x=displ,y=hwy, size= hwy))
# for geom_smooth
# 1- For linetype: I got an Error: A continuous variable can not be mapped to linetype
# 2- For shape: I got an  Error: A continuous variable can not be mapped to shape
# 3- For color and size: A continuous variable does not categorize the plot one color one shape one size.

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, size = hwy))
# for geom_point
# 1- For shape: I got an Error: A continuous variable can not be mapped to shape
# 2- For color and size: I got a range of color and size that cannot help me categorize the variables.

# Q4- What happens if you map the same variable to multiple aesthetics?
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = hwy, y = hwy))
# The plot displays a slope line with geom_smooth and a slop points with geom_point.

# Q5- What does the stroke aesthetic do? What shapes does it work with? (Hint: use ?geom_point)
?geom_point
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy),stroke = 2, shape=5)
# All shapes under shape works and stroke is used to increase the shape size.

# Q6- What happens if you map an aesthetic to something other than a variable name, like aes(colour = displ < 5)? Note, you’ll also need to specify x and y.
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, color = displ < 5 ))
# color = displ < 5 will give us a category or a filter by true and false depend on the given condition.

#==================================================
# 3.5.1 Exercises
#==================================================
# Q1- What happens if you facet on a continuous variable?
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy)) + 
  facet_wrap(~ displ, nrow = 2)
# 1- The plot will display each different value in the data frame as subplots.
# 2- R will take longer time to run the code comparing to the categorical variables.
# 3- The result will not give a clear explanation of a story.

# Q2- What do the empty cells in plot with facet_grid(drv ~ cyl) mean? How do they relate to this plot?
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = drv, y = cyl))+
  facet_grid(drv ~ cyl)
# The empty cells in plot with facet_grid(drv ~ cyl) mean there is no intersection between drv and cyl.

# Q3- What plots does the following code make? What does . do?
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy)) +
  facet_grid(drv ~ .)
# Displaying the data in 3 different horizontal subplots that are categorized by drv.

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy)) +
  facet_grid(. ~ cyl)
# Displaying the data in 4 different vertical subplots that are categorized by cyl.

# Q4- Take the first faceted plot in this section:
# What are the advantages to using faceting instead of the colour aesthetic? What are the disadvantages? How might the balance change if you had a larger dataset?
  ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy)) + 
  facet_wrap(~ class, nrow = 2)
# The advantages:
# 1- Helping to understand the behavior for each category
# 2- Helping us telling a clearer story if I have a large dataset.  
# The disadvantages:
# Increasing the subplots may confuse to the reader.
#### If I had a larger dataset, the balance change might clearer between the subplots.  
 
# Q5- Read ?facet_wrap. What does nrow do? What does ncol do? What other options control the layout of the individual panels? Why doesn’t facet_grid() have nrow and ncol arguments?
?facet_wrap  
# Number of rows that separate the subplots. 
# The other options that control the layout of the individual panels, are ncol, scales, switch, dir, and strip.position.
# The reason that facet_grid() does not have nrow and ncol arguments because the subplots depend on the categories inside the two compared variables.
  
# Q6- When using facet_grid() you should usually put the variable with more unique levels in the columns. Why?
# Because the subplots depend on the categories inside the two compared variables. By increasing the subplots, the result will not be useful.
  
#==================================================
# 3.6.1 Exercises
#==================================================
# Q1- What geom would you use to draw a line chart? A boxplot? A histogram? An area chart?
geom_line()       # draw a line chart 
geom_boxplot()    # draw a boxplot
geom_histogram()  # draw a histogram chart 
geom_area()       # draw an area chart 
  
# Q2- Run this code in your head and predict what the output will look like. Then, run the code in R and check your predictions.
ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color = drv)) + 
    geom_point() + 
    geom_smooth(se = FALSE)
  
# Q3- What does show.legend = FALSE do? What happens if you remove it? Why do you think I used it earlier in the chapter?
ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color = drv)) + 
  geom_point(show.legend = FALSE) + 
  geom_smooth(se = FALSE, show.legend = FALSE)    
# show.legend = FALSE means removing the geom plot type from the legend.
# If I remove show.legend = FALSE, the geom plot type categories will appear in the result with the plot.
# I think the instructor used it earlier in the chapter to remove drv categories from the plots.

# Q4- What does the se argument to geom_smooth() do?
# se = FALSE means remove the confidence bands on the smooth.
    
# Q5- Will these two graphs look different? Why/why not?
ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
    geom_point() + 
    geom_smooth()
ggplot() + 
    geom_point(data = mpg, mapping = aes(x = displ, y = hwy)) + 
    geom_smooth(data = mpg, mapping = aes(x = displ, y = hwy))
# These two graphs look the same.
# To avoid this type of repetition of mapping inside the functions, in the first code, the variables are as passing a set of mappings in ggplot()

# Q6- Recreate the R code necessary to generate the following graphs.
ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
  geom_point(stroke = 2) + 
  geom_smooth(se = FALSE)

ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
  geom_point(stroke = 2) + 
  geom_smooth(mapping = aes(group = drv), se = FALSE)

ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color = drv)) + 
  geom_point(stroke = 2) + 
  geom_smooth(mapping = aes(group = drv), se = FALSE)

ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
  geom_point(mapping = aes(color = drv), stroke = 2) + 
  geom_smooth(se = FALSE, show.legend = FALSE)

ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
  geom_point(mapping = aes(color = drv), stroke = 2) + 
  geom_smooth(mapping = aes(linetype=drv), se = FALSE)

ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
  geom_point(
    mapping = aes(fill = drv), 
    stroke = 2,
    shape = 21,
    color = "white",
    size = 5
    ) 

#==================================================
# 3.7.1 Exercises
#==================================================
# Q1- What is the default geom associated with stat_summary()? How could you rewrite the previous plot to use that geom function instead of the stat function?
# The default geom associated with stat_summary() is 
#> Warning: `fun.y` is deprecated. Use `fun` instead.
#> Warning: `fun.ymin` is deprecated. Use `fun.min` instead.
#> Warning: `fun.ymax` is deprecated. Use `fun.max` instead.

#rewrite the previous plot to use that geom function instead of the stat function
ggplot(data = diamonds,mapping = aes(x = cut, y = depth)) +
  geom_pointrange(
    stat = "summary",
    fun.min = min,
    fun.max = max,
    fun = median
  )

# Q2- What does geom_col() do? How is it different to geom_bar()?
demo <- tribble(
  ~cut,         ~freq,
  "Fair",       1610,
  "Good",       4906,
  "Very Good",  12082,
  "Premium",    13791,
  "Ideal",      21551
)
ggplot(data = demo) +
  geom_bar(mapping = aes(x = cut, y = freq), stat="identity")

ggplot(data = demo) +
  geom_col(mapping = aes(x = cut, y = freq))

# geom_col draws a bar chart and requires axes x and y. The different than geom_bar() is that I don't need to call stat = identity.
# geom_bar requires axes x on general. I must call a stat if I assumed y.

# Q3- Most geoms and stats come in pairs that are almost always used in concert. Read through the documentation and make a list of all the pairs. What do they have in common?
#   geom                  | stat                  
#  ===========================================================
#  geom_bar()             | stat_count()          
#  geom_boxplot()         | stat_boxplot()        
#  geom_count()           | stat_sum()            
#  geom_smooth()          | stat_smooth()         

# Q4- What variables does stat_smooth() compute? What parameters control its behaviour?
# stat_smooth() compute x, y and method.
# Smoothers fit a model to our data and then plot predictions from the model.
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point() +
  stat_smooth(method = "lm", formula = y ~ poly(x, 3),level = 0.95, n= 234) 
?stat_smooth
# The parameters control its behaviour are formula, method, level, and n. 

# Q5- In our proportion bar chart, we need to set group = 1. Why? In other words what is the problem with these two graphs?
ggplot(data = diamonds) + 
  geom_bar(mapping = aes(x = cut, y = stat(prop),fill = color, group = 1))
ggplot(data = diamonds) + 
  geom_bar(mapping = aes(x = cut, y = stat(prop)))
ggplot(data = diamonds) + 
  geom_bar(mapping = aes(x = cut, y = stat(prop), fill = color))
# If we remove proportion, then cut proportion will count as 1 for all kinds
# If we add fill=color, the color will be the category to cut and cut proportion will count as 1.

#==================================================
# 3.8.1 Exercises
#==================================================
# Q1- What is the problem with this plot? How could you improve it?
ggplot(data = mpg, mapping = aes(x = cty, y = hwy)) + 
  geom_point()

# The problem is overplotting. The values of hwy and cty are rounded so the points appear on a grid and many points overlap each other.
# It can improve by adding position = "jitter".
ggplot(data = mpg, mapping = aes(x = cty, y = hwy)) + 
    geom_point(position = "jitter")
  
# Q2- What parameters to geom_jitter() control the amount of jittering?
?geom_jitter()    
ggplot(data = mpg, mapping = aes(x = cty, y = hwy)) + 
  geom_jitter()
# parameters to geom_jitter() control the amount of jittering, are position, stat, width, and height. 

# Q3- Compare and contrast geom_jitter() with geom_count().
ggplot(data = mpg, mapping = aes(x = cty, y = hwy)) + 
  geom_jitter(position = "jitter")
?geom_count()
ggplot(data = mpg, mapping = aes(x = cty, y = hwy)) + 
  geom_count(position = "jitter")
# geom_count re-count the point by re-sizing and re-categorizing them to remove the overlapping problem. 
# geom_jitter spread the point vertically and horizontally to remove the overlapping problem.

# Q4- What’s the default position adjustment for geom_boxplot()? Create a visualisation of the mpg dataset that demonstrates it.
ggplot(data = mpg,mapping = aes(x = cty, y = hwy)) +
  geom_boxplot(mapping = aes(color = drv))
  
