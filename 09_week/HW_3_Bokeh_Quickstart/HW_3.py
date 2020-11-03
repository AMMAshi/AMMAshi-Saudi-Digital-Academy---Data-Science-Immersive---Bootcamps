# ========================================================================================
# Arwa Ashi - HW3 - Week 9 - Nov 3rd, 2020 - Saudi Digital Academy
# ========================================================================================

# 05. Presentation and Layout
# ----------------------------------------------------------------------------------------
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.plotting import output_file

# Rows 
# ----------------------------------------------------------------------------------------
from bokeh.layouts import row
output_file("result.html")


x = list(range(11))
y0, y1, y2 = x, [10-i for i in x], [abs(i-5) for i in x]

# create a new plot
s1 = figure(width=250, plot_height=250)
s1.circle(x, y0, size=10, color="navy", alpha=0.5)

# create another one
s2 = figure(width=250, height=250)
s2.triangle(x, y1, size=10, color="firebrick", alpha=0.5)

# create and another
s3 = figure(width=250, height=250)
s3.square(x, y2, size=10, color="olive", alpha=0.5)

# show the results in a row
show(row(s1, s2, s3))


# EXERCISE: use column to arrange a few plots vertically (don't forget to import column)
# ----------------------------------------------------------------------------------------
from bokeh.layouts import column
output_file("result2.html")

# create a new plot
s1 = figure(width=250, plot_height=250)
s1.circle(x, y0, size=10, color="navy", alpha=0.5)

# create another one
s2 = figure(width=250, height=250)
s2.triangle(x, y1, size=10, color="firebrick", alpha=0.5)

# create and another
s3 = figure(width=250, height=250)
s3.square(x, y2, size=10, color="olive", alpha=0.5)

# show the results in a column
show(column(s1, s2, s3))

# Grid plots
# ----------------------------------------------------------------------------------------
from bokeh.layouts import gridplot
output_file("result3.html")

# create a new plot
s1 = figure(width=250, plot_height=250)
s1.circle(x, y0, size=10, color="navy", alpha=0.5)

# create another one
s2 = figure(width=250, height=250)
s2.triangle(x, y1, size=10, color="firebrick", alpha=0.5)

# create and another
s3 = figure(width=250, height=250)
s3.square(x, y2, size=10, color="olive", alpha=0.5)

# put all the plots in a gridplot
p = gridplot([[s1, s2], [s3, None]], toolbar_location=None)

# show the results
show(p)

# EXERCISE: create a gridplot of your own
# ----------------------------------------------------------------------------------------
from bokeh.layouts import gridplot
output_file("result4.html")

# create a new plot
s1 = figure(width=250, plot_height=250)
s1.circle(x, y0, size=10, color="navy", alpha=0.5)

# create another one
s2 = figure(width=250, height=250)
s2.triangle(x, y1, size=10, color="firebrick", alpha=0.5)

# create and another
s3 = figure(width=250, height=250)
s3.square(x, y2, size=10, color="olive", alpha=0.5)

# put all the plots in a gridplot
p = gridplot([[s1], [s2, s3, None]], toolbar_location=None)

# show the results
show(p)















