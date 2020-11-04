# ========================================================================================
# Arwa Ashi - HW3 - Week 9 - Nov 3rd, 2020 - Saudi Digital Academy
# ========================================================================================

# Practice creating some plots using the line, scatter, and bar plots you created yesterday in Section 12.
# ----------------------------------------------------------------------------------------
# packages for data
from bokeh.plotting import figure, output_file, show, save, ColumnDataSource
import pandas as pd

# packages for tools
from bokeh.models.tools import HoverTool
from bokeh.models import CDSView, ColumnDataSource, GroupFilter
from bokeh.layouts import gridplot

# packages for color map
from bokeh.transform import factor_cmap, factor_mark

output_file("covid_impact_on_airport_traffic.html")

# Read in csv
df = pd.read_csv('covid_impact_on_airport_traffic.csv')

# Fix date and set it as index
df['Date'] = pd.to_datetime(df['Date']).dt.date
df         = df.set_index('Date')

# Create a ColumnDataSource from data frame
source = ColumnDataSource(df)

COUNTRIES = ['Australia', 'Canada', 'Chile', 'United States of America (the)']
MARKERS   = ['hex', 'circle_x', 'triangle', 'square']

# Add plot
p1 = figure(
    plot_width   = 800,
    plot_height  = 600,
    title        = 'Covid19 Impact on Airport  Traffic - Country with Top Percent of Baseline',
    x_axis_label = 'Date',
    y_axis_label = 'Percent of Baseline',
    tools        = "pan,box_select,zoom_in,zoom_out,save,reset"
)

# Render glyph
p1.scatter(
    x          = 'Date',
    y          = 'PercentOfBaseline',
    size       = 7,
    fill_alpha = 0.4,
    source     = source,
    legend     = 'Country',
    marker     = factor_mark('Country', MARKERS, COUNTRIES),
    color      = factor_cmap('Country', 'Category10_4', COUNTRIES)
)

# Add Lagend
p1.legend.orientation='vertical'
p1.legend.location='top_right'
p1.legend.label_text_font_size='10px'

# Show Results
show(p1)

# Save file
# save(p)
