#==================================================
# Arwa Ashi - Weekend week 5 - Oct 8, 2020
#==================================================
# Munging Project - Optional
# Check out Piazza for more information.
# Pick a data set that is not clean and go through the process from beginning to end until it is tidy, 
# Then perform EDA on that data set and tell us what you learned. The following websites contain links 
# to several not tidy data sets, but feel free to find something that you like to work with 
# (You can work in teams of 2 for this project).

#==================================================
# Munging Project - Methodology
#==================================================
# 1-  using data sets from https://data.gov.sa/Data/en/organization/saudi_customs_authority?page=2
# 2-  translating data from Arabic into english
# 3-  creating tables data frame if needed
# 4-  performing EDA and creating tidy data
# 5-  connecting tables
# 6-  Finding an interesting business questions! 

#==================================================
# Using data sets from https://data.gov.sa/Data/en/organization/saudi_customs_authority?page=2
#==================================================
# Data sets are:
# 1- exports_by_country_2019 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 2- exports-by-product_2019 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 3- import_by_country_2019 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 4- import_by_product_2019 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 5- total-exports-value-by-country-of-destination-2008-to-2018 <<<<<
# 6- total-imports-value-by-country-of-origin-2008-to-2018 <<<<<<<<<<
# 7- total-imports-value-by-port-type-2008-to-2018 <<<<<<<<<<<<<<<<<<
# 8- total-Exports-value-by-port-type-2008-to-2018 <<<<<<<<<<<<<<<<<<
# 9- total-imports-value-by-product-2008-to-2018 <<<<<<<<<<<<<<<<<<<<
#==================================================
# Using https://developers.google.com/public-data/docs/canonical/countries_csv for map plot
#==================================================
# 10- world_country_coordinates <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#==================================================
#  Translating data from Arabic into english
#==================================================
# Using Excel translate data from Arabic into english

#==================================================
#  Creating tables data frame if needed
#==================================================
# data frame were created in excel manually

#==================================================
#  Performing EDA and creating tidy data
#==================================================
library(tidyverse)
library(lubridate)
library(scales)
library(ggrepel)

#--------------------------------------------------
# Reading the data set csv files
#--------------------------------------------------
ExportsCountry2019<- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/exports_by_country_2019.csv", 
                     col_types = cols(The_country_of_destination= col_character(),
                                      Value_million_riyals = col_double(),
                                      Weight_tons = col_double()))
ImportsCountry2019<- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/imports_by_country_2019.csv", 
                              col_types = cols(The_country_of_destination= col_character(),
                                               Value_million_riyals = col_double(),
                                               Weight_tons = col_double()))
TotalExportsValueByCountry<- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/TotalExportsValueByCountry.csv", 
                              col_types = cols(The_country_of_destination= col_character(),
                                               "2008" = col_double(),"2009" = col_double(),
                                               "2010" = col_double(),"2011" = col_double(),
                                               "2012" = col_double(),"2013" = col_double(),
                                               "2014" = col_double(),"2015" = col_double(),
                                               "2016" = col_double(),"2017" = col_double(),
                                               "2018" = col_double()))
TotalImportsValueByCountry<- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/TotalImportsValueByCountry.csv", 
                                      col_types = cols(The_country_of_destination= col_character(),
                                                       "2008" = col_double(),"2009" = col_double(),
                                                       "2010" = col_double(),"2011" = col_double(),
                                                       "2012" = col_double(),"2013" = col_double(),
                                                       "2014" = col_double(),"2015" = col_double(),
                                                       "2016" = col_double(),"2017" = col_double(),
                                                       "2018" = col_double()))
ExportsProducts2019<- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/exports_by_product_2019.csv", 
                              col_types = cols(Product_type = col_character(),
                                               Value_million_riyals = col_double(),
                                               Weight_tons = col_double()))
ImportsProducts2019<- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/Imports_by_product_2019.csv", 
                               col_types = cols(Product_type = col_character(),
                                                Value_million_riyals = col_double(),
                                                Weight_tons = col_double()))
TotalsImports_by_Products<- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/TotalsImports_by_Products.csv", 
                                      col_types = cols(Product_type= col_character(),
                                                       "2008" = col_double(),"2009" = col_double(),
                                                       "2010" = col_double(),"2011" = col_double(),
                                                       "2012" = col_double(),"2013" = col_double(),
                                                       "2014" = col_double(),"2015" = col_double(),
                                                       "2016" = col_double(),"2017" = col_double(),
                                                       "2018" = col_double()))
TotalsImports_by_Port<- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/TotalsImports_by_Port.csv", 
                                     col_types = cols(Port_type= col_character(),
                                                      "2008" = col_double(),"2009" = col_double(),
                                                      "2010" = col_double(),"2011" = col_double(),
                                                      "2012" = col_double(),"2013" = col_double(),
                                                      "2014" = col_double(),"2015" = col_double(),
                                                      "2016" = col_double(),"2017" = col_double(),
                                                      "2018" = col_double()))
TotalsExports_by_Port <- read_csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/05_week/assignment/project/clean_data/TotalsExports_by_Port.csv", 
                                  col_types = cols(Port_type= col_character(),
                                                   "2008" = col_double(),"2009" = col_double(),
                                                   "2010" = col_double(),"2011" = col_double(),
                                                   "2012" = col_double(),"2013" = col_double(),
                                                   "2014" = col_double(),"2015" = col_double(),
                                                   "2016" = col_double(),"2017" = col_double(),
                                                   "2018" = col_double()))

#--------------------------------------------------
# calling the world data to plot ExportsCountry2019
#--------------------------------------------------
world <- map_data("world")
df<-ExportsCountry2019 %>% filter(!is.na(Latitude))

# The plot shows total exports by country in 2019 on a map.
# Saudi exported to China, UAE, India and Singapore the most.
ggplot() +
  geom_map(
    data = world, map = world,
    aes(long, lat, map_id = region),
    color = "white", fill = "lightgray", size = 0.1
  ) +
  geom_point(
    data = df,
    aes(x=Longitude, y=Latitude, 
        color = Value_million_riyals,
        size  = Value_million_riyals),
    alpha = 0.5,show.legend = T
  )+
  scale_colour_continuous(low = '#4682B4', high = '#ff4040')+
  theme_void() +
  theme(legend.position = "none")+
  labs(title="2019 Exports From Saudi Arabia by Country",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )

#--------------------------------------------------
# Merging Exports by Country 2019 and Imports by Country 2019 
# Performing EDA
# Creating tidy data
#--------------------------------------------------
(ExportsImportsCountry2019 <- ExportsCountry2019 %>%
   mutate(Exports_VMR=Value_million_riyals,Exports_WT = Weight_tons) %>%
   select(The_country_of_destination,Exports_VMR,Exports_WT)%>%
   left_join(ImportsCountry2019, by="The_country_of_destination")%>%
   mutate(Imports_VMR=Value_million_riyals,Imports_WT = Weight_tons)%>%
   select(-Value_million_riyals,-Weight_tons))

ExportsImportsCountry2019 %>%
  filter(Exports_VMR >= 10000)%>%
  select(The_country_of_destination, Exports_WT)%>%
  mutate(The_country_of_destination = fct_reorder(The_country_of_destination, desc(Exports_WT))) %>%
  ggplot(aes(The_country_of_destination,Exports_WT))+
  geom_col(aes(fill=The_country_of_destination),show.legend = F)+
  labs(title="2019 Exports From Saudi Arabia by Country by Weight Tons ",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Result:
# The highest exported Weight tons is to China.

ExportsImportsCountry2019 %>%
  filter(Exports_VMR >= 10000)%>%
  select(The_country_of_destination, Exports_VMR)%>%
  mutate(The_country_of_destination = fct_reorder(The_country_of_destination, desc(Exports_VMR))) %>%
  ggplot(aes(The_country_of_destination,Exports_VMR))+
  geom_col(aes(fill=The_country_of_destination), show.legend = F)+
  labs(title="2019 Exports From Saudi Arabia by Country in Value Million Riyals",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Result:
# The highest exported value in million riyals is to China.

ExportsImportsCountry2019 %>%
  filter(Exports_VMR >= 4000, Imports_VMR >= 4000)%>%
  pivot_longer(c('Exports_VMR','Imports_VMR'), 
               names_to = "action_type",
               values_to ="value_million_riyals")%>%
  select(The_country_of_destination, action_type,value_million_riyals)%>%
  mutate(The_country_of_destination = fct_reorder(The_country_of_destination,(value_million_riyals))) %>%
  ggplot(aes(x=The_country_of_destination,
             y=value_million_riyals, 
             fill=action_type))+
  geom_bar(stat = "identity", position = "dodge", width = 0.5)+
  coord_flip()+
  labs(title="2019 Exports and Imports From To Saudi Arabia by Country in Value SAR M",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )

ExportsImportsCountry2019 %>%
  filter(Exports_VMR >= 4000, Imports_VMR >= 4000)%>%
  pivot_longer(c('Exports_WT','Imports_WT'), 
               names_to = "action_type",
               values_to ="Weight_tons")%>%
  select(The_country_of_destination, action_type, Weight_tons)%>%
  mutate(The_country_of_destination = fct_reorder(The_country_of_destination,(Weight_tons))) %>%
  ggplot(aes(x=The_country_of_destination,
             y=Weight_tons, 
             fill=action_type))+
  geom_bar(stat = "identity", position = "dodge", width = 0.5)+
  coord_flip()+
  labs(title="2019 Exports and Imports From To Saudi Arabia by Country in Weight Tons",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Result:
# exported value in million riyals (EVMR) compared to Imported value in million riyals (IVMR):
# exported Weight tons (EWT) compared to Imported Weight tons (IWT):
# Saudi EVMR & EWT to Singapore is higher than IVMR & IWT from Singapore!

df1<- TotalExportsValueByCountry%>%
  pivot_longer(`2008`:`2018`, names_to = "year", values_to = "Value")%>%
    filter(The_country_of_destination == "China" |
           The_country_of_destination == "United Arab Emirates" |
           The_country_of_destination == "Singapore" |
           The_country_of_destination == "India")

ggplot(df1,aes(x = year, y = Value,group=The_country_of_destination))+
    geom_point(aes(color=The_country_of_destination)) +
    geom_line(aes(color=The_country_of_destination),show.legend = TRUE)+
    guides(color = FALSE) +
    geom_label_repel(df1%>% filter(year==2018),
                     mapping=aes(label = The_country_of_destination),
                     force=1) +
    labs(title="Exports From Saudi Arabia by Country From 2008 to 2018",
         subtitle = "Saudi Digital Academy 2020",
         caption = "Munging Project"
    )

df2<- TotalImportsValueByCountry%>%
  pivot_longer(`2008`:`2018`, names_to = "year", values_to = "Value")%>%
  filter(The_country_of_destination == "China" |
           The_country_of_destination == "United Arab Emirates" |
           The_country_of_destination == "Singapore" |
           The_country_of_destination == "India") 

ggplot(df2, mapping=aes(x = year, y = Value,group=The_country_of_destination))+
  geom_point(mapping = aes(color=The_country_of_destination)) +
  geom_line(mapping = aes(color=The_country_of_destination),show.legend = TRUE)+
  guides(color = FALSE) +
  geom_label_repel(df2%>% filter(year==2018),
                   mapping=aes(label = The_country_of_destination),
                   force=1) +
  labs(title="Imports To Saudi Arabia by Country From 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )

TotalExportsValueByCountry%>%
  left_join(TotalImportsValueByCountry, by="The_country_of_destination")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(The_country_of_destination == "China") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type)) +
  geom_line(aes(color=Action_type),show.legend = TRUE)+
  labs(title="Comparing Exports Imports From To Saudi Arabia to China From 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# What happen in 2015? Why did total imports decrease?

TotalExportsValueByCountry%>%
  left_join(TotalImportsValueByCountry, by="The_country_of_destination")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(The_country_of_destination == "United Arab Emirates") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type)) +
  geom_line(aes(color=Action_type),show.legend = TRUE)+
  labs(title="Comparing Exports Imports From To Saudi Arabia to UAE From 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Total imports and exports were equals in 2010 but in 2018 imports are 4 times higher exports. Why?

TotalExportsValueByCountry%>%
  left_join(TotalImportsValueByCountry, by="The_country_of_destination")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(The_country_of_destination == "Singapore") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type)) +
  geom_line(aes(color=Action_type),show.legend = TRUE)+
  labs(title="Comparing Exports Imports From To Saudi Arabia to Singapore From 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# What happens in 2012, why did exports start decreasing?

TotalExportsValueByCountry%>%
  left_join(TotalImportsValueByCountry, by="The_country_of_destination")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(The_country_of_destination == "India") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type)) +
  geom_line(aes(color=Action_type),show.legend = TRUE)+
  labs(title="Comparing Exports Imports From To Saudi Arabia to India From 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# The gap between total export and import is increasing with time, why?

#--------------------------------------------------
# Merging Exports Products 2019 and Imports products 2019 
# Performing EDA
# Creating tidy data
#--------------------------------------------------
ExportsProducts2019 %>%
  filter(Value_million_riyals >= 9000)%>%
  mutate(Product_type = fct_reorder(Product_type, desc(Value_million_riyals))) %>%
  ggplot(aes(Product_type,Value_million_riyals))+
  geom_col(aes(fill=Product_type),show.legend = F)+
  labs(title="2019 Exports From Saudi Arabia by Products in Value million riyals ",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Plastics and organic chemical products have the highest exporting Value in million riyals.

ExportsProducts2019 %>%
  filter(Weight_tons >= 8000000)%>%
  mutate(Product_type = fct_reorder(Product_type, desc(Weight_tons))) %>%
  ggplot(aes(Product_type,Weight_tons))+
  geom_col(aes(fill=Product_type),show.legend = T)+
  labs(title="2019 Exports From Saudi Arabia by Products by Weight tons ",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# More than 15M of Plastics and organic chemical products' weight tons were exported in 2019.

ImportsProducts2019 %>%
  filter(Value_million_riyals >= 30000)%>%
  mutate(Product_type = fct_reorder(Product_type, desc(Value_million_riyals))) %>%
  ggplot(aes(Product_type,Value_million_riyals))+
  geom_col(aes(fill=Product_type),show.legend = F)+
  labs(title="2019 Imports To Saudi Arabia by Products in Value million riyals ",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Machinery, electrical equipment ...
# Nuclear reactors, boilers, ...
# vehicles, ...
# have the highest imported Value million riyals in 2019. 

ImportsProducts2019 %>%
  filter(Weight_tons >= 10000000)%>%
  mutate(Product_type = fct_reorder(Product_type, desc(Weight_tons))) %>%
  ggplot(aes(Product_type,Weight_tons))+
  geom_col(aes(fill=Product_type),show.legend = F)+
  labs(title="2019 Imports To Saudi Arabia by Products by Weight tons ",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# More than 10M of grain & Iron and steel & Mineral fuels products weight tons were Imported in 2019.

TotalsImports_by_Products%>%
  pivot_longer(`2008`:`2018`, names_to = "year", values_to = "Value")%>%
  filter(Value >= 30000000000 )%>%
  ggplot(aes(x = year, y = Value,group= Product_type))+
  geom_point(aes(color=Product_type),show.legend = F) +
  geom_line(aes(color= Product_type),show.legend = F)+
  labs(title="Imports From Saudi Arabia by Products from 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Articles of iron or cast (steel)
# Machinery, electrical equipment ...
# Nuclear reactors, boilers, ...
# vehicles, ...
# imported the most from 2008 to 2018.

#--------------------------------------------------
# total-imports-value-by-port-type-2008-to-2018
# Performing EDA
# Creating tidy data
#--------------------------------------------------
TotalsImports_by_Port%>%
  pivot_longer(`2008`:`2018`, names_to = "year", values_to = "Value")%>%
  ggplot(aes(x = year, y = Value,group= Port_type))+
  geom_point(aes(color=Port_type),show.legend = T) +
  geom_line(aes(color= Port_type),show.legend = T)+
  labs(title="Imports To Saudi Arabia by Port from 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# The seaport is the most active port to import products from 2008 to 2018.
# why did the capacity in the seaport decrease after 2015?

TotalsExports_by_Port%>%
  pivot_longer(`2008`:`2018`, names_to = "year", values_to = "Value")%>%
  ggplot(aes(x = year, y = Value,group= Port_type))+
  geom_point(aes(color=Port_type),show.legend = T) +
  geom_line(aes(color= Port_type),show.legend = T)+
  labs(title="Exports From Saudi Arabia by Port from 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# The seaport is the most active port to Export products from 2008 to 2018
# why did the capacity in the seaport increase after 2016?

TotalsExports_by_Port%>%
  left_join(TotalsImports_by_Port, by="Port_type")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(Port_type == "Sea port") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type),show.legend = T) +
  geom_line(aes(color=Action_type),show.legend = T)+
  labs(title="Comparing Exports Imports From To Saudi Arabia by Seaport from 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Exporting and importing by using Seaport is quite closer than the other ports, why?

TotalsExports_by_Port%>%
  left_join(TotalsImports_by_Port, by="Port_type")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(Port_type == "railway") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type),show.legend = T) +
  geom_line(aes(color=Action_type),show.legend = TRUE)+
  labs(title="Comparing Exports Imports From To Saudi Arabia by railway from 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# These is a huge gap between total export and import by using the railway, why?

TotalsExports_by_Port%>%
  left_join(TotalsImports_by_Port, by="Port_type")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(Port_type == "Land port") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type),show.legend = T) +
  geom_line(aes(color=Action_type),show.legend = TRUE)+
  labs(title="Comparing Exports Imports From To Saudi Arabia by Land port from 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# Exporting and importing by using Land port were quietly closer in 2009 than the other years, why?

TotalsExports_by_Port%>%
  left_join(TotalsImports_by_Port, by="Port_type")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(Port_type == "Air port") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type),show.legend = T) +
  geom_line(aes(color=Action_type),show.legend = TRUE)+
  labs(title="Comparing Exports Imports From To Saudi Arabia by Air port from 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# The capacity of exporting by using the airport is 4 times less than importing capacity, why?

TotalsExports_by_Port%>%
  left_join(TotalsImports_by_Port, by="Port_type")%>%
  pivot_longer(`2008.x`:`2018.y`, names_to = "year", values_to = "Value")%>%
  separate(year, into = c("year","Action_type"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "x", "Total_Exports"))%>% 
  mutate(Action_type = stringr::str_replace(Action_type, "y", "Total_Imports"))%>% 
  filter(Port_type == "post office") %>%
  ggplot(aes(x = year, y = Value,group=Action_type))+
  geom_point(aes(color=Action_type),show.legend = T) +
  geom_line(aes(color=Action_type),show.legend = TRUE)+
  labs(title="Comparing Exports Imports From To Saudi Arabia by post office from 2008 to 2018",
       subtitle = "Saudi Digital Academy 2020",
       caption = "Munging Project"
  )
# No exporting by using a post office, why?

#--------------------------------------------------
# Using the data to determine if there are differences among mean value for the four
# exporting ports at the 0.5 significance level
# One-Way ANOVA
#--------------------------------------------------
# 1- 
# H_0 : alpha_i = 0 for i = 1,2,3,4
# H_1 : alpha_i != 0 for at least one value of i

TotalsImports_by_Port_STAT <- TotalsImports_by_Port%>%
  pivot_longer(`2008`:`2018`, names_to = "year", values_to = "Value")
TotalsImports_by_Port_STAT$Port_type <- as.factor(TotalsImports_by_Port_STAT$Port_type)
boxplot(Value~Port_type, data = TotalsImports_by_Port_STAT, xlab = "Port Type", ylab="value")

# First fit
fit <- lm(Value~Port_type-1,data=TotalsImports_by_Port_STAT)
summary(fit)
plot(fit)

# calling the mean for each port
by(TotalsImports_by_Port_STAT$Value,TotalsImports_by_Port_STAT$Port_type,mean)
# Second Fit

fit2 <- lm(Value~1,data=TotalsImports_by_Port_STAT)
anova(fit2,fit)
# Df = 4, Sum of squares = SST = 7.0839e+23, F = 51.515 

# The null hypothesis must be rejected, and we conclude that
# the four exported ports are not all equally effective.

# Using the corner Constraint
fit <- lm(Value~Port_type,data=TotalsImports_by_Port_STAT)
summary(fit)

model.matrix(fit)
