# --------------------------------------------------
# Arwa Ashi - Homework 3 week 5 - Oct 14, 2020
# --------------------------------------------------
# HomeWork:
# Build a dashboard for Memoryless Bar where they can change parameters for their ordering 
# and see the effect it has on inventory over the long run.

# --------------------------------------------------
# Library
# --------------------------------------------------
#install.packages("shinydashboard")
# app.R 
library(shinydashboard)
library(shiny)

# Visualization
library(ggplot2)
library('heemod')
library('diagram')

# A transition matrix 
library(pgirmess)
library(expm)

# --------------------------------------------------
# Markov Chains
# Creating transition matrix
# --------------------------------------------------
T <- matrix(c(0.20, 0.15, 0.65, 0.00,
              0.15, 0.20, 0.65, 0.00,
              0.25, 0.05, 0.65, 0.05,
              0.20, 0.15, 0.60, 0.05), nrow = 4, byrow = TRUE)

# creating column and row names
colnames(T) = c("Keg_0","Keg_1","Keg_2","Keg_3")
rownames(T) = c("Keg_0","Keg_1","Keg_2","Keg_3")
T

# The steady-state values
T%^%20

# --------------------------------------------------
# Dashboard
# Building a Shiny Dashboard for Memoryless Bar
# --------------------------------------------------

# --------------------------------------------------
# UI
# --------------------------------------------------
ui <- dashboardPage(
  
  dashboardHeader(title = "Brewing Company",
                  titleWidth = 260),
  
  
  ## Sidebar content
  # --------------------------------------------------
  dashboardSidebar(
    
    width = 260,
    
    # Also add some custom CSS to make the title background area the same
    # color as the rest of the header.
    tags$head(tags$style(HTML('
                              .skin-blue .main-header .logo {
                              background-color: #3c8dbc;
                              }
                              .skin-blue .main-header .logo:hover {
                              background-color: #3c8dbc;
                              }
                              '))),
    
    # The dynamically-generated user panel
    uiOutput("userpanel"),
    
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard"))
    )
  ),
  
  ## Body content
  # --------------------------------------------------
  dashboardBody(
    tabItems(
      
      # First tab content
      tabItem(tabName = "dashboard",
              fluidRow(
                box(
                  title = "State Space Markov Chains Memoryless Bar",
                  solidHeader = TRUE,
                  status = "warning",
                  #plotOutput
                  plotOutput("plot", brush = "plot_brush"),
                  # Nested HTML tags
                  div(class = "my-class", 
                      p("Changing stages for the ordering and see the effect 
                        it has on inventory over the long run. n = 1 is the first stage and the long run is  n > 20")),
                  #sliderInput
                  sliderInput("n", "n", min = 1, max = 100, value = 1)
                  
                  ),
                
                box(
                  title = "Model Parameters for Markov Chains Memoryless Bar",
                  solidHeader = TRUE,
                  status = "warning",
                  # Nested HTML tags
                  div(class = "my-class", 
                      p("Changing parameters for the ordering and see the effect 
                        it has on inventory over the long run.")),
                  #sliderInput
                  sliderInput("Keg_0", "Keg_0 to Keg_0", min = 0, max = 1, value = 0.2),
                  sliderInput("Keg_1", "Keg_0 to Keg_1", min = 0, max = 1, value = 0.15),
                  sliderInput("Keg_2", "Keg_0 to Keg_2", min = 0, max = 1, value = 0.25),
                  sliderInput("Keg_3", "Keg_0 to Keg_3", min = 0, max = 1, value = 0.20)
                      )
              )
      )
    )
  )
)

# --------------------------------------------------
# Server
# --------------------------------------------------
server <- function(input, output,session) {
  set.seed(122)

  T <- matrix(c(0.20, 0.15, 0.65, 0.00,
                0.15, 0.20, 0.65, 0.00,
                0.25, 0.05, 0.65, 0.05,
                0.20, 0.15, 0.60, 0.05), nrow = 4, byrow = TRUE)
  
  # creating column and row names
  colnames(T) = c("Keg_0","Keg_1","Keg_2","Keg_3")
  rownames(T) = c("Keg_0","Keg_1","Keg_2","Keg_3")
  T
  
  output$plot <- renderPlot(
    res = 40,
    {
    plotmat(round(T%^%input$n,2), pos = c(2, 2), curve = 0.04, name = c("Keg_0","Keg_1","Keg_2","Keg_3"),
            lwd = 1, box.lwd = 1, cex.txt = 0.8, self.cex = 0.5, box.size = 0.09,
             self.shiftx = c(-0.1, 0.1, -0.1, 0.1),
            arr.type = "triangle",
            box.type = "hexa", box.prop = 0.6,
            #main = "Long run steady-state",
            arr.len = 0.8, arr.width = 0.20,
            relsize=0.95)
  }
  )
  
}

shinyApp(ui, server)






