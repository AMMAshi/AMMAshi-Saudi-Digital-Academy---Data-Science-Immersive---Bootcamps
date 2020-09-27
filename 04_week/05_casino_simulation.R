# =====================================================
# ِArwa Ashi - HomeWork 5 - Sep 24, 2020
# =====================================================
# QUESTION: Build and simulate the expected profit to the casino of running different games 
# (craps, roulette, slot machines, whichever is your favorite)
# Hint: Stay away from poker since bluffing plays a critical role in the game and it has no effect on computer code

# -----------------------------------------------------
# Rules for Blackjack, Craps, Roulette: 
# -----------------------------------------------------
# player bets $1, if player win he/she gets his/her money back plus $1 #(casino loss $1) 
# if player loses he/she loses $1 # ( casino add one dollar)
# if a tie happens, player recovers her/his money # (gain and loses of zero to the casino)

# Simulate Casino Profits 
# -----------------------------------------------------
# Assuming everyone plays a perfect strategy
# -----------------------------------------------------
n = 10000                            # total hand at the blackJack
winnings<-vector("integer",n)        # 10,000 terms or hand at the blackJack
acc_winngings<-vector("integer",n)   # adding up how much we are making over other hands

# -----------------------------------------------------
# Creating the BLJ Simulation
# -----------------------------------------------------
# One hand of play
numplayers <- 6   
Number_of_decks<- 8

BLJ_Simulation<- function(numplayers=numplayers, Number_of_decks=Number_of_decks){
  BLJ_prob = c(0.493,0.4222,0.0848)
  
  #Creating the Simulation
  for (i in 1:length(winnings)){
    onehand<- sample(c("D","P","T"),numplayers,replace = TRUE,prob = BLJ_prob)
    # Number of decks &	House advantage
    if( Number_of_decks == 1){EV = length(onehand[onehand == "P"])*1*0.0017}
    if( Number_of_decks == 2){EV = length(onehand[onehand == "P"])*1*0.0046}
    if( Number_of_decks == 4){EV = length(onehand[onehand == "P"])*1*0.0060}
    if( Number_of_decks == 6){EV = length(onehand[onehand == "P"])*1*0.0064}
    if( Number_of_decks == 8){EV = length(onehand[onehand == "P"])*1*0.0066}
    # if player win, assuimg player money back plus 100% of original bet
    winnings[i] <- length(onehand[onehand == "P"])*-1 + EV + length(onehand[onehand == "D"])
    
    if(i != 1){
      acc_winngings[i] <- acc_winngings[i-1] + winnings[i]
    } else{
      acc_winngings[i] <- winnings[i]
    }
  }
  #Creating profit list
  profit_list<- list("acc_win" = acc_winngings,"win" = winnings)
  return(profit_list) 
}
# creating a list of accumulated profit and profit
BLJ_win_list <- BLJ_Simulation(numplayers, Number_of_decks)

# the casino accumulated blackjack profit
BLJ_acc_casino_profit<-BLJ_win_list$acc_win
BLJ_acc_casino_profit[n] 

# updating the casino blackjack profit
BLJ_casino_profit <-BLJ_win_list$win

# -----------------------------------------------------
# Creating the Craps Simulation
# -----------------------------------------------------
# https://people.richland.edu/james/misc/simulation/craps.html

Craps_Simulation<- function(numplayers=6){
  Craps_prob = c(0.5070707,0.4929293)
  #Creating the Simulation
  for (i in 1:length(winnings)){
    onehand<- sample(c("D","P"),numplayers,replace = TRUE,prob = Craps_prob)
    # if player win, assuimg player money back plus 100% of original bet
    winnings[i] <- length(onehand[onehand == "P"]) * -1 + length(onehand[onehand == "D"])
    if(i != 1){
      acc_winngings[i] <- acc_winngings[i-1] + winnings[i]
    } else{
      acc_winngings[i] <- winnings[i]
    }
  }
  #Creating profit list
  profit_list<- list("acc_win" = acc_winngings,"win" = winnings)
  return(profit_list) 
}
# creating a list of accumulated profit and profit
Craps_win_list <- Craps_Simulation(numplayers)

# the casino accumulated Craps profit
Craps_acc_casino_profit<-Craps_win_list$acc_win
Craps_acc_casino_profit[n] 

# updating the Craps card deck
Craps_profit <-Craps_win_list$win

# -----------------------------------------------------
# Creating the Roulette Simulation
# -----------------------------------------------------
# https://www.roulettesites.org/rules/odds/#:~:text=To%20be%20precise%2C%20'even%20money,12%20numbers%20on%20the%20table.

Roulette_Simulation<- function(numplayers=6){
  Roulette_prob = c(0.5263,0.4737)
  #Creating the Simulation
  for (i in 1:length(winnings)){
    onehand<- sample(c("D","P"),numplayers,replace = TRUE,prob = Roulette_prob)
    # if player win, assuimg player money back plus 100% of original bet
    winnings[i] <- length(onehand[onehand == "P"])*-1 + length(onehand[onehand == "D"])
    if(i != 1){
      acc_winngings[i] <- acc_winngings[i-1] + winnings[i]
    } else{
      acc_winngings[i] <- winnings[i]
    }
  }
  #Creating profit list
  profit_list<- list("acc_win" = acc_winngings,"win" = winnings)
  return(profit_list) 
}
# creating a list of accumulated profit and profit
Roulette_win_list <- Craps_Simulation(numplayers)

# the casino accumulated Roulette profit
Roulette_acc_casino_profit<-Roulette_win_list$acc_win
Roulette_acc_casino_profit[n] 

# updating the Roulette card deck
Roulette_profit <- Roulette_win_list$win

# -----------------------------------------------------
# Creating the Slot Machine Simulation
# -----------------------------------------------------
# https://www.dummies.com/education/math/using-probability-when-hitting-the-slot-machines/
  
SM_Simulation<- function(numplayers=6){
  SM_prob = c(0.1,0.9)
  #Creating the Simulation
  for (i in 1:length(winnings)){
    onehand<- sample(c("D","P"),numplayers,replace = TRUE,prob = SM_prob)
    # if player win, assuimg player money back plus 0% of original bet & casino gain is 10% of original player bet
    winnings[i] <- length(onehand[onehand == "D"]) + length(onehand[onehand == "P"])*SM_prob[1]
    if(i != 1){
      acc_winngings[i] <- acc_winngings[i-1] + winnings[i]
    } else{
      acc_winngings[i] <- winnings[i]
    }
  }
  #Creating profit list
  profit_list<- list("acc_win" = acc_winngings,"win" = winnings)
  return(profit_list) 
}
# creating a list of accumulated profit and profit
SM_win_list <- SM_Simulation(numplayers)

# the casino accumulated Slot Machine profit
SM_acc_casino_profit<-SM_win_list$acc_win
SM_acc_casino_profit[n] 

# updating the Slot Machine card deck
SM_profit <- SM_win_list$win

# -----------------------------------------------------
# Casino total profit 
# -----------------------------------------------------
Total_profit<- sum(BLJ_casino_profit) + sum(Craps_profit) + sum(Roulette_profit) + sum(SM_profit)







