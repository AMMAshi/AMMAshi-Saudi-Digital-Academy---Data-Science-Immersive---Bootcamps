# =====================================================
# ِArwa Ashi - HomeWork 4 - Sep 24, 2020
# =====================================================

# Simulate Casino Profits and provide some plots on winnings
#-------------------------------------------------------------------------------
# QUESTION: Use all the previous scripts and functions to simulate 10,000 blackjack hands 
# and calculate the profit to the casino!!
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Blackjack Rules: 
#-------------------------------------------------------------------------------
# player bets $1, if player win he/she gets his/her money back plus $1 #(casino loss $1 )
# if player loses he/she loses $1 # (casino add one dollar)
# if a tie happens, player recovers her/his money # (gain and loses of zero to the casino)
# chance of player winning is 0.4222
# chance of tie is 0.0848
# chance of dealer winning is 0.493
#-------------------------------------------------------------------------------
# House edge: 
# Example: the expected value of a $100 for a player's bet long-term is $95 the house egde is 5%
# Example: the expected value of a $1 for a player's bet long-term is $0.9983 the house egde is 0.17% (Single deck)
#-------------------------------------------------------------------------------
# the table is from https://en.wikipedia.org/wiki/Blackjack#Rules_of_play_at_casinos
#-------------------------------------------------------------------------------
# Number of decks |	House advantage | the expected value of a $1 bet long-term
#-------------------------------------------------------------------------------
# Single deck	    |  0.0017         |  $0.9983                    
# Double deck	    |  0.0046         |  $0.9954                    
# Four decks     	|  0.0060         |  $0.994                     
# Six decks	      |  0.0064         |  $0.9936                    
# Eight decks    	|  0.0066         |  $0.9934                     
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Assuming everyone plays a perfect strategy and the long-term start at i = 1
#-------------------------------------------------------------------------------
n = 10000                            # total hand at the blackJack
winnings<-vector("integer",n)        # 10,000 terms or hand at the blackJack
acc_winngings<-vector("integer",n)   # adding up how much we are making over other hands
#-------------------------------------------------------------------------------
# Creating the Simulation
#-------------------------------------------------------------------------------
# One hand of play
numplayers <- 6   
Number_of_decks<-8

BLJ_profit<- function(numplayers=numplayers, Number_of_decks=Number_of_decks){
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
    # updating the profit vectors
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

# creating a list of acc_win and win
win_list <- BLJ_profit(Number_of_decks=Number_of_decks)

# blackjack game hands table
acc_casino_profit<-win_list$acc_win
acc_casino_profit[n] 

# updating the blackjack card deck
casino_profit <-win_list$win
Total_profit<- sum(casino_profit)

#-------------------------------------------------------------------------------
# plots on winnings
#-------------------------------------------------------------------------------
plot(acc_casino_profit)
plot(casino_profit)
hist(acc_casino_profit)
hist(casino_profit)












