# =====================================================
# ِArwa Ashi - HomeWork 2 - Sep 21, 2020
# =====================================================

# =====================================================
# creating a world for card
# =====================================================
deck <- read.csv("~/Desktop/Arwa_Academy/CodingDojo/Data_Science_Immersive/04_week/deck.csv", stringsAsFactors=FALSE)
# ----------------------------------------------------
# Creating backjack
# ----------------------------------------------------
blackjack <- deck
# changing the value of face in backjack king,queen,jack = 10
facecard<-c("king","queen","jack")
blackjack$value[blackjack$face %in% facecard]
blackjack[blackjack$face %in% facecard,]
blackjack$value[blackjack$face %in% facecard]<-10
# changing the value of ace in backjack king,queen,jack = 10
blackjack$value[blackjack$face =="ace"]<-NA
# ----------------------------------------------------
# Creating hearts
# ----------------------------------------------------
hearts <- deck
#everything = 0
hearts$value <- 0
# only hearts = 1
hearts$value[hearts$suit == "hearts"]<-1
# spades queen = 13
hearts$value[hearts$suit == "spades" & hearts$face=="queen"]<-13

# =====================================================
# write a couple of custom functions to shuffle 
# a playing deck and deal the necessary cards 
# for 2 players (and a dealer in the case of blackjack).
# =====================================================

# ----------------------------------------------------
# Write custom functions for shuffling the deck
# ----------------------------------------------------
Shuffling<-function(deck=deck){
  cards<-sample(nrow(deck))
  ShuffledDeck <- deck[cards, ]
  return(ShuffledDeck)
}
ShuffledDeck <- Shuffling(deck=deck)
ShuffledDeck

# ----------------------------------------------------
# Selecting a card for the following games
# ----------------------------------------------------
deal<-function(deck=deck){
  # shuffling
  deckcards<-sample(nrow(deck))
  ShuffledDeck <- deck[deckcards, ]
  
  # selecting cards from the first row
  gamecards<-sample(nrow(ShuffledDeck))
  player_card <- ShuffledDeck[gamecards[1:1],]
  print(player_card)
}

# ----------------------------------------------------
# Write a custom function for dealing cards to 2 players in hearts
# ----------------------------------------------------
# hearts game Playing Rules for 2 players
# https://www.youtube.com/watch?v=UWXXi-KIYnQ
# 1- removing three, five, seven, nine, jack, king from all suits
# 2- set a card for a widow
# 3- give each player 13 cards
# 4- remaining card for a widow
HeartsGame_TwoPlayers<-function(deck=hearts){
  # hands table
  hands<-c()
  
  # 1- removing three, five, seven, nine, jack, king from all suits
  deck<- subset(deck, face!= "three")
  deck<- subset(deck, face!= "five")
  deck<- subset(deck, face!= "seven")
  deck<- subset(deck, face!= "nine")
  deck<- subset(deck, face!= "jack")
  deck<- subset(deck, face!= "king")
  
  # 2- set a card for a widow
  newcard<-deal(deck=deck)
  newcard
  deck <- subset(deck, as.integer(rownames(deck)) != as.integer(rownames(newcard)))
  newcard$player <- "widow" 
  hands<-rbind(hands,newcard)
  hearts_list<- list("hands" = hands,"deck" = deck)
  
  # 3- give each player 13 cards
  for(i in 1:13){
    # First selecting 
    newcard<-deal(deck=deck)
    newcard
    deck <- subset(deck, as.integer(rownames(deck)) != as.integer(rownames(newcard)))
    newcard$player <- "player1" 
    hands<-rbind(hands,newcard)
    hearts_list<- list("hands" = hands,"deck" = deck)
    deck<- deck
    
    # second selecting 
    newcard<-deal(deck=deck)
    newcard
    deck <- subset(deck, as.integer(rownames(deck)) != as.integer(rownames(newcard)))
    newcard$player <- "player2" 
    hands<-rbind(hands,newcard)
    hearts_list<- list("hands" = hands,"deck" = deck)
    deck<- deck
  }
  
  # 4- remaining card for a widow
  newcard<-deal(deck=deck)
  newcard
  deck <- subset(deck, as.integer(rownames(deck)) != as.integer(rownames(newcard)))
  newcard$player <- "widow" 
  hands<-rbind(hands,newcard)
  hearts_list<- list("hands" = hands,"deck" = deck)
  deck<- deck
  
  hearts_list<- list("hands" = hands,"deck" = deck)
  return(hearts_list)
}

# creating a list of hands and remaining cards 
heartsGame_list <-HeartsGame_TwoPlayers(deck=hearts)

# hearts game hands table
HeartsGame_hands<-heartsGame_list$hands
HeartsGame_hands

# updating the hearts card deck
hearts<- heartsGame_list$deck

# ----------------------------------------------------
# write a custom function for dealing cards to 2 players and a dealer in blackjack
# ----------------------------------------------------
# Black Jack game Playing Rules for 2 players
# 1- give each players and the dealer 2 cards.
BLJGame_TwoPlayers<-function(deck=blackjack){
  # hands table
  hands<-c()
  
  # 1- give each players and the dealer 2 cards.
  for(i in 1:2){
    # First selecting 
    newcard<-deal(deck=deck)
    newcard
    deck <- subset(deck, as.integer(rownames(deck)) != as.integer(rownames(newcard)))
    newcard$player <- "player1" 
    hands<-rbind(hands,newcard)
    BLJ_list<- list("hands" = hands,"deck" = deck)
    
    # second selecting 
    newcard<-deal(deck=deck)
    newcard
    deck <- subset(deck, as.integer(rownames(deck)) != as.integer(rownames(newcard)))
    newcard$player <- "player2" 
    hands<-rbind(hands,newcard)
    BLJ_list<- list("hands" = hands,"deck" = deck)
    
    # third selecting 
    newcard<-deal(deck=deck)
    newcard
    deck <- subset(deck, as.integer(rownames(deck)) != as.integer(rownames(newcard)))
    newcard$player <- "dealer" 
    hands<-rbind(hands,newcard)
    BLJ_list<- list("hands" = hands,"deck" = deck)
  }
  BLJ_list<- list("hands" = hands,"deck" = deck)
  return(BLJ_list) 
}

# creating a list of hands and remaining cards
BLJGame_list <-BLJGame_TwoPlayers(deck=blackjack)

# blackjack game hands table
BLJGame_hands<-BLJGame_list$hands
BLJGame_hands 

# updating the blackjack card deck
blackjack <-BLJGame_list$deck 

