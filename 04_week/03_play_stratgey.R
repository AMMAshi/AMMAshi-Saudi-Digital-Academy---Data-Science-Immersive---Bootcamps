# =====================================================
# ِARWA ASHI - HOMEWORK 3 - SEP 23, 2020
# =====================================================

# >> PLEASE run homework 2 's code then this code <<

# -----------------------------------------------------
# Building BlackJack Player strategies!
# -----------------------------------------------------
# -----------------------------------------------------
# The game goal is to beat the dealer
# Beating the dealer by getting closer to 21 than the dealer
# Without going over 21
# -----------------------------------------------------

# SELECTING AND UPDATING CARDS FUNCTION
deal<-function(currentdeck){
  card<-currentdeck[1,]
  assign("blackjack", currentdeck[-1,],envir = globalenv())
  card
}

# ORIGINAL SCORE 
hands <- BLJGame_hands
dealer_score  <- sum(BLJGame_hands$value[BLJGame_hands$player=="dealer"],na.rm = TRUE)
player1_score <- sum(BLJGame_hands$value[BLJGame_hands$player=="player1"],na.rm = TRUE)
player2_score <- sum(BLJGame_hands$value[BLJGame_hands$player=="player2"],na.rm = TRUE)

# CREATING FACE CARD FOR BLACKJACK 
facecard<-c("king","queen","jack","ten")

# -----------------------------------------------------
# Strategy for the player1 
# -----------------------------------------------------
# NO MORE ACTION BLACKJACK
if(any(hands$player[hands$face == "ace"] == "player1")){
  if(any(hands$player[hands$face %in% facecard] == "player1")){
    if(any(hands$player[hands$face == "ace"] != "dealer") && any(hands$player[hands$face %in% facecard] != "dealer")){
      print("PLAYER 1 WIN NO MORE ACTION")}
  }
}

# UPDATING ACE BEFORE HITTING
if(player1_score > 10){
  hands$value[hands$player == "player1" & hands$face=="ace"]<- 1
}else{
  hands$value[hands$player == "player1" & hands$face=="ace"]<- 11
}
player1_score <- sum(hands$value[hands$player=="player1"],na.rm = TRUE)

#HARD TOTALS EXCLUDING PAIRS
if(hands$face[hands$player == "dealer"][2] != "ace"){
  if(player1_score < 16 & hands$value[hands$player == "dealer"][2]>=7){
    while(player1_score < 16){
      newcard<-deal(blackjack)
      newcard$player<- "player1"
      hands<-rbind(hands,newcard)
      # UPDATING ACE AFTER HITTING
      if(sum(hands$value[hands$player == "player1"],na.rm = TRUE)>10){
        hands$value[hands$player == "player1" & hands$face=="ace"]<- 1
      }else{
        hands$value[hands$player == "player1" & hands$face=="ace"]<- 11
      }
      player1_score <- sum(hands$value[hands$player=="player1"],na.rm = TRUE)
    }
  }
}

# UPDATING SCORE PLAYER !
player1_score <- sum(hands$value[hands$player=="player1"],na.rm = TRUE)

# -----------------------------------------------------
# Strategy for the player2
# -----------------------------------------------------
# NO MORE ACTION BLACKJACK
if(any(hands$player[hands$face == "ace"] == "player2")){
  if(any(hands$player[hands$face %in% facecard] == "player2")){
    if(any(hands$player[hands$face == "ace"] != "dealer") && any(hands$player[hands$face %in% facecard] != "dealer")){
      print("PLAYER 2 WIN NO MORE ACTION")}
  }
}
# UPDATING ACE BEFORE HITTING 
if(player2_score>10){
    hands$value[hands$player == "player2" & hands$face=="ace"]<- 1
}else{
    hands$value[hands$player == "player2" & hands$face=="ace"]<- 11 
}
player2_score <- sum(hands$value[hands$player=="player2"],na.rm = TRUE)

# HIT OR SPLIT OR STAND
if(hands$face[hands$player == "player2"][1] != hands$face[hands$player == "player2"][2]){
  #HARD TOTALS EXCLUDING PAIRS
  if(hands$face[hands$player == "dealer"][2] != "ace"){
    if(player2_score < 16 & hands$value[hands$player == "dealer"][2]>=7){
      while(player2_score < 16){
        newcard<-deal(blackjack)
        newcard$player<- "player2"
        hands<-rbind(hands,newcard)
        # UPDATING ACE AFTER HITTING
        if(sum(hands$value[hands$player == "player2"],na.rm = TRUE)>10){
          hands$value[hands$player == "player2" & hands$face=="ace"]<- 1
        }else{
          hands$value[hands$player == "player2" & hands$face=="ace"]<- 11
        }
        player2_score <- sum(hands$value[hands$player=="player2"],na.rm = TRUE)
      }
    }
  }
}else{
  # PAIRS
  if(any(hands$face[hands$player == "player2" ] != "ten")){
    if(hands$face[hands$player == "dealer"][2] != "ace"){
      if(hands$value[hands$player == "dealer"][2]>=7){
        print(" PAIR BUT HIT")
        while(player2_score < 16){
          newcard<-deal(blackjack)
          newcard$player<- "player2"
          hands<-rbind(hands,newcard)
          # UPDATING ACE AFTER HITTING
          if(sum(hands$value[hands$player == "player2"],na.rm = TRUE)>10){
            hands$value[hands$player == "player2" & hands$face=="ace"]<- 1
          }else{
            hands$value[hands$player == "player2" & hands$face=="ace"]<- 11
          }
          player2_score <- sum(hands$value[hands$player=="player2"],na.rm = TRUE)
        }
      }else{
        #SPLIT
        print("PAIR SPLIT")
        # [1] 
        newcard<-deal(blackjack)
        newcard$player<- "player2"
        hands<-rbind(hands,newcard)
        # UPDATING ACE AFTER HITTING
        if(sum(hands$value[hands$player == "player2"],na.rm = TRUE)>10){
          hands$value[hands$player == "player2" & hands$face=="ace"]<- 1
        }else{
          hands$value[hands$player == "player2" & hands$face=="ace"]<- 11
        }
        score1_player2_split1<-score_hand(hands[hands$player == "player2",])-hands$value[hands$player == "player2"][2]
        # [2]
        newcard<-deal(blackjack)
        newcard$player<- "player2"
        hands<-rbind(hands,newcard)
        # UPDATING ACE AFTER HITTING
        if(sum(hands$value[hands$player == "player2"],na.rm = TRUE)>10){
          hands$value[hands$player == "player2" & hands$face=="ace"]<- 1
        }else{
          hands$value[hands$player == "player2" & hands$face=="ace"]<- 11
        }
        score1_player2_split2<-score_hand(hands[hands$player == "player2",])-hands$value[hands$player == "player2"][1]-hands$value[hands$player == "player2"][3]
      }
    }
  }
}

#UPDATING SCORE 
player2_score <- sum(hands$value[hands$player=="player2"],na.rm = TRUE)
# -----------------------------------------------------
# Strategy for the dealer
# -----------------------------------------------------
if(any(hands$player[hands$face == "ace"] == "dealer") && any(hands$player[hands$face %in% facecard] == "dealer")){
  if(any(hands$player[hands$face == "ace"] != "player1") && any(hands$player[hands$face %in% facecard] != "player1")){
    print("Dealer win No more action")
  }else if(any(hands$player[hands$face == "ace"] != "player2") && any(hands$player[hands$face %in% facecard] != "player2")){
    print("Dealer win No more action")
  }
}else{
  # UPDATING ACE BEFORE HITTING
  if (sum(hands$value[hands$player == "dealer"],na.rm = TRUE)>10){
    hands$value[hands$player == "dealer" & hands$face=="ace"]<- 1
  }else{
    hands$value[hands$player == "dealer" & hands$face=="ace"]<- 11
  }
  dealer_score <- sum(hands$value[hands$player=="dealer"],na.rm = TRUE)
  while(dealer_score < 17 ){
    newcard<-deal(blackjack)
    newcard$player<- "dealer"
    hands<-rbind(hands,newcard)
    # UPDATING ACE AFTER HITTING
    if (sum(hands$value[hands$player == "dealer"],na.rm = TRUE)>10){
      hands$value[hands$player == "dealer" & hands$face=="ace"]<- 1
    }else{
      hands$value[hands$player == "dealer" & hands$face=="ace"]<- 11
    }
    dealer_score <- sum(hands$value[hands$player=="dealer"],na.rm = TRUE)
  }
}



dealer_score <- sum(hands$value[hands$player=="dealer"],na.rm = TRUE)


