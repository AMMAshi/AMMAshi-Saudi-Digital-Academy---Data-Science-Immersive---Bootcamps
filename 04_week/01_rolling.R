# -------------------------------------------------------------
# Arwa Ashi - Homework 1 - Sep 20, 2020
# -------------------------------------------------------------

# Create a 10 sided dice and a 20 sided dice.
die1<-1:10
die2<-1:20

dice1<- sample(die1,size = 2, replace = TRUE, prob=c(1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10))
dice2<- sample(die2,size = 2, replace = TRUE, prob=c(1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20, 1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20))

prob1<-c(1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10)
prob2<-c(1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20, 1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20)

# Create a function to roll 6 of these dice at a time and calculate the sum
rolling_1<-function(die1,prob1,die2,prob2){
  totalsum <- 0
  for(i in 1:3) {
    dice1 <- sample(die1,size=2,replace = TRUE, prob = prob1)
    dice2 <- sample(die2,size=2,replace = TRUE, prob = prob2)
    #print(dice1)
    #print(sum(dice1))
    #print(dice2)
    #print(sum(dice2))
    #print(sum(dice1)+sum(dice2))
    totalsum <- totalsum + sum(dice1)+sum(dice2)
  }
  a <- c("Total sum of rolling 6 dice of 10 and 20 sided dice at a time",totalsum)
  print(a)
}
rolling_1(die1,prob1,die2,prob2)

# The following function rolling 6 time the two different dice together and sum each individual in each rolling
rolling_2<-function(die1,prob1,die2,prob2){
  for(i in 1:6) {
    dice1<- sample(die1,size=2,replace = TRUE, prob = prob1)
    dice2<- sample(die2,size=2,replace = TRUE, prob = prob2)
    #print(dice1)
    #print(dice2)
    #print(sum(dice1))
    #print(sum(dice2))
  }
}
rolling_2(die1,prob1,die2,prob2)

# Create another function to calculate how many dice rolled more than 6 (for the 10 sided) or 16 (for the 20 sided).
rolling_3<-function(die1,prob1,die2,prob2){
  count<- 0
  for(i in 1:3) {
    dice1<- sample(die1,size=2,replace = TRUE, prob = prob1)
    dice2<- sample(die2,size=2,replace = TRUE, prob = prob2)
    #print(dice1)
    #print(dice2)
    #print(sum(dice1))
    #print(sum(dice2))
    totalrolldice1<-sum(dice1)
    totalrolldice2<-sum(dice2)
    a<- c("Sum rolling number",i,"dice 1",totalrolldice1)
    b<- c("Sum rolling number",i,"dice2",totalrolldice2)
    #print(a)
    #print(b)
    if(any(totalrolldice1 > 6)) {
      count <- count + 1
    }
    if( any(totalrolldice2 > 16)) {
      count <- count + 1
    }
  }
  c <- c("Total dice rolled more than 6 (for the 10 sided) or 16 (for the 20 sided)",count)
  print(c)
}
rolling_3(die1,prob1,die2,prob2)
