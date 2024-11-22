library(baseballr)
library(tidyverse)
library(ggplot2)
library(DescTools)
library(ggthemes)
library(pROC)
library(mgcv)

game_info <- get_game_info_sup_petti()
combinedData <- mlb_pbp(game_pk = 718700)
for(i in 1:500) {
  print(i)
  combinedData <- rbind(combinedData, mlb_pbp(game_pk = game_info$game_pk[i]), fill = TRUE)
}


# Filter data to only include pitches
# Create new variables
# same_hand (whether the matchup was L/L or R/R), 1 for same hand 0 for different
# Swing (0 for non-swing, 1 for swing)
# Select necessary variables for model

filtered_Data <- combinedData %>%
  filter(isPitch == "TRUE" & (details.description %in% c("Ball","Called Strike", "Ball In Dirt")) == FALSE) %>%
  mutate(same_hand = if_else(matchup.batSide.code == matchup.pitchHand.code, 1, 0)) %>%
  mutate(whiff =  if_else(details.description %in% c("Swinging Strike","Swinging Strike (Blocked)"), 1, 0)) %>%
  mutate(count.balls.start = if_else(details.description %in% c('Ball', "Ball In Dirt", "Hit By Pitch"), count.balls.start - 1, count.balls.start)) %>%
  mutate(count.strikes.start = if_else(details.description %in% c('Swinging Strike', "Foul", "Called Strike", "Swinging Strike (Blocked)", "Foul Tip"),
                                       count.strikes.start - 1, count.strikes.start)) %>%
  select(whiff, count.balls.start, count.strikes.start,pitchData.startSpeed, pitchData.extension,
         pitchData.coordinates.pX, pitchData.coordinates.pZ, pitchData.breaks.spinRate,
         pitchData.breaks.spinDirection, pitchData.breaks.breakVerticalInduced,
         pitchData.breaks.breakHorizontal, matchup.pitchHand.code, same_hand, matchup.batSide.code, details.type.code, pitchData.strikeZoneBottom,
         pitchData.strikeZoneTop,details.type.code,details.description) 

### Filter Data to create a DataFrame of only four seam fastballs
four_seam_whiff_Data <- filtered_Data %>%
  filter(
    details.type.code == "FF",
    (pitchData.coordinates.pX > -.83 & pitchData.coordinates.pX < .83) & (pitchData.coordinates.pZ < pitchData.strikeZoneTop & pitchData.coordinates.pZ > pitchData.strikeZoneBottom)) %>%
  mutate(height_above_zone = pitchData.coordinates.pZ - pitchData.strikeZoneTop)

four_seam_whiff_Data <- four_seam_whiff_Data %>%
  mutate(pitchData.coordinates.pX = if_else(matchup.batSide.code == "L", pitchData.coordinates.pX * -1, pitchData.coordinates.pX))

### Filter Data to create a DataFrame of Right Handed four seam fastballs
RHP_data <- four_seam_whiff_Data %>%
  mutate(pitchData.breaks.breakHorizontal = 
           if_else(matchup.pitchHand.code == "R", pitchData.breaks.breakHorizontal, (-1 * pitchData.breaks.breakHorizontal))
  )

### Split Data into training Data

set.seed(8)

number_rows <- nrow(RHP_data)
order_rows <- sample(number_rows)
RHP_data <- RHP_data[order_rows,]

train_FF_whiff <- RHP_data[1:24825] %>%
  select(pitchData.coordinates.pX,
         height_above_zone,
         pitchData.breaks.breakVerticalInduced,
         pitchData.breaks.breakHorizontal,
         pitchData.startSpeed,
         pitchData.extension,
         whiff,
  )

train_FF_whiff <- na.omit(train_FF_whiff)

### Split remaining data into testing Data

test_FF_whiff <- RHP_data[24826:33100] %>%
  select(pitchData.coordinates.pX,
         height_above_zone,
         pitchData.breaks.breakVerticalInduced,
         pitchData.breaks.breakHorizontal,
         pitchData.startSpeed,
         pitchData.extension,
         whiff
  )
test_FF_whiff <- na.omit(test_FF_whiff)

### Train a GAM on training Data
Whiff_GAM <- gam(whiff ~ s(pitchData.coordinates.pX) + s(height_above_zone,pitchData.breaks.breakVerticalInduced) +
                   s(pitchData.breaks.breakHorizontal) + s(pitchData.startSpeed, pitchData.extension),
                 data = train_FF_whiff, method = 'REML', family = binomial)

test_FF_whiff$xwhiff <- predict(Whiff_GAM, type = "response", newdata = test_FF_whiff)

### Test the efficacy of the model
mean((test_FF_whiff$xwhiff - test_FF_whiff$whiff)^2)

### Produce an ROC curve
par(pty = "s")
roc(data = test_FF_whiff, whiff, xwhiff, plot = TRUE, percent = TRUE,
    xlab = "False Positive Percentage", ylab = "True Positive")


### Test training Data to see if the probabilities match up between xwhiff and the true probability of a whiff
probability_data_set <- data.frame(interval_upper = 0, mean_xwhiff = 0, mean_whiff = 0)

p = .1
while(p < .6) {
  probability_data_set <- rbind(probability_data_set, c(p, mean(subset(test_FF_whiff, xwhiff < p & xwhiff > (p - .1))$xwhiff), mean(subset(test_FF_whiff, xwhiff < p & xwhiff > (p - .1))$whiff)))
  p = p + .05
}
ggplot(
  data = probability_data_set,
  aes(x = mean_xwhiff, y = mean_whiff)
) + geom_point() +
  labs(x = "Mean xWhiff between intervals (0-.05,.1-.15,.2-.25 ...)",
       y = "Mean true whiff rate between intervals (0-.05,.1-.15,.2-.25 ...)",
       title = "Mean xWhiff over an interval with length .05 is highly similar to the true whiff proportion over the same interval") + 
  theme(aspect.ratio = 1) +
  geom_abline(slope = 1, intercept = 0, colour = "blue")

### Function to produce a plot between predicted xWhiff and the predictor variable or predictor variables
variable_plots <- function(variable) {
  ggplot(
    data = test_FF_whiff,
    aes(x = {{variable}}, y = xwhiff)
  ) +
    geom_point() + geom_smooth()
}
variable_plots(pitchData.breaks.breakHorizontal)

means_data_set <- data.frame(height_above_zone = 0, mean_whiff = 0, mean_xwhiff = 0)

i = -2.0
while(i < 0) { 
  means_data_set <- rbind(means_data_set, c(i, mean(subset(test_FF_whiff, height_above_zone > i )$whiff), mean(subset(test_FF_whiff, height_above_zone > i)$xwhiff)))
  i = i + .1
}
means_data_set <- means_data_set[-1,]

means_data_set <- melt(means_data_set, id.vars = "height_above_zone", variable.name = "type")

ggplot(
  data = means_data_set,
  aes(height_above_zone, value)) +
  geom_point(aes(colour = type)) +
  geom_smooth(aes(colour = type))

ggplot(
  data = means_data_set,
  aes(x = height_above_zone, y = mean_xwhiff)
) + 
  geom_point() + geom_smooth()


simulated_data <- function(IVB, Horizontal_Break, Velocity, Extension, name) {
  
  simulation_data <- data.frame(pitchData.coordinates.pX = 0,height_above_zone = 0,
                                pitchData.breaks.breakVerticalInduced = 0,pitchData.breaks.breakHorizontal = 0,
                                pitchData.startSpeed = 0, pitchData.extension = 0)
  i <- -.83
  j <- -2
  while(i < .83) {
    while(j < 0) {
      simulation_data <- rbind(simulation_data, c(i,j,IVB,Horizontal_Break,Velocity,Extension))
      j = j +.25
    }
    j = -2
    i = i +.25
  }
  simulation_data <- simulation_data[-1,]
  
  simulation_data$xwhiff <- predict(Whiff_GAM, newdata = simulation_data, type = "response")
  
  plot <- ggplot(
    data = simulation_data,
    aes(x = pitchData.coordinates.pX, y = height_above_zone,
        fill = xwhiff)
  ) + 
    labs(x = "Horizontal Deviation from Center of Home Plate",
         y = "Height Below the Top of the Zone",
         title = paste("xwhiff heatmap for", "",name)) + 
    geom_tile() +
    scale_fill_gradient(low ="lightblue", high = "red", limits = c(0,.5))
  print(plot)
  return(data.frame(xWhiff_top_of_zone = mean(subset(simulation_data, height_above_zone > -.5)$xwhiff)))
}

print(simulated_data(20.5,6.8,99.5,6.6,"Félix Bautista"))
