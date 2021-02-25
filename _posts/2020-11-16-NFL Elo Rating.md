---
title:  "NFL Game Predictions Using Elo Rating"
category: posts
date: 2020-11-16
header:
  teaser: "images/nfl.jfif"
excerpt: The purpose of this project is to create an Elo model to predict which team will win NFL games.
---

| ![PNG](/images/nfl.jfif)|

Link to full [code](https://github.com/twrobbins/Github-Files-Updated/blob/main/DSC680-Applied%20Data%20Science/Project%202/DSC680-Updated_Final_NFL_Prediction.ipynb)


### Abstract
Predicting the winner of sporting events has been an exciting topic for decades.  Ever since the 1950’s, people have attempted to measure the strength of an athlete or team and predict the outcome of such games.  One approach I came across, is the use of the Elo rating system, which is basically a measure of the strength of teams in head-to-head games.  The website FiveThirtyEight.com uses their own version of the Elo rating system to make predictions that often rival Vegas odds.  In this study, I plan to create my own Elo model to predict the winners of NFL games.  

The remainder of this study is organized as follows:  The background around NFL predictions and the Elo rating system is discussed.  I then discuss the methodology, including dataset selection, data preprocessing, and the modeling process.  Finally, the study is concluded with the results and conclusion.  

Keywords: elo, nfl, American football, Super Bowl, playoffs, team ratings, sports forecasting


### Introduction
American football is one of the most popular sports in the United States, and betting on the winner of National Football League (NFL) games is a common practice.  Being able to predict the correct winner of an NFL-game can generate significant cash flow.  The American Gaming Association (AMA) estimated that fans in the United States spent over $90 billion on NFL and college games in the 2017 season.  The popularity of NFL games has led to the vast release of information on games, players, and statistics.  Experts of well-known sports-related organizations, such as ESPN, CBS sports and gambling institutions have made huge investments in the predictions of such games.  Numerous attempts have been made to outperform such experts and gambling institutions using data analytics or machine learning.

A popular approach for predicting the outcome of professional sporting events is done using the Elo rating system.  An Elo rating system is a method to evaluate teams or players in competitive games.  This method was first created and applied to single player games like chess by Arpad Elo, a Hungarian-born American physics professor.  Elo ratings were later used to rank and predict team sports, such as the FIFA World Cup, NFL, MLB, NBA, and others.  Different sports have found differing levels of predictive power from Elo ratings, with the NFL being one sport where Elo ratings have proved successful.  

The process of calculating Elo ratings is straightforward.  The standard method of implementing an Elo rating system begins with setting every team with a starting value, usually 1,500.  All teams start with the same rating.  When teams play each other the expected probability of each team winning can be calculated using the following formula:

| ![PNG](/images/nfl01_win_prob.png)   | 
|:--:| 
| *Figure 1: Home Team Win Probability* |

Ea represents the probability of the home team winning, Rb is the Elo rating of the away team and Ra is the Elo rating of the home team prior to the game.  The probability of the away team is calculated in the same manner, but since the 2 probabilities must sum to 1 (100%), the away team score can be calculated by subtracting the home team expected probability of winning from 1.  The relationship between the ELO difference between the two teams (Rb – Ra) and the probability of winning (Ea) can be visualized with the following figure:

| ![PNG](/images/nfl02_rating_diff.png)   | 
|:--:| 
| *Figure 2: Elo Rating Difference and Probability of Winning* |
 
After the game is played, each team’s new rating is calculated as the difference between their expected score and their actual score using the following formula:


| ![PNG](/images/nfl03_updated_ratings.png)   | 
|:--:| 
| *Figure 3: Updated Elo Rating* |
 
RA is team A’s Elo rating prior to the game, K is the K-factor, SA represents the actual score, and EA the probability of team A winning.  If the home team wins the game, SA is reflected as an outcome of 1.  If the away team wins SA is reflected as a 0.  The K-factor is a constant term that can be adjusted to determine how much each new game changes a team’s Elo rating.  The higher the K, the greater the impact of each individual game on a team’s rating.  Team B’s post-game Elo rating is calculated in the same manner as team A’s. 

As teams play each other throughout the season, ratings will adjust, and teams will begin to separate themselves based on their performance.  Thus, Elo ratings provide a good ranking of each team’s performance over time.  


### Methodology
#### Datasets
Two different data sets were obtained from FiveThirtyEight.com for this project.  First, it was necessary to have starting Elo ratings for each of the 32 teams at the beginning of the 2020 season.  FiveThirtyEight is known for making fairly accurate game predictions, so I used the final 2019 Elo ratings of each team as a starting point for the 2020 season, as shown below:

| ![PNG](/images/nfl04_dataset_1.png)   | 
|:--:| 
| *Figure 4: Dataset 1 – Initial Elo Ratings* |

A density plot of the inital Elo ratings is plotted below:

| ![PNG](/images/nfl05_dist.png)   | 
|:--:| 
| *Figure 5: Density Plot of Initial Elo Ratings* |

The second dataset obtained from FiveThirtyEight consisted of the team matchups for the 2020 regular season, as well as the actual scores through week 7.   There are 16 regular season games for each of the 32 teams, for a total of 512 games.  The head and tail of the dataframe is shown below:

| ![PNG](/images/nfl06_dataset_2.png)   | 
|:--:| 
| *Figure 6: Dataset 2 – Regular Season Schedule* |

After doing some research, I found that team1 represents the home team and team2 the away team.  This will be useful later as history has shown the odds to favor a home team win over an away team win.  score1 and score 2 represent the actual scores of each game.  

### Methodology - Elo Calculation Cycle 
I’m now going to provide you an overview of the Elo calculation cycle.  We start with the pre-game Elo ratings.  For week 1, this is going to be the final Elo ratings from the 2019 season which we have in the 1st dataset we obtained from 538.com.  For subsequent weeks, we’re going to take the previous week’s post-game Elo rating.  

We then account for any adjustments.  In this model, I have an adjustment for home field advantage.  Studies have shown that teams playing at home have a greater probability of winning.  We’re thus going to add 35 points to the home team’s pre-game Elo rating.
We then take these numbers and feed them into formula 1 to calculate each team’s odds of winning.  The formula looks rather complicated, but its’ basically a function of the difference between the two teams’ pre-game Elo ratings.  The team with an Elo rating greater than .50 will be considered the predicted winner.  

We then compare the predicted results with the actual.  If the home team wins, we indicate this with a one.  If the home team loses, we indicate this with a 0.  

We then feed this data into formula 2 to calculate the post-game Elo ratings for each team.  RA is the pre-game Elo rating, to which we add the difference between the actual score, SA, and the team’s probability of winning, EA from formula 1.  We then multiply this difference by k, which represents the K-factor.  The K-factor is a constant we use to determine how much each new game changes the teams’ Elo rating.  The higher the K, the greater the impact of each individual game on a team’s Elo rating.  For this study, I’m going to use a K of 20, which is pretty standard for NFL games.  

Once we have calculated the post-game Elo ratings using formula 2, we carry this forward to the next week’s pre-game Elo rating.  The cycle then repeats itself for each of the games on the schedule in dataset 2.  

If we repeat this process through week 7 of the NFL season, we come up with the following graphs, which show the change in Elo rating for each team, by division, through week 7.  We can use this data to determine who the winner of each division is likely to be and get a taste of what the playoff s will look like.

| ![PNG](/images/nfl07_week7_ratings.png)   | 
|:--:| 
| *Figure 7: Week 7 NFL Rankings* |

As a practical application, we can take the post-game Elo ratings from week 7 and carry them forward to week 8 to make game predictions.  Basically, you just look at the far- right column to see who the predicted winner is and place your bet on that team.  For example, for the Atlanta/Carolina game on the first line, the predicted winner would be Atlanta, as they have the greater probability of winning.

| ![PNG](/images/nfl08_week8_preds.png)   | 
|:--:| 
| *Figure 8: Week 8 NFL Predictions* |

This slide shows the top 12 ranked teams sorted in descending order by Elo rating.  Based on the top Elo ratings through week 7, we can get a picture of who might be in the Super Bowl.  Looks like Kansas City has the highest Elo rating, so they have the best chance of going to the Superbowl and representing the AFC.  Tennessee has the 2nd highest Elo rating, but they are also in the AFC, and there can only be one winner for each conference.  Thus the 3rd ranked team, Green Bay, who is in the NFC, would likely face Kansas City in the Superbowl.  

| ![PNG](/images/nfl09_top_elo.png)   | 
|:--:| 
| *Figure 9: Top Elo Ratings* |

The remaining teams have the highest percentage of filling the 12 playoff spots for the 2020 season.  


### Results                                                           
To evaluate how well my model performed, I calculated a confusion matrix, along with some other useful statistics.  

| ![PNG](/images/nfl10_cm.png)   | 
|:--:| 
| *Figure 10: Confusion Matrix* |




The confusion matrix shows that there were 41 true positives (win was predicted and they won), 25 false positives (win was predicted when they actually lost), 12 false negatives (loss predicted when they actually won), and 27 true negatives (loss predicted and they lost).  This reflects an accuracy of 64.76% for the 2020 season through week 7, which appears to be pretty accurate.

I also calculated a few other statistics, to shed more light on the predicted vs actual results.  

| ![PNG](/images/nfl11_stats.png)   | 
|:--:| 
| *Figure 11: Statistics* |

Sensitivity, also referred to as recall or the true positive rate (TPR), is a measure of the proportion of correct positive results, whereas specificity, also referred to as the true negative rate (TPR) is a measure of the proportion of correct negative results.  The sensitivity of 0.7736 indicates that the model correctly predicted wins at a high rate.  On the other hand, the specificity of 0.5192 indicates that the rate at which the model correctly predicted losses was not too accurate, but better than random guessing.  Based on these statistics, it appears the model is pretty accurate at predicting the proportion of wins, but not very accurate at predicting the proportion of losses.  

The positive and negative predictive values (PPV and NPV) are the proportions of true positive and true negative results.  The PPV of 0.6212 indicates that the model is fairly accurate in predicting actual wins, while the NPV of 0.6923 indicates that the model is more accurate at predicting actual losses


### Conclusion
The Elo model developed for this study provided results very comparably to FiveThirtyEight’s more complex Elo model, which has an accuracy of approximately 65% correct per year.  However, it’s worth noting that the accuracy of the predictions for the model in this study was based on a relatively small population of 105 games and thus a larger sample could lead to different results.
Future work to predict the outcome of NFL games could include more adjustments to the pre-game Elo ratings, such as the adjusting for the quarterback Elo rating, rest time between games, and distance traveled.  In addition, no individual player data was used.  Having top players who have gone to the Pro Bowl and/or were awarded MVP could significantly affect the results.

### References
1. Alvarez, R. (2020). Machine Learning Playoff Predictions: Predicting the Football Greats.
2. Balreira, E. C., & Miceli, B. K. (2019). Improving Foresight Predictions in the 2002-2018 NFL        Regular-Seasons: A Classic Tale of Quantity vs. Quality. Journal of Advances in Mathematics and Computer Science, 1-14.
Bonds-Raacke, J. M., Fryer, L. S., Nicks, S. D., & Durr, R. T. (2001). Hindsight bias demonstrated in the prediction of a sporting event. The Journal of social psychology, 141(3), 349-352.
Boulier, B. L., & Stekler, H. O. (2003). Predicting the outcomes of National Football League games. International Journal of forecasting, 19(2), 257-270.
Bosch, P. (2018). Predicting the winner of NFL-games using Machine and Deep Learning.
Harville, D. (1980). Predictions for National Football League games via linear-model methodology. Journal of the American Statistical Association, 75(371), 516-524.
Klein, J., Frowein, A., & Irwin, C. (2018). Predicting Game Day Outcomes in National Football League Games. SMU Data Science Review, 1(2), 6.
Lee, M. D., Danileiko, I., & Vi, J. (2018). Testing the ability of the surprisingly popular method to predict NFL games. Judgment and Decision Making, 13(4), 322.
Miller, T. W. (2015). Sports analytics and data science: winning the game with methods and models. FT Press.
Pelechrinis, K., & Papalexakis, E. (2016). The anatomy of american football: evidence from 7 years of NFL game data. PLoS one, 11(12), e0168716.
Uzoma, A. O., & Nwachukwu, E. O. (2015). A hybrid prediction system for american NFL results. International Journal of Computer Applications Technology and Research, 4(01), 42-47.
how-our-nfl-predictions-work. (2020, October 15). Retrieved from FiveThirtyEight.com: https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/


