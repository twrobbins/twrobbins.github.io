---
title:  "Customer Churn"
category: posts
date: 2020-11-20
header:
  teaser: "images/churn.png"
excerpt: The purpose of this project is to create a model to predict which customers are likely to churn, as well as the causes for the churn.
---

| ![PNG](/images/churn.png)|

Link to full [code](https://github.com/twrobbins/Github-Files-Updated/blob/main/DSC680-Applied%20Data%20Science/Project%203/DSC680-Customer%20Churn-Final.ipynb)


### Abstract
Customer churn refers to the rate at which customers leave a company or service.  Churn could be caused for many different reasons and churn analysis helps to identify the cause and timing of the churn, opening up opportunities to implement effective retention strategies.  A predictive churn model looks at customer activity from the past and checks to see who is active after a certain time and identify the steps and stages when a customer is leaving a service or product.  In this study, I plan to create a predictive churn model to predict which customers are likely to churn, as well as the causes for the churn.

Keywords:  customer churn, customer segmentation, telecommunications, 


### Introduction
Customer churn is the tendency of customers to abandon a brand and stop being a paying client of a particular business.  The percentage of customers that discontinue using a company’s products or services during a particular time period it called customer churn rate.  Churn rate is a health indicator of a busines whose customers are subscribers and paying for services on a recurring basis, such as in the telecommunications industry.  Although some natural churn is inevitable, having a higher churn rate can be a sign that a business is doing something wrong.  

There are many things that companies may do wrong.  For example, if customers aren’t given easy-to-understand information about product usage they may choose to cancel.  In addition, lifetime customers may feel unappreciated because they don’t get as many bonuses as new ones.  In general, it ‘s the overall customer experience that defines brand perception and influences how customers recognize value for money of products and services they use.

Churn rates tend to correlate with a company’s lost revenue and overall growth potential, so keeping churn to a minimum is key.  In addition, customers who have negative experiences with a brand are more likely to share their experiences with other potential customers through review sites or social media.  HubSpot research found that 49% of buyers reported sharing an experience they had with a company on social media.  In a world of eroding trust in business, word of mouth plays a more critical role in the buying process than ever before. 

Underestimating the impact of even a tiny percentage of churn can have a huge impact.  A small rate of monthly/quarterly churn will compound quickly over time.  According to Michael Redbord, general manager of Service Hub at HubSpot, “just 1 percent monthly churn translates to almost 12 percent yearly churn” for a subscription-based business.  Thus, companies with high churn rates can quickly find themselves in a financial hole as they will need to devote more resources to new customer acquisition.  Getting a new customer may cost up to five times more than retaining an existing customer.  


### Methods
#### Datasets
I obtained the test and train datasets for my project from Kaggle.com, which had already been divided into 80% training data and 20% test data (https://www.kaggle.com/mnassrib/telecom-churn-datasets?select=churn-bigml-80.csv, https://www.kaggle.com/mnassrib/telecom-churn-datasets?select=churn-bigml-20.csv).  The observations in the dataset contained phone data for each individual customer.  A description of the fields is provided in the table below:

| ![PNG](/images/cc1_dataset.png)   | 
|:--:| 
| *Figure 1: Dataset Fields* |

The dataset was fairly clean and contained no missing values.  State, area code, and the Boolean variables (including the churn target variable) were converted to factors.

#### EDA
A frequency distribution of the churn target variable indicated that 388 customers churned out of a total of 2,666 customers, reflecting an overall churn rate of 14.6%.  

| ![PNG](/images/cc2_distribution.png)   | 
|:--:| 
| *Figure 2: Distribution of Target Variable* |

Histograms, density plots, boxplots, and frequency distributions (for the categorical variables) were used to analyze each of the variables individually, as well as in combination with other variables.  

| ![PNG](/images/cc3_intl_bar.png)   | 
|:--:| 
| *Figure 3: International Plan Churn - Bar Plot* |

| ![PNG](/images/cc4_intl_prop.png)   | 
|:--:| 
| *Figure 4: International Plan Churn - Proportion* |

| ![PNG](/images/cc5_cs_bar.png)   | 
|:--:| 
| *Figure 5: Customer Service Call Churn - Bar Plot* |

| ![PNG](/images/cc2_distribution.png)   | 
|:--:| 
| *Figure 6: Customer Service Call Churn - Proportion* |

Key findings during the EDA phase indicated that churn was much higher for customers who had the international plan and more customer service calls.  It was also found that there were strong correlations between the number of minutes and total charges for each of day, evening, night, and international calls.  This made sense as total charges should be a function of the time spent on calls.  As a result, the number of minutes should be excluded from the model to avoid the problem of multicollinearity.  

#### Modeling
          Numerous logistic regression models were created in R, based on the initial EDA, correlation analysis and personal experience.  The final model calculated predicted churn based on whether the customer had the international plan and/or the voicemail plan, the number of international and customer service calls, and each of the charges for day, evening, night, and international calls.  Model parameters and statistics are shown below:

| ![PNG](/images/TBD.png)   | 
|:--:| 
| *Figure 2: Logistic Regression Model Parameters and Statistics* |





The churn variable was converted from Boolean to a numeric value of 0 (False) or 1 (True), and a confusion matrix was created to analyze the results on the training dataset, as shown below:

Exhibit 3 – Confusion Matrix and Statistics for Training Dataset
Metric	Score
accuracy	0.864591
kap	0.270778
sens	0.972783
spec	0.229381
ppv	0.881113
npv	0.589404
mcc	0.308418
j_index	0.202165
bal_accuracy	0.601082
detection_prevalence	0.943361
precision	0.881113
recall	0.972783
f_meas	0.924682
 
          Based on the high accuracy, precision, and recall from the training dataset, the model was then fit on the test dataset.







### Results
          The confusion matrix and statistics for the test set were in line with the training set as shown below:
Exhibit 4 – Confusion Matrix and Statistics for Test Dataset
Metric	Score
accuracy	0.853074
kap	0.193152
sens	0.965035
spec	0.178947
ppv	0.876191
npv	0.45946
mcc	0.219836
j_index	0.143982
bal_accuracy	0.571991
detection_prevalence	0.944528
precision	0.876191
recall	0.965035
f_meas	0.918469
 

The confusion matrix for the test set indicates strong values for accuracy, precision, and recall, as with the training set.  However, when analyzed more closely, it appears that the model was not as good at predicting true positives.  In this case, that would mean that the model did not correctly predict that a customer would churn when they, in fact, did.  This could indicate that the dataset used for this study may not have included all of the factors affecting customer churn.  For example, the customer may have found a better deal with another provider and cancelled without making any customer service calls.  This would be difficult to detect based on the dataset for this model, without being able to compare the fees charged by the company in question with what other providers are charging.  

In addition, the most predictive variables for the final model were arranged in order of importance as shown below:


Field	 Overall 
International.planYes	       13.21 
Customer.service.calls	       11.53 
Total.day.charge	       10.41 
Voice.mail.planYes	         5.58 
Total.intl.charge	         4.45 
Total.eve.charge	         4.40 
Total.intl.calls	         4.14 
Total.night.charge	         2.21 

Based on these results, the client may want to further explore why customers with international plans are canceling and consider offering additional services.  In addition, efforts should be made to minimize customer service calls, as well as improving the quality of such calls to further engage the customer, especially for customers with higher total day charges.  


### Conclusion
          This study found that a logistic model can be used to accurately predict overall customer churn, as well as determining the key variables contributing to such churn.  Although the model was good overall at predicting whether a customer would churn or not, additional efforts could be made to focus on customers that were not expected to churn, but actually did, as the unexpected loss of customers can have significant impact on company revenue and growth.  Datasets incorporating more features relating to churn could also be incorporated.  
          The key fields leading to churn were identified for possible corrective action.  Although such fields were identified, further analysis could be done to get to the true source of the problem (such as why customers with the international plan have a higher churn rate), so additional retention efforts can be made.


### References

https://towardsdatascience.com/machine-learning-powered-churn-analysis-for-modern-day-  business-leaders-ad2177e1cb0d
https://www.kdnuggets.com/2019/05/churn-prediction-machine-learning.html
https://data-flair.training/blogs/r-data-science-project-customer-segmentation/
https://rstudio-pubsstatic.s3.amazonaws.com/582266_8f09a8597a3742d1a5fc90fa231c908b.html
https://www.kdnuggets.com/2018/12/data-science-projects-business-impact.html
https://www.dataoptimal.com/churn-prediction-with-r/
https://www.altexsoft.com/blog/business/customer-churn-prediction-for-subscription-businesses-using-machine-learning-main-approaches-and-models/
https://towardsdatascience.com/predict-customer-churn-with-r
https://rpubs.com/dhaval8895/CustomerChurn
https://lukesingham.com/how-to-make-a-churn-model-in-r/
https://www.kaggle.com/blastchar/improve-customer-retention-churn/data?select=WA_Fn-UseC_-Telco-Customer-Churn.csv


