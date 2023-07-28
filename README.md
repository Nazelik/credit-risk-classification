# credit-risk-classification

## Overview of the Analysis

In this Challenge, various techniques are used to train and evaluate a model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company is used to build a model that can identify the creditworthiness of borrowers.

So, the purpose of the analysis is to identify the borrowers’ creditworthiness. If the borrower is not creditworthy (will not be able to return the loan) but gets loan, the lender will lose money. If the borrower is creditworthy but does not get loan, the lender will lose a customer. So, need to be careful with False Positive and False Negative predictions about the borrowers’ statuses of creditworthiness. Supervised Machine Learning is used to predict which loan has low-risk (status-0) and which loan has high-risk (status-1). 

The data was on the following information (the features 'X'):
loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt
loan status was the data to be predicted (labels set 'y')
 
value_counts() command is used (status_counts = df["loan_status"].value_counts()) to check if the lables data (variable 'y') was balanced. 
It was not. The number of healthy loans ('loan_status'=0) = 75036, the number of unhealthy loans ('loan_status'=1) = 2500. To avoid bias
and improve the performance of the model, 'RandomOverSampler' module from 'imblearn.over_sampling' library is used to resample the data
and take equal number of 'healty'/'unhealthy' data.

The following stages of the machine learning process were performed:
1. The given data was split into training and testing sets (training - 75%, testing - 25%),
2. Logistic Regression model was chosen,
3. The model was fitted with the training data,
4. The model was tested with the testing data,
5. Predictions was performed with the model using the testing data,
6. Model's performance was evaluated,
7. The original training data was fitted to the random_oversampler model, a new sample of training data was obtained,
8. 2-6 steps were performed for the new sample,
9. Results comparison was performed.

The predicted outcome, which is a column of labels, is binary (0/1 or 'low-risk'/'high-risk'). So, the model used was Logistic Regression model which is a classification model.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

### Logistic Regression model (for original data):
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  Confusion matrix was generated to get the number of true negative (loans was healthy, model prediction also was 'healthy'),
  true positive (loan - unhealthy, pred.- 'unhealthy'), false negative (loan - unhealthy, pred.- 'healthy) and false positive (loan-healthy, pred.- 'unhealthy') predictions. The matrix is the following: 
                   [[18679, 80]
                   [67,   558]]
               

  Accuracy = (TP+TN)/(TP+TN+FP+FN) = (num. of correct predictions)/(total num. of predictions) = 99%
  Precision = TP/(TP+FP) = 558/(80+558) = 87%    <---------------------for high-risk loan
  Precision = 100%                               <---------------------for low-risk loan
  Recall = TP/(TP+FN) = 89%                      <---------------------for high-risk loan
  Recall = 100%                                  <---------------------for low-risk loan
  
 The accuracy score of the model is 99% which is really high. For healthy loans the precision and recall scores are perfect (100%).
 For unhealthy loans the precision and recall scores are not bad (87%, 89%).

### Logistic Regression model (for resampled data):
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  Accuracy = (TP+TN)/(TP+TN+FP+FN) = (num. of correct predictions)/(total num. of predictions) = 100%
  Precision = TP/(TP+FP) = 558/(80+558) = 87%    <---------------------for high-risk loan
  Precision = 100%                               <---------------------for low-risk loan
  Recall = TP/(TP+FN) = 100%                     <---------------------for high-risk loan
  Recall = 100%                                  <---------------------for low-risk loan

  
  The results for Oversampled data model are much better. This time the accuracy score is perfect (100%), the Precision score for
  low-risk loans is 100%. That means all the loans that the model classified as 'low-risk' were actually 'low-risk'.
  The Recall score for high-risk loans and low-risk loans were 100%, which is a perfect result. For example, for high-risk loan, Recall =
  = TP/(TP+FN) = 100% => FN = 0 => There is no loan that the model classified as 'low-risk' but it was actually 'high-risk' loan. =>
  no money loses.
  For high-risk loan, the Precision score = 87% which is not bad. That means there are some false positives (loans which classified
  as ‘high-risk' but were actually 'low-risk'). In this case the lender will lose potential customers, who are creditworthy but classified
  as not creditworthy. But the score is not as bad, => the number of FP is small.
  
## Summary

I would recommend using Logistic Regression model because it performed pretty good. After resampling, it performed much better. 


<br/><br/>



## References
This project is a part of UC Berkeley "Data Analysis and Visualisation" Boot Camp education services.







