# credit-risk-classification

In this Challenge, various techniques are used to train and evaluate a model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company is used to build a model that can identify the creditworthiness of borrowers.

So, the purpose of the analysis is to identify the borrowers’ creditworthiness. If the borrower is not creditworthy (will not be able to return the loan) but gets loan, the lender will lose money. If the borrower is creditworthy but does not get loan, the lender will lose a customer. So, need to be careful with False Positive and False Negative predictions about the borrowers’ statuses of creditworthiness. Supervised Machine Learning is used to predict which loan has low-risk (status-0) and which loan has high-risk (status-1). 

The features ('X') were the following columns: <br/>
loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt <br/>
loan status was the data to be predicted (labels set 'y')
 
value_counts() command is used (status_counts = df["loan_status"].value_counts()) to check if the lables data (variable 'y') was balanced. <br/> 
It was not balanced. The number of healthy loans ('loan_status'=0) = 75036, <br/> 
the number of unhealthy loans ('loan_status'=1) = 2500. <br/>
In order to avoid bias and improve the performance of the model, 'RandomOverSampler' module from 'imblearn.over_sampling' library was used, and resampled data was obtained. <br/>
The target value got balanced ('0' - 56277, '1' - 56277) <br/>

The stages performed are the following:<br/>
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

### Logistic Regression model (for original data):
  #### Description. Accuracy, Precision, and Recall scores: <br/>
  Confusion matrix was generated to get the number of true negative (actual loan was healthy, model prediction also was 'healthy'), <br/>
  true positive (loan - unhealthy, pred.- 'unhealthy'), <br/>
  false negative (loan - unhealthy, pred.- 'healthy) and <br/>
  false positive (loan-healthy, pred.- 'unhealthy') predictions. <br/>
  The matrix is the following: <br/>
                   [[18679, 80] <br/>
                   [67,   558]]
      

  Accuracy = (TP+TN)/(TP+TN+FP+FN) = (num. of correct predictions)/(total num. of predictions) = 99% <br/>
    for high-risk loans <br/>
  * Precision = TP/(TP+FP) = 558/(80+558) = 87% <br/>
  * Recall = TP/(TP+FN) = 89% <br/> <br/>
for low-risk loans <br/>
  * Precision = 100% <br/>
  * Recall = 100% <br/>
  
 The accuracy score of the model is 99% which is really high. For low-risk loans the precision and recall scores are perfect (100%). <br/>
 For high-risk loans the precision and recall scores are not bad (87%, 89%). <br/> <br/><br/>

### Logistic Regression model (for resampled data):
  #### Description. Accuracy, Precision, and Recall scores. <br/>
  Accuracy = (TP+TN)/(TP+TN+FP+FN) = (num. of correct predictions)/(total num. of predictions) = 100% <br/>
    for high-risk loans <br/>
  * Precision = TP/(TP+FP) = 558/(80+558) = 87% <br/>
  * Recall = TP/(TP+FN) = 100% <br/> <br/>
for low-risk loans <br/>
  * Precision = 100% <br/>
  * Recall = 100% <br/> <br/>

  
  The results for Oversampled data model are much better. This time the accuracy score is perfect (100%). <br/> 
  The Precision score for low-risk loans is 100%. That means all the loans that the model classified as 'low-risk' were actually 'low-risk'. <br/>
  The Recall score for high-risk loans and low-risk loans were 100%, which is a perfect result. <br/>
   For example, for high-risk loans, <br/> 
   Recall = TP/(TP+FN) = 100%   => FN = 0   => There is no loan that the model classified as 'low-risk' but it was actually 'high-risk' loan.   => <br/>
  no money loses. <br/>
  For high-risk loans,<br/> 
   the Precision score = 87% which is not bad. That means there are some false positives (loans which classified
  as ‘high-risk' but were actually 'low-risk'). <br/> 
  In this case, the lender will lose potential customers, who are creditworthy but classified
  as not creditworthy. <br/>
  But the score is not as bad,   => the number of FP is small.<br/>
  
## Summary

I would recommend using Logistic Regression model because it performed pretty good.  After resampling, it performed much better. 


<br/><br/>



## References
This project is a part of UC Berkeley "Data Analysis and Visualisation" Boot Camp education services.







