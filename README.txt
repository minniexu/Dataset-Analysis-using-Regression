README.txt:
4 .py files are included in this project.

-----------------------------------------------------------------------------------------------------------------------------------
project1_1_plot.py:
This file shows the plots required for Network Backup dataset problem 1 and part of problem 2.

First of all, you should import the data as array first through 'Import data' in the spyder and named as ‘data’.And in order to run the code properly, you should put the .csv file within the same file as the source code and instal pybrain package using command 'pip install -i https://pypi.binstar.org/pypi/simple pybrain' in terminal.

Afterwards directly run this file and 5 plots which are “Actual copy size on a time period of 20 days” for each workflow and “residuals vs fitted value plot” for Network Backup Dataset would be shown. No need to change any parameters.

-----------------------------------------------------------------------------------------------------------------------------------
neuralnetwork.py:
This file includes the analyzed result for neural network of the Network Backup dataset.

For neural network regression, you can change the number of hidden layers in the function buildNetwork, ourvalue is 100 and you can specify maxEpoches by changing the maxEpochs parameter in function trainer.trainUntilConvergence.

Run this file and there would be several results return:
(Because verbose=True, it will print “Total error” for each of epoch.)
10-fold cross validation RMSE for each fold;
10-fold cross validation average RMSE;

-----------------------------------------------------------------------------------------------------------------------------------
project1_1.py:
This file analyzes the Network Backup dataset and includes all the results required for question 2 and 3 except neural network.

About how to tune the parameters for regressions, you can change the 10-fold-validation to any k-fold-validation by change the "cv = " parameter in function "cross_val_predict" and you can also change the regression model by change the first parameter in this funciton. In our code lr stand for linear regression and rf stand for random forest regression. You can specify the number of trees and depth of each tree by the changing the parameter 'n_estimator' and 'max_depth' in "rf = RandomForestRegressor(n_estimators=20, max_depth=4, warm_start=True, oob_score=True)". For polynomial regression, you can change to the degree by changing variable 'degree' in our code. 

You could also manually change the parameter of degree for polynomial regression part.
degree = 3  #default is 3
Warning: larger the degree number, longer the time would run.

Run this file and there would be several results return:
RMSE of Linear Regression for workflow0;
RMSE of Linear Regression for workflow1; 
RMSE of Linear Regression for workflow2;
RMSE of Linear Regression for workflow3; 
RMSE of Linear Regression for workflow4; 
RMSE of Linear Regression;
RMSE of Random Forest Regression;
(There might be UserWarning here, just ignore it);
‘Actual Value vs Predicted Value’ plot;
Polynomial Regression from degree = 1 to degree = # without 10-fold;
‘Average RMSE VS degree of polynomial without 10-fold’ plot;
Average RMSE from degree = 1 to degree = # with 10-fold cross validation;
‘Average RMSE VS degree of polynomial with 10-fold’ plot;

-----------------------------------------------------------------------------------------------------------------------------------
project1_2.py: 
This file analyzes the dataset of Boston housing and includes all the results required for question 4 and 5.

You could manually change the parameters of alpha of Ridge Regression and Lasso Regression:
clf = Ridge(alpha = 0.1);
clf = linear_model.Lasso(alpha = 0.1);
Degree for polynomial regression part could also be manually changed then.
degree = 4  #default is 4

Run this file and there would be several results return:
RMSE of Linear Regression;
‘Prices vs Predicted Prices' plot;
‘residuals versus fitted values’ plot;
RMSE for Ridge Regression;
RMSE for Lasso Regularization;
Polynomial Regression from degree to degree = # without 10-fold;
‘Average RMSE VS degree of polynomial without 10-fold' plot;
Average RMSE from degree = 1 to degree = # in 10-fold cross validation;
'Average RMSE VS degree of polynomial with 10-fold' plot;

-----------------------------------------------------------------------------------------------------------------------------------
Helpful Links:
Linear Regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
Random Forest Regression:
http://scikit-learn.org/stable/modules/ensemble.html
Polynomial Regression:
http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
Ridge Regression:
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
Lasso Regression:
http://scikit-learn.org/stable/modules/linear_model.html
Neural Network: 
building upnetwork:
http://pybrain.org/docs/api/tools.html?highlight=buildnetwork#pybrain.tools.shortcuts.buildNetwork
training:
http://pybrain.org/docs/api/supervised/trainers.html?highlight=backprop#pybrain.supervised.trainers.BackpropTrainer

