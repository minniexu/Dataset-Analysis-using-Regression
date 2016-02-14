# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:53:51 2016

@author: Tongtong
"""
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import math
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
data = pd.read_csv("housing_data.csv", header = None)
xtt = data.values

train = [dict(r.iteritems()) for _, r in data.iterrows()]
vector = DictVectorizer()
vector_s = vector.fit_transform(train)
vector_array = vector_s.toarray()
vector_array=np.random.permutation(vector_array)

#data_file_name = join(module_path, 'data', 'housing_data.csv')
with open("housing_data.csv") as f:
    data_file = csv.reader(f)
    temp = next(data_file)
    n_samples = len(vector_array)
    n_features = len(vector_array[0]) -1
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,))
    feature_names = vector.get_feature_names()

    for i in range(0,len(vector_array)):
             target[i] = vector_array[i][13]
    vector_array=np.delete(vector_array, 13, axis=1)
lr = linear_model.LinearRegression()
predict = cross_val_predict(lr, vector_array, target, cv=10)
print "RMSE of Linear Regression", math.sqrt(mean_squared_error(target, predict))
fig, ax = plt.subplots()
ax.scatter(target, predict)
ax.plot([target.min(), target.max()], [target.min(), target.max()], lw=4)
ax.set_xlabel('Price')
ax.set_ylabel('Predicted Prices')
plt.title('Prices vs Predicted Prices')
plt.show()

#residual 
residuals = []
for i in range(0, len(target)):
    residuals.append(predict[i] - target[i])
plt.figure()
plt.plot(predict, residuals, 'o')
plt.xlabel("Fitted value")
plt.ylabel("Residuals")
plt.title("residuals versus fitted values plot")
plt.show()


#ridge regression
clf = Ridge(alpha = 0.1)
clf.fit(vector_array, target)
predict_r = clf.predict(vector_array)
print 'RMSE for Ridge Regression: ', math.sqrt(mean_squared_error(target, predict_r))
#alpha = 0.1   4.67957
#alpha = 0.01  4.67920
#alpha = 0.001 4.67919

#Lasso regression
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(vector_array, target)
predict_p = clf.predict(vector_array)
print 'RMSE for Lasso Regularization: ', math.sqrt(mean_squared_error(target, predict_p))
#alpha = 0.1   4.80083
#alpha = 0.01  4.68309
#alpha = 0.001 4.67923


#polynomial
xtt = vector_array
n = len(xtt)
degree = 4
train_poly_x = xtt[(n/10):]
test_poly_x = xtt[:(n/10)]
train_poly_y = target[(n/10):]
test_poly_y = target[:(n/10)]
deg = range(1,degree + 1)
mse_p = []
for d in range(1, degree + 1):
    polynomial_features = PolynomialFeatures(degree = d, include_bias = False)
    linear_regression = LinearRegression()
    model = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    model.fit(train_poly_x, train_poly_y)
    pre_p = model.predict(test_poly_x)
    mse_test_p = math.sqrt(mean_squared_error(test_poly_y, pre_p))
    mse_p.append(mse_test_p)
print "Polynomial Regression from degree to degree = %s without 10-fold: " % degree, mse_p
plt.plot(deg, mse_p, '--bo')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('Average RMSE VS degree of polynomial without 10-fold')
plt.show()

#polynomial 10-fold validation

folds = 10 #when 10-fold cross validation, change to 10
kf = KFold(n, n_folds = folds)
kf_dict = dict([("fold_%s" % i, []) for i in range(1, folds + 1)])
fold = 0

mse=np.zeros((degree, folds))


for train_index, test_index in kf:
    fold += 1
    train_x, test_x = xtt[train_index], xtt[test_index]
    train_y, test_y = target[train_index], target[test_index]
    for d in range(1, degree + 1):
        polynomial_features = PolynomialFeatures(degree = d, include_bias = False)
        linear_regression = LinearRegression()
        model = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
        model.fit(train_x, train_y)
        pre_y = model.predict(test_x)
        mse_test = math.sqrt(mean_squared_error(test_y, pre_y))
        mse[d - 1][fold - 1] = mse_test

ave=[]
for i in range(1,degree+1):
    sb = np.sum(mse[i - 1]) / 10
    ave.append(sb)

print "Average RMSE from degree = 1 to degree = %s in 10-fold cross validation:" % degree, ave  

plt.plot(deg, ave, '--bo')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('Average RMSE VS degree of polynomial with 10-fold')
plt.show()




