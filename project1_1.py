# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:41:35 2016

@author: Tongtong
"""
import math
import numpy
from sklearn import linear_model
from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
import random
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
enc = OneHotEncoder()
fileName = 'network_backup_dataset.csv'
file = open(fileName)
data = []
file.readline()
DAY = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
       'Friday', 'Saturday', 'Sunday']
for line in file:
    words = line.split(',')
    # week
    words[0] = int(words[0])
    # day of week
    for i in range(len(DAY)):
        if words[1] == DAY[i]:
            words[1] = i + 1
    # Backup Start Time - Hour of Day
    words[2] = int(words[2])
    # Work-Flow-ID
    words[3] = int(words[3][10:])
    # File Name
    words[4] = int(words[4][5:])
    # Size of Backup (GB)
    words[5] = float(words[5])
    # Backup Time (hour)
    words[6] = int(words[6])
    data.append(words)
random.shuffle(data)
df = pd.DataFrame(data, columns=['week', 'weekday', 'st', 'id', 'filename', 'copysize', 'et'])
workflow0 = df[(df.id == 0)]
workflow1 = df[(df.id == 1)]
workflow2 = df[(df.id == 2)]
workflow3 = df[(df.id == 3)]
workflow4 = df[(df.id == 4)]

enc = OneHotEncoder()

backupsize0 = workflow0.copysize
del workflow0['copysize']
del workflow0['id']
onehot0 = enc.fit_transform(workflow0).toarray()
lr = linear_model.LinearRegression()
lrpredicted0 = cross_val_predict(lr, onehot0, backupsize0, cv=10)
backupsize0 = backupsize0.values
#print 'RMSE of Linear Regression for workflow0', math.sqrt(mean_squared_error(backupsize0, lrpredicted0))
sum = 0
for i in range(0, len(backupsize0)):
    t = (backupsize0[i] - lrpredicted0[i]) * (backupsize0[i] - lrpredicted0[i])
    if t > 1:
        continue
    sum += t
    if sum > 1:
        print i
        break
    #print sum
print 'RMSE of Linear Regression for workflow0', math.sqrt(sum / (float(len(lrpredicted0) - 1)))


backupsize1 = workflow1.copysize
del workflow1['copysize']
del workflow1['id']
onehot1 = enc.fit_transform(workflow1).toarray()
lr = linear_model.LinearRegression()
lrpredicted1 = cross_val_predict(lr, onehot1, backupsize1, cv=10)
print 'RMSE of Linear Regression for workflow1', math.sqrt(mean_squared_error(backupsize1, lrpredicted1))


backupsize2 = workflow2.copysize
del workflow2['copysize']
del workflow2['id']
onehot2 = enc.fit_transform(workflow2).toarray()
lr = linear_model.LinearRegression()
lrpredicted2 = cross_val_predict(lr, onehot2, backupsize2, cv=10)
print 'RMSE of Linear Regression for workflow2', math.sqrt(mean_squared_error(backupsize2, lrpredicted2))

backupsize3 = workflow3.copysize
del workflow3['copysize']
del workflow3['id']
onehot3 = enc.fit_transform(workflow3).toarray()
lr = linear_model.LinearRegression()
lrpredicted3 = cross_val_predict(lr, onehot3, backupsize3, cv=10)
print 'RMSE of Linear Regression for workflow3', math.sqrt(mean_squared_error(backupsize3, lrpredicted3))


backupsize4 = workflow4.copysize
del workflow4['copysize']
del workflow4['id']
onehot4 = enc.fit_transform(workflow4).toarray()
lr = linear_model.LinearRegression()
lrpredicted4 = cross_val_predict(lr, onehot4, backupsize4, cv=10)
print 'RMSE of Linear Regression for workflow4', math.sqrt(mean_squared_error(backupsize4, lrpredicted4))

backupsize = df.copysize

del df['copysize']
onehot = enc.fit_transform(df).toarray()
lr = linear_model.LinearRegression()
lrpredicted = cross_val_predict(lr, onehot, backupsize, cv=10)

#Random Forest 
rf = RandomForestRegressor(n_estimators=20, max_depth=4, warm_start=True, oob_score=True)
RFpredicted = cross_val_predict(rf, onehot, backupsize, cv=10)

print 'RMSE of Linear Regression', math.sqrt(mean_squared_error(backupsize, lrpredicted))
print 'RMSE of Random Forest Regression', math.sqrt(mean_squared_error(backupsize, RFpredicted))

fig, ax = plt.subplots()
ax.scatter(backupsize, RFpredicted)
ax.plot([backupsize.min(), backupsize.max()], [backupsize.min(), backupsize.max()], lw=4)
ax.set_xlabel('Measured value')
ax.set_ylabel('Predicted value')
plt.title('Actual Value vs Predicted Value')
plt.show()

#polynomial
degree = 3
xtt = df.values
mse_p = []
n = len(xtt)
deg = range(1,degree + 1)
train_poly_x = xtt[(n/10):]
test_poly_x = xtt[:(n/10)]
train_poly_y = backupsize[(n/10):]
test_poly_y = backupsize[:(n/10)]

for d in range(1, degree + 1):
    polynomial_features = PolynomialFeatures(degree = d, include_bias = False)
    linear_regression = LinearRegression()
    model = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    model.fit(train_poly_x, train_poly_y)
    pre_p = model.predict(test_poly_x)
    mse_test_p = math.sqrt(mean_squared_error(test_poly_y, pre_p))
    mse_p.append(mse_test_p)
print "Polynomial Regression from degree = 1 to degree = %s without 10-fold: " % degree, mse_p
plt.plot(deg, mse_p, '--bo')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('Average RMSE VS degree of polynomial without 10-fold')
plt.show()



kf = KFold(n, n_folds = 10)
kf_dict = dict([("fold_%s" % i, []) for i in range(1, 11)])
fold = 0

mse=numpy.zeros((degree, 10))
for train_index, test_index in kf:
    fold += 1
    train_x, test_x = xtt[train_index], xtt[test_index]
    train_y, test_y = backupsize[train_index], backupsize[test_index]
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

print "Average RMSE from degree = 1 to degree = %s with 10-fold cross validation:" % degree, ave  

plt.plot(deg, ave, '--bo')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('Average RMSE VS degree of polynomial with 10-fold')
plt.show()
