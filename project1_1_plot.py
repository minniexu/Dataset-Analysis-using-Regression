# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 23:15:37 2016

@author: Tongtong
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:43:57 2016

@author: ddddc
"""
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
import math

enc = OneHotEncoder()
fileName = 'network_backup_dataset.csv'
file = open(fileName)
data = []
file.readline()  # 跳过首行
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
#temp = df[(df.id == 0)]
backupsize = df.copysize
del df['copysize']
#del temp['id']
onehot = enc.fit_transform(df).toarray()
lr = linear_model.LinearRegression()
lrpredicted = cross_val_predict(lr, onehot, backupsize, cv=10)
rf = RandomForestRegressor(n_estimators=20, max_depth=4, warm_start=True, oob_score=True)
RFpredicted = cross_val_predict(rf, onehot, backupsize, cv=10)
lrpredicted = pd.DataFrame(lrpredicted, columns=['presize'])
RFpredicted = pd.DataFrame(RFpredicted, columns=['RFpresize'])
df = pd.concat([df, lrpredicted], axis=1)
df = pd.concat([df, backupsize], axis=1)
df = pd.concat([df, RFpredicted], axis=1)
sdf = df.sort_values(by=['week', 'weekday', 'id', 'st'])
for k in range(0 ,5):
    plt.figure()
    for i in range(1, 4):
        for j in range(1, 8):
                temp = sdf[(sdf.id == k)&(sdf.week == i)&(sdf.weekday==j)]
                day = [(i - 1) * 7 + j] * len(temp)
                plt.plot(day, temp.copysize, 'bo')
                plt.plot(day, temp.presize, 'ro')
                plt.plot(day, temp.RFpresize, 'go')
                

residual = []
backupsize = backupsize.values
lrpredicted = lrpredicted.values
for i in range(0, len(backupsize)):
    residual.append(lrpredicted[i] - backupsize[i])
plt.figure();
plt.plot(lrpredicted, residual, 'o')
plt.xlabel("Fitted value")
plt.ylabel("Residuals")
plt.title("residuals versus fitted values plot")
plt.show()
