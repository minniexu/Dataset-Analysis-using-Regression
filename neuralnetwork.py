#Neural network
import math
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
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

def convertDataNeuralNetwork(data, values):
    fulldata = SupervisedDataSet(data.shape[1], 1)
    for d, v in zip(data, values):  
        fulldata.addSample(d, v)    
    return fulldata

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
backupsize = df.copysize
del df['copysize']
onehot = enc.fit_transform(df).toarray()

resall = []
for i in range(10):
    test = onehot[i*18588/10:(i+1)*18588/10]
    train = np.concatenate((onehot[:i*18588/10],onehot[(i+1)*18588/10:]), axis=0)
    train_v = np.concatenate((backupsize[:1*18588/10],backupsize[2*18588/10:]), axis=0)   
    test_v = backupsize[i*18588/10:(i+1)*18588/10]
    regressionTrain = convertDataNeuralNetwork(train, train_v)
    regressionTest = convertDataNeuralNetwork(test, test_v)
    net = buildNetwork(regressionTrain.indim,
                   100, # number of hidden units
                   regressionTrain.outdim,
                   bias = True,
                   hiddenclass = SigmoidLayer,
                   outclass = LinearLayer
                   )
    trainer = BackpropTrainer(net, dataset=regressionTrain, verbose=True)
    trainer.trainUntilConvergence(maxEpochs = 100)
    res = net.activateOnDataset(regressionTest)
    resp = mean_squared_error(res, test_v)   #0.00331332229115
    print resp
    resall.append(resp)
print "10-fold cross validation RMSE for each fold", resall    
aveRMSE = math.sqrt(sum(resall) / float(len(resall)))
print "10-fold cross validation average RMSE", aveRMSE