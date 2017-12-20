#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/20 8:54
# @Author  : YuanJing
# @File    : MIV.py

import pandas as pd
import  numpy as np
import  ann
from sklearn.neural_network import MLPClassifier  # import the classifier

pandas_data = pd.read_csv('sql_eigen.csv')
data = pandas_data.fillna(np.mean(pandas_data))

data['age'][data['age'] > 200] = 91
data2 = data.drop(['hr_cov', 'bpsys_cov', 'bpdia_cov', 'bpmean_cov', 'pulse_cov', 'resp_cov', 'spo2_cov','height'], axis=1)
dataSet=np.array(data2)
dataSet[:,0:78]=ann.preprocess(dataSet[:,0:78])
dataSet[:,0:78]=ann.preprocess1(dataSet[:,0:78])

dataMat=dataSet[:,0:78]
labelMat=dataSet[:,78]
dataMat=np.array(dataMat)
labelMat=np.array(labelMat)

clf = MLPClassifier(hidden_layer_sizes=(78,), activation='tanh',
                    shuffle=True, solver='sgd', alpha=1e-6, batch_size=5,
                    learning_rate='adaptive')
clf.fit(dataMat,labelMat)

for i in range(78):
    tmpdata=dataMat.copy()
    tmpdata[:, i]=tmpdata[:,i]*0.9
    train_dec=tmpdata
    tmpdata[:,i]=tmpdata[:,i]*11/9
    train_inc=tmpdata

    pre_dec=clf.predict_proba(train_dec)
    pre_inc=clf.predict_proba(train_inc)
    IV[:,i]=pre_dec[:,0]-pre_inc[:,0]