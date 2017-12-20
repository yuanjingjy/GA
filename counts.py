#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/20 14:45
# @Author  : YuanJing
# @File    : counts.py

import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import  ann
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier  # import the classifier



result=pd.read_csv("result.csv")
counts=sum(result)
import  global_list as gl
colnames=gl.data2
eigencounts=pd.DataFrame([counts],columns=colnames.keys())
sorteigen=eigencounts.sort_values(ascending=False,axis=1)

dataSet=np.array(colnames)
dataSet[:,0:78]=ann.preprocess(dataSet[:,0:78])
dataSet[:,0:78]=ann.preprocess1(dataSet[:,0:78])

labelMat=dataSet[:,78]
fitscore=[]
for i in range(78):
    col=sorteigen.keys()
    index=col[0:i]
    dataMat=dataSet[:,col]
    skf = StratifiedKFold(n_splits=10)
    scores=[]
    for train, test in skf.split(dataMat, labelMat):
        print("%s %s" % (train, test))
        train_in = dataMat[train]
        test_in = dataMat[test]
        train_out = labelMat[train]
        test_out = labelMat[test]
        clf = MLPClassifier(hidden_layer_sizes=(i,), activation='tanh',
                            shuffle=True, solver='sgd', alpha=1e-6, batch_size=1,
                            learning_rate='adaptive')
        clf.fit(train_in, train_out)
        score = clf.score(test_in,test_out)
        scores.append(score)
    scores = np.array(scores)
    mean_score = np.mean(scores)
    fitscore.append(mean_score)
fitscore = np.array(fitscore)

fig, ax1 = plt.subplots()
line1 = ax1.plot(fitscore, "b-", label="score")
ax1.set_xlabel("Number of features")
ax1.set_ylabel("Scores", color="b")
plt.show()

