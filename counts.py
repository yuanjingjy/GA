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



result = pd.read_csv('GAresult.csv')
result=np.array(result)
counts=sum(result)
import  global_list as gl
colnames1=gl.data2
colnames=colnames1.drop(['class_label'], axis=1)
eigencounts=pd.DataFrame([counts],index=['score'],columns=colnames.keys())
sorteigen=eigencounts.sort_values(by='score',ascending=False,axis=1)
sorteigen.to_csv('D:/GAsum.csv', encoding='utf-8', index=True)

dataSet=np.array(colnames1)
dataSet[:,0:62]=ann.preprocess(dataSet[:,0:62])
dataSet[:,0:62]=ann.preprocess1(dataSet[:,0:62])
labelMat=dataSet[:,62]

dataSet=dataSet[:,0:62]
eigenwithname = pd.DataFrame(dataSet,columns=colnames.keys())#归一化后的特征值加标签
sorteigenname=sorteigen.keys()#GA排序后的列名
GA=eigenwithname.loc[:,sorteigenname]
GA['class_label']=labelMat
GA.to_csv('D:/GA.csv', encoding='utf-8', index=True)

fitscore=[]
for i in range(62):
    #col=sorteigen.keys()
    #index=col[0:i+1]
    dataMat=dataSet[:,0:i+1]
    dataMat=np.array(dataMat)
    skf = StratifiedKFold(n_splits=10)
    scores=[]
    for train, test in skf.split(dataMat, labelMat):
        print("%s %s" % (train, test))
        train_in = dataMat[train]
        test_in = dataMat[test]
        train_out = labelMat[train]
        test_out = labelMat[test]
        clf = MLPClassifier(hidden_layer_sizes=(i+1,), activation='tanh',
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

