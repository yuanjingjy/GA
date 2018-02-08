#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/20 8:54
# @Author  : YuanJing
# @File    : MIV.py

import pandas as pd
import  numpy as np
import  ann
from sklearn.neural_network import MLPClassifier  # import the classifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

pandas_data = pd.read_csv('eigen62.csv')
data = pandas_data.fillna(np.mean(pandas_data))

data['age'][data['age'] > 200] = 91.4
#data2 = data.drop(['hr_cov', 'bpsys_cov', 'bpdia_cov', 'bpmean_cov', 'pulse_cov', 'resp_cov', 'spo2_cov','height'], axis=1)
data2=data
dataSet=np.array(data2)#之前是程序删的无用项，现在手动删除
dataSet[:,0:62]=ann.preprocess(dataSet[:,0:62])
dataSet[:,0:62]=ann.preprocess1(dataSet[:,0:62])

dataMat=dataSet[:,0:62]
labelMat=dataSet[:,62]
# dataMat=np.array(dataMat)
# labelMat=np.array(labelMat)

clf = MLPClassifier(hidden_layer_sizes=(30,), activation='tanh',
                    shuffle=True, solver='sgd', alpha=1e-6, batch_size=3,
                    learning_rate='adaptive')
clf.fit(dataMat,labelMat)
IV=[]
for i in range(62):
    tmpdata=dataMat.copy()
    tmpdata[:, i]=tmpdata[:,i]*0.8
    train_dec=tmpdata
    pre_dec = clf.predict_proba(train_dec)
    tmpdata[:,i]=tmpdata[:,i]*11/8
    pre_inc = clf.predict_proba(tmpdata)



    IVi=pre_dec[:,0]-pre_inc[:,0]
    meanIV=np.mean(IVi)
    IV.append(meanIV)
IV=np.array(IV)
IV=np.abs(IV)
print("test")

datacolname= data2.drop(['class_label'], axis=1)#带名称的特征向量
eigencounts = pd.DataFrame([IV],index=['score'],columns=datacolname.keys())
sorteigen = eigencounts.sort_values(by='score',ascending=False, axis=1)#对MIV进行排序
sorteigen.to_csv('D:/anntest.csv', encoding='utf-8', index=True)
eigenwithname = pd.DataFrame(dataMat,columns=datacolname.keys())
sorteigenname=sorteigen.keys()#MIV排序后的列名
MIV30index=sorteigenname[0:30]
MIV30=data2.loc[:,MIV30index]
MIV30['class_label']=labelMat
MIV30.to_csv('D:/MIV30.csv', encoding='utf-8', index=True)
print()

fitscore=[]
for i in range(62):
    col=sorteigen.keys()
    index=col[0:i+1]
    dataMat=eigenwithname.loc[:,index]
    dataMat=np.array(dataMat)

    skf = StratifiedKFold(n_splits=5)
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
