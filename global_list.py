import pandas as pd
import  numpy as np
import  ann
global dataMat
global labelMat


###read the data###
pandas_data = pd.read_csv('sql_eigen.csv')
sql_eigen = pandas_data.fillna(np.mean(pandas_data))

data = sql_eigen.iloc[:, 0:85]
# data.iloc[:,84][data.iloc[:,84]>200]=91
data['age'][data['age'] > 200] = 91
data2 = data.drop(['hr_cov', 'bpsys_cov', 'bpdia_cov', 'bpmean_cov', 'pulse_cov', 'resp_cov', 'spo2_cov'], axis=1)
label = sql_eigen['class_label']
dataMat1 = np.array(data2)
data01 = ann.preprocess(dataMat1)
dataMat = ann.preprocess1(data01)
dataMat = np.array(dataMat)
labelMat = np.array(label)
 ###read the data