import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import torch
from scipy import signal
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, accuracy_score

# y = np.array([0, 0, 1, 1]) # 真实值
# y_pred1 = np.array([0.3, 0.2, 0.25, 0.7]) # 预测值
# y_pred2 = np.array([0, 0, 1, 0]) # 预测值
al = []
dataset = pd.read_csv('image/res501.csv', delimiter=",", header=None)
dataset = dataset.values
print(dataset.shape)
# lab = dataset[:,-1]
y = dataset[2,:]
print(y)
y_pred = dataset[1,:]
y_soce = dataset[0,:]
# 预测值是概率
auc_score = roc_auc_score(y, y_soce)
# auc = roc_auc_score(y, y_soce,average=None)
cm = confusion_matrix(y, y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
# 敏感性Sensitivity(SE) 特异性Specificity(SP)
sp = TN / float(TN + FP)
sn = TP / float(TP + FN)
tpr = TP/(TP+FN)
fpr = FP/(FP+TN)
# print(fpr)
# print(tpr)
acc = accuracy_score(y,y_pred)
print(acc)
mcc = matthews_corrcoef(y, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y,y_soce)
# print(tpr)
# print(fpr)
print(auc_score) # 0.75
# print(auc)
print(sp)
print(sn)
print(mcc)
al.append(acc)
al.append(auc_score)
al.append(sp)
al.append(sn)
al.append(mcc)

a = torch.tensor(al)

# np.savetxt('all.csv',a,delimiter=',')




