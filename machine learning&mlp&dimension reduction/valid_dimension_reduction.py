from copy import deepcopy

import joblib
from sklearn import metrics
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, precision_score, recall_score, precision_recall_curve, confusion_matrix,roc_auc_score
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from sklearn.model_selection import train_test_split, KFold
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.nn import AdaptiveAvgPool2d
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from scipy import signal
from xgboost import XGBClassifier

import SPA
import CARS
def process(data):
    d = np.empty((data.shape[0], 40, 200))
    s = 50
    w = 200
    for i in range(data.shape[0]):
        for j in range(40):
            for k in range(200):
                d[i][j][k] = data[i, j * s + k]
    return d
# Savitzky-Golay平滑滤波
def SG(data, w=7, p=2,d=0):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param w: int
       :param p: int
       :return: data after SG :(n_samples, n_features)
    """
    return  signal.savgol_filter(data, w, p,deriv=d)
# 标准正态变换
# 均值中心化
def CT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MeanScaler :(n_samples, n_features)
       """
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data
def MA(data, WSZ=7):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param WSZ: int
       :return: data after MA :(n_samples, n_features)
    """

    for i in range(data.shape[0]):
        out0 = np.convolve(data[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(data[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(data[i, :-WSZ:-1])[::2] / r)[::-1]
        data[i] = np.concatenate((start, out0, stop))
    return data

def SNV(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after SNV :(n_samples, n_features)
    """
    m = data.shape[0]
    n = data.shape[1]
    # print(m, n)  #
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return  data_snv

a=[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
best_score = 0
bestvalue = None
pv = [0.90,0.92,0.94,0.96,0.98]
pca_values = [10, 15,20, 25,30,35, 40, 45,50]
tsnev = [4,6,8,10,12,14,16,18,20]
Mcc = []
Auc = []
from sklearn.manifold import TSNE

result = {}
all = np.empty((9,3))
data = pd.read_csv('shuju111.csv', header=None)

data = data.values
# data = torch.tensor(data, dtype=torch.float)
for h in range(9):
    c = []
    b=0
    p = tsnev[h]

    print(data.shape)

    data_D = SNV(data[:, :-1])

    features = np.array(data_D)
    x = features.reshape(660,2151)
    lables = data[:, -1]
    y = lables

    # 划分训练集和测试集
    train_data, test_data = x[165:], x[:165]
    train_lable, test_lable = y[165:],y[:165]
    print(train_data.shape)
    train_data = TSNE(n_components = p, method='exact').fit_transform(train_data)

    kfold = KFold(n_splits=10, shuffle=True,random_state=3)
    # train_data = PCA(n_components=p).fit_transform(train_data)
    # # 4. 训练数据
    # 训练和评估模型
    for train_index, test_index in kfold.split(train_data, train_lable):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_lable[train_index], train_lable[test_index]

        # pca = PCA(n_components=p).fit(X_train)
        # X_train = pca.transform(X_train)
        # X_test = pca.transform(X_test)
        print(X_train.shape)
        # svc = SVC(kernel='rbf', probability=True, gamma=0.1)
        xgb = XGBClassifier(learning_rate=0.1, n_estimators=250, min_child_weight=1, max_depth=2, gamma=0.0001,subsample=0.7,colsample_bytree=0.4,reg_alpha=0,scale_pos_weight=1,objective= 'binary:logistic')
        xgb.fit(X_train, y_train)
        scores = xgb.predict_proba(X_test)[:, 1]
        y_pred = xgb.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        au = metrics.roc_auc_score(y_test, scores)
        mc = matthews_corrcoef(y_test, y_pred)
        a.append(accuracy)
        Auc.append(au)
        Mcc.append(mc)
        b += 1
    avg_score = round(np.mean(np.array(a)),4)
    avg_auc = round(np.mean(np.array(Auc)),4)
    avg_mcc = round(np.mean(np.array(Mcc)),4)
    if avg_score > best_score:
        best_score = avg_score
        bestvalue = p
    print(avg_score)
    print(bestvalue)

    result[h] ={'Acc':avg_score,'Auc':avg_auc,'Mcc':avg_mcc}
    a = list(result[h].values())
    # print(a)
    all[h] = a
    #
    # all_transposed = all.transpose((1, 0, 2))
    # alll = all_transposed.reshape(3, 520)
    # alll = torch.tensor(alll)
    # np.savetxt("1.csv", alll, delimiter=",")

print(all)
# np.savetxt("test1.csv", all, delimiter=",")
    # print(all)