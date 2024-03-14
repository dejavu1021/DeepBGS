# 导包
from copy import deepcopy

import joblib
import pywt
import torch
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression, LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, \
    roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC, SVR
from sklearn import metrics, svm, linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef

import SPA
import CARS
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import warnings

def regression_model(features_train, features_test, Organic_train, Organic_test, model):

    # 构造模型
    model.fit(features_train, Organic_train)

    # 使用测试集进行预测
    y_pred_0 = model.predict(features_train)
    y_pred = model.predict(features_test)
    # print(y_pred)
    # if (model == dtc)or(model == rfc):
    scores = model.predict_proba(features_test)[:,1]
    # print(roc_auc_score(Organic_test, y_pred))
    # au = roc_auc_score(Organic_test, y_pred)
    # print()
    # print('scores',scores.shape)
    # scores = np.asarray((scores))
    # else:
    #     scores = model.decision_function(features_test)
    fpr, tpr, thresholds = metrics.roc_curve(Organic_test, scores)
    # Organic_test = np.asarray(Organic_test)
    # precision, recall, _ = p_r_curve(Organic_test, scores)
    precisions, recalls, thresholds = precision_recall_curve(Organic_test, scores)

    # auc的输入为很简单，就是fpr, tpr值（至于这两个怎么算请看上面基本参数的介绍）
    auc = metrics.auc(fpr, tpr)
    cm = confusion_matrix(Organic_test,y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    tpr = TP/(TP+FN)
    # print(tpr)
    # 敏感性Sensitivity(Sn) 特异性Specificity(SP)
    sp = TN / float(TN+FP)
    sn = TP / float(TP+FN)
    mcc = matthews_corrcoef(Organic_test,y_pred)
    # # 均方根误差RMSE
    # RMSE_0 = round(np.sqrt(metrics.mean_squared_error(Organic_train, y_pred_0)), 3)
    # RMSE = round(np.sqrt(metrics.mean_squared_error(Organic_test, y_pred)), 3)
    # MSE_0 = round(metrics.mean_squared_error(Organic_train, y_pred_0), 3)
    # MSE = round(metrics.mean_squared_error(Organic_test, y_pred), 3)
    # # R^2 score，即决定系数，反映因变量的全部变异能通过回归关系被自变量解释的比例
    # r2_0 = round(r2_score(Organic_train, y_pred_0), 3)
    # r2 = round(r2_score(Organic_test, y_pred), 3)
    accuracy = round(metrics.accuracy_score(Organic_test, y_pred)*100,3)
    return accuracy,auc,precisions,recalls,fpr,tpr,sp,sn,mcc
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

# 移动平均平滑
def MA(data, WSZ=11):
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

def mean_centralization(sdata):
        """
        均值中心化
        """
        sdata = deepcopy(sdata)
        temp1 = np.mean(sdata, axis=0)
        temp2 = np.tile(temp1, sdata.shape[0]).reshape(
            (sdata.shape[0], sdata.shape[1]))
        return sdata - temp2,temp1

def pro(data,me):
    sdata = deepcopy(data)
    temp2 = np.tile(me, sdata.shape[0]).reshape(
        (sdata.shape[0], sdata.shape[1]))
    return sdata - temp2

# # 导入数据集切割训练与测试数据
tvi = np.empty((650,1))
npci = np.empty((650,1))
pri = np.empty((650,1))
nir = np.empty((650,8))
nvid = np.empty((650,80))
rvi = np.empty((650,80))
vb = np.empty((650,10))
data = pd.read_csv(f'result.csv', header=None)
data = data.values
lables = data[:, -1]
print(data.shape)
# lables = lables.reshape(650,1)
data_D = data[:,:-1]
features = np.array(data_D)
print(features.shape)
# 724 738 750 764 776 790 802 814
nir[:,0] = features[:,374]
nir[:,1] = features[:,388]
nir[:,2] = features[:,400]
nir[:,3] = features[:,414]
nir[:,4] = features[:,426]
nir[:,5] = features[:,440]
nir[:,6] = features[:,452]
nir[:,7] = features[:,464]

# 601 605 614  627 636 644 652 660 669 677
vb[:,0] = features[:,251]
vb[:,1] = features[:,255]
vb[:,2] = features[:,264]
vb[:,3] = features[:,277]
vb[:,4] = features[:,286]
vb[:,5] = features[:,294]
vb[:,6] = features[:,302]
vb[:,7] = features[:,310]
vb[:,8] = features[:,319]
vb[:,9] = features[:,327]


for i in range(8):
    for j in range(10):
        nvid[:,i*10+j] = (nir[:,i] - vb[:,j])/(nir[:,i] + vb[:,j])
        rvi[:,i*10+j] = (nir[:,i])/(vb[:,j])

min = np.min(rvi,axis=1).reshape(650,1)
# print(min)
max = np.max(rvi,axis=1).reshape(650,1)
print(min.shape)
nvid = np.array(nvid).reshape(650,80)
# rvi = np.array((rvi-min)/(max-min)).reshape(650,80)
print(rvi.shape)
# 0.5[120(750-550)-200（670-550）】
tvi = 0.5*(120*(features[:,400]-features[:,200])-200*(features[:,320]-features[:,200]))
tvi = np.array(tvi).reshape(650,1)
print(tvi)
# 680-430 / 680+430
npci = (features[:,330]-features[:,80])/(features[:,330]+features[:,80])
npci =  np.array(npci).reshape(650,1)
# 531-570 / 531+570
pri = (features[:,181]-features[:,220])/(features[:,181]+features[:,220])
pri = pri.reshape(650,1)

dat = np.hstack((nvid,rvi,npci,pri,tvi,lables.reshape(650,1)))
# print(all.shape)
np.savetxt('da.csv',dat,delimiter=',')



acc_all = np.empty((1, 4))
auc_all = np.empty((1, 4))
sp_all = np.empty((1,4))
sn_all = np.empty((1,4))
mcc_all = np.empty((1,4))
au_all = np.empty((1,4))

# data_train, data_test, label_train, label_test = train_test_split(features,lables,test_size=0.3,shuffle=True)
# # 模型训练与拟合linear  rbf  poly
t0 = time()
# data = pd.read_csv(f'da.csv', header=None)
# data = data.values
# print(data.shape)
# np.random.shuffle(dat)
# dat = torch.tensor(data, dtype=torch.float)
# np.savetxt(f"d29.csv", dat, delimiter=",")
# print(data)

# lables = data[:, -1]
# data_D = SG(data[:,:-1])
data_D = dat[:,:-1]
# print(data_D.shape)
# data_D = MA(data[:,:-1])
# data_D = mean_centralization(data[:,:-1])
# data_D = SNV(dat[:,:-1])
features = np.array(data_D)
# print(features)
# tsne = TSNE(n_components = p, method='exact').fit_transform(features)

# lables = lables.reshape(-1, 1)
# dataa = np.hstack((features,lables))
# train = dataa[:520]
# test = dataa[520:]
# np.random.shuffle(train)
# np.random.shuffle(test)
# train_data,train_lable = train[:,:-1],train[:,-1]
# test_data,test_lable = test[:,:-1],test[:,-1]
train_data, test_data = features[:520], features[520:]
train_lable, test_lable = lables[:520], lables[520:]
dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier()

svc = SVC(kernel='linear', probability=True)
svc2 = SVC(kernel='rbf', probability=True)
# svc2 = SVC(kernel='linear')
xgb = XGBClassifier()
#     # adj_params, fixed_params = xgboost_parameters()
#     # model_adjust_parameters(adj_params, fixed_params)
Models = [dtc, rfc,  svc2, xgb]
for i in range(4):
    model = Models[i]
    model.fit(train_data,train_lable)

    # 使用测试集进行预测
    # y_pred_0 = model.predict(features_train)
    y_pred = model.predict(test_data)
    # print(y_pred)
    # if (model == dtc)or(model == rfc):
    scores = model.predict_proba(test_data)[:, 1]
    auc = metrics.roc_auc_score(test_lable,scores)
    mcc = matthews_corrcoef(test_lable, y_pred)
    accuracy = round(metrics.accuracy_score(test_lable, y_pred) ,4)
    # acc_all[0,i] = accuracy
    # auc_all[0,i] = auc
    # mcc_all[0, i] = mcc
    print(accuracy)
    print(auc)
    print(mcc)
# a = np.vstack((acc_all,auc_all,mcc_all))
# print(a.shape)
# print(a)




