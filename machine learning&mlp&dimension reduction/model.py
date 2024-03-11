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
    scores = model.predict_proba(features_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(Organic_test, scores)
    precisions, recalls, thresholds = precision_recall_curve(Organic_test, scores)

    # auc的输入为很简单，就是fpr, tpr值（至于这两个怎么算请看上面基本参数的介绍）
    auc = metrics.auc(fpr, tpr)
    cm = confusion_matrix(Organic_test,y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    tpr = TP/(TP+FN)
    print(tpr)
    # 敏感性Sensitivity(Sn) 特异性Specificity(SP)
    sp = TN / float(TN+FP)
    sn = TP / float(TP+FN)
    mcc = matthews_corrcoef(Organic_test,y_pred)
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

# 导入数据集切割训练与测试数据

acc_all = np.empty((1, 4))
auc_all = np.empty((1, 4))
sp_all = np.empty((1,4))
sn_all = np.empty((1,4))
mcc_all = np.empty((1,4))
au_all = np.empty((1,4))
spa = SPA.SPA()
for b in range(1):
    # data_train, data_test, label_train, label_test = train_test_split(features,lables,test_size=0.3,shuffle=True)
    # # 模型训练与拟合linear  rbf  poly
    t0 = time()
    data = pd.read_csv(f'result.csv', header=None,dtype=np.float32)
    data = data.values
    print(data.shape)
    # data = torch.tensor(data, dtype=torch.float)
    # print(data)

    lables = data[:, -1]
    # data_D = SG(data[:,:-1])
    # data_D = data[:,:-1]
    # data_D = MA(data[:,:-1])
    data_D = SNV(data[:,:-1])
    features = np.array(data_D)

    # features = TSNE(n_components = 8, method='exact').fit_transform(features)
    # da = np.hstack((features,lables.reshape(650,1)))
    # np.savetxt('tsnevalue.csv',da,delimiter=',')
    #
    # lis = CARS.CARS_Cloud(features, lables)
    # print("获取波段数：", len(lis))
    # print(lis)
    # features = features[:,lis]
    # da = np.hstack((features,lables.reshape(-1,1)))
    # np.savetxt('cars22.csv',da,delimiter=',')

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

    # 建模筛选
    # m_max 默认为 50(Xcal样本大于52) ,如果 Xcal(m*n) m < 50 m_max=m-2
    var_sel, var_sel_phase2 = spa.spa(
        train_data, train_lable, m_min=2, m_max=50, Xval=test_data, yval=test_lable, autoscaling=1)
    print(var_sel)
    print(len(var_sel))
    train_data = train_data[:,var_sel]
    test_data = test_data[:,var_sel]
    da = np.vstack((train_data,test_data))
    da = np.hstack((da,lables.reshape(-1,1)))
    np.savetxt('spa2.csv',da,delimiter=',')
    print(da.shape)

    # train_data,mean = mean_centralization(train_data)
    # test_data = pro(test_data,mean)
    #

    # pca = PCA(n_components =0.98).fit(train_data)
    # train_data = pca.transform(train_data)
    # test_data = pca.transform(test_data)
    # fe = np.vstack((train_data,test_data))
    # da = np.hstack((fe,lables.reshape(650,1)))
    # np.savetxt('pcavalue.csv',da,delimiter=',')
    #
    # print(train_data.shape)
    # joblib.dump(value=pca, filename=f'pca_model.m')
#   预处理方法
    ac_list = []
    auclist = []
    aulist = []
    fpr_list = []
    tpr_list = []
    pre_list = []
    recall_list = []
    sp_list = []
    sn_list = []
    mcc_list = []
    # 模型预测
#
    dtc = DecisionTreeClassifier()

    rfc = RandomForestClassifier()

    svc = SVC(kernel='rbf',probability=True)
    xgb = XGBClassifier()
    Models = [ dtc, rfc,svc,xgb]
    for i in range(len(Models)):
        ac,auc,precisions,recalls,fpr,tpr,sp,sn,mcc= regression_model(train_data,test_data, train_lable, test_lable,
                                                  Models[i])
#
        ac_list.append(ac)
        auclist.append(auc)
        pre_list.append(precisions)
        recall_list.append(recalls)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        sp_list.append(sp)
        sn_list.append(sn)
        mcc_list.append(mcc)
        # aulist.append(au)
#
#
    print('ac_listof test data',ac_list)
    # print('auc',auclist)
    # print('sp',sp_list)
    # print('sn',sn_list)
    # print('mcc',mcc_list)

#
    acc_all[b] = ac_list
    auc_all[b] = auclist
    sp_all[b] = sp_list
    sn_all[b] = sn_list
    mcc_all[b] = mcc_list
    # au_all[b] = aulist
    # print('fpr',fpr_list)
    # print('tpr',tpr_list)
    # print('pre',pre_list)
    # print('recall',recall_list)
    # tpr_all[b] = tpr_list
    # pre_all[b] = pre_list
    # recall_all[b] = recall_list


a1 = np.array(acc_all)
# a0 = np.array(au_all)
a2 = np.array(auc_all)
a3 = np.array(sp_all)
a4 = np.array(sn_all)
a5 = np.array(mcc_all)

print(a1)
# print(a0)
print(a2)
print(a3)
print(a4)
print(a5)
# a = np.concatenate((a,a1),axis=0)
a = np.vstack((a1,a2,a5))
a = torch.tensor(a)
# #
# np.savetxt('all0_2.csv',a,delimiter=',')




