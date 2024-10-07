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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
def xgboost_parameters():
    """模型调参过程"""
    # 第二步：min_child_weight 以及 max_depth
    # 参数的最佳取值：{'max_depth': 2, 'min_child_weight': 1}
    # 最佳模型得分:0.9180952380952381，模型分数未提高
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}

    # 其他参数设置，每次调参将确定的参数加入，不写即默认参数
    fine_params = {'n_estimators': 50}
    return params, fine_params



def model_adjust_parameters(cv_params, other_params,X_train, y_train):
    """模型调参"""
    # 模型基本参数
    model = XGBClassifier(**other_params)
    # sklearn提供的调参工具，训练集k折交叉验证
    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1)
    # 模型训练
    optimized_param.fit(X_train, y_train)
    # 对应参数的k折交叉验证平均得分
    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean_score: %f,  params: %r" % (mean, param))
    # 最佳模型参数
    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
    # 最佳参数模型得分
    print('最佳模型得分:{0}'.format(optimized_param.best_score_))

    # 模型参数调整得分变化曲线绘制
    parameters_score = pd.DataFrame(params, means)
    parameters_score['means_score'] = parameters_score.index
    parameters_score = parameters_score.reset_index(drop=True)
    parameters_score.to_excel('parameters_score.xlsx', index=False)
    # 画图
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(parameters_score.iloc[:, :-1], 'o-')
    plt.legend(parameters_score.columns.to_list()[:-1], loc='upper left')
    plt.title('Parameters_size', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.subplot(2, 1, 2)
    plt.plot(parameters_score.iloc[:, -1], 'r+-')
    plt.legend(parameters_score.columns.to_list()[-1:], loc='upper left')
    plt.title('Score', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.show()
def regression_model(features_train, features_test, Organic_train, Organic_test, model):

    # 构造模型
    model.fit(features_train, Organic_train)

    # 使用测试集进行预测
    y_pred_0 = model.predict(features_train)
    train_score = model.predict_proba(features_train)[:, 1]
    train_acc = round(metrics.accuracy_score(Organic_train, y_pred_0),4)
    train_auc = roc_auc_score(Organic_train, train_score)
    train_cm = confusion_matrix(Organic_train, y_pred_0)
    train_TP = train_cm[1, 1]
    train_TN = train_cm[0, 0]
    train_FP = train_cm[0, 1]
    train_FN = train_cm[1, 0]
    # 敏感性Sensitivity(Sn) 特异性Specificity(SP)
    train_sp = train_TN / float(train_TN + train_FP)
    train_sn = train_TP / float(train_TP + train_FN)
    train_mcc = matthews_corrcoef(Organic_train, y_pred_0)

    y_pred = model.predict(features_test)
    # print(np.sum(y_pred==Organic_test))
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
    # 敏感性Sensitivity(Sn) 特异性Specificity(SP)
    sp = TN / float(TN+FP)
    sn = TP / float(TP+FN)
    mcc = matthews_corrcoef(Organic_test,y_pred)
    # # 均方根误差RMSE

    accuracy = round(metrics.accuracy_score(Organic_test, y_pred),4)
    return accuracy,auc,sp,sn,mcc, train_acc,train_auc,train_sp,train_sn,train_mcc
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

def MSC(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MSC :(n_samples, n_features)
    """
    n, p = data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(data, axis=0)

    # 线性拟合
    for i in range(n):
        y = data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc
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
# 标准化
def SS(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after StandScaler :(n_samples, n_features)
       """
    return StandardScaler().fit_transform(data)
# 趋势校正(DT)
def DT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after DT :(n_samples, n_features)
    """
    x = np.asarray(range(350, 2501), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)
    return out



# 导入数据集切割训练与测试数据

acc_all = np.empty((1, 4))
auc_all = np.empty((1, 4))
sp_all = np.empty((1,4))
sn_all = np.empty((1,4))
mcc_all = np.empty((1,4))
au_all = np.empty((1,4))
trainacc_all = np.empty((1, 4))
trainauc_all = np.empty((1, 4))
trainsp_all = np.empty((1,4))
trainsn_all = np.empty((1,4))
trainmcc_all = np.empty((1,4))
spa = SPA.SPA()
for b in range(1):
    # data_train, data_test, label_train, label_test = train_test_split(features,lables,test_size=0.3,shuffle=True)
    # # 模型训练与拟合linear  rbf  poly
    t0 = time()
    data = pd.read_csv(f'cars1.csv', header=None)
    data = data.values
    # print(data.shape)
    #
    # np.random.shuffle(data)
    # dat = torch.tensor(data, dtype=torch.float)
    # np.savetxt(f"worm11.csv", dat, delimiter=",")

    lables = data[:, -1]
    # data_D = SG(data[:,:-1])
    data_D = data[:,:-1]
    # data_D = MA(data[:,:-1])
    # data_D = DT(data[:,:-1])
    # data_D = SNV(data[:,:-1])
    scaler = StandardScaler()
    features = np.array(data_D)

    # features = TSNE(n_components = 12, method='exact').fit_transform(features)
    # da = np.hstack((features,lables.reshape(660,1)))
    # np.savetxt('tsnevalue1.csv',da,delimiter=',')
    #

    test_data, train_data = features[:165], features[165:]
    test_lable, train_lable = lables[:165], lables[165:]
    # da = np.vstack((train_data,test_data))
    # np.savetxt('da.csv',da,delimiter=)
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.transform(test_data)

    # train_data,mean = mean_centralization(train_data)
    # test_data = pro(test_data,mean)
    # features = np.vstack((train_data, test_data))



    # features = TSNE(n_components = 12).fit_transform(features)
    # train_data, test_data = features[:253], features[253:]
    # da = np.hstack((features,lables.reshape(312,1)))
    # np.savetxt('tsnevalue.csv',da,delimiter=',')
    #
    # lis = CARS.CARS_Cloud(features, lables)
    # print("获取波段数：", len(lis))
    # print(lis)
    # features = features[:,lis]
    # da = np.hstack((features,lables.reshape(-1,1)))
    # np.savetxt('cars1.csv',da,delimiter=',')
    # #
    # train_data, test_data = features[:253], features[253:]
    # print(train_data.shape)
    # lables = lables.reshape(-1, 1)
    # dataa = np.hstack((features,lables))
    # train = dataa[:520]
    # test = dataa[520:]
    # np.random.shuffle(train)
    # np.random.shuffle(test)
    # train_data,train_lable = train[:,:-1],train[:,-1]
    # test_data,test_lable = test[:,:-1],test[:,-1]


    # 建模筛选
    # m_max 默认为 50(Xcal样本大于52) ,如果 Xcal(m*n) m < 50 m_max=m-2
    # var_sel, var_sel_phase2 = spa.spa(
    #     train_data, train_lable, m_min=2, m_max=50, Xval=test_data, yval=test_lable, autoscaling=1)
    # print(var_sel)
    # print(len(var_sel))
    # train_data = train_data[:,var_sel]
    # test_data = test_data[:,var_sel]
    # lable = np.vstack((test_lable.reshape(-1,1),train_lable.reshape(-1,1)))
    # da = np.vstack((test_data,train_data))
    # da = np.hstack((da,lable))
    # np.savetxt('spa2.csv',da,delimiter=',')
    # print(da.shape)

    #

    # pca = FastICA(n_components=10).fit(train_data)
    # pca = PCA(n_components=30).fit(train_data)
    # # 加载LDA模型并训练，降维
    # # LDA为监督学习 需要使用标签信息
    # # print("各主成分的方差值：", pca.explained_variance_)  # 打印方差
    # print("各主成分的方差贡献率：", np.sum(pca.explained_variance_ratio_))  # 打印方差贡献率
    # train_data = pca.transform(train_data)
    # test_data = pca.transform(test_data)
    # print(train_data)
    # fe = np.vstack((test_data,train_data))
    # da = np.hstack((fe,lables.reshape(660,1)))
    # np.savetxt('pcavalue1.csv',da,delimiter=',')
    #
    print(train_data.shape)
    # joblib.dump(value=pca, filename=f'pca_model.m')
#   预处理方法
    ac_list = []
    auc_list = []
    fpr_list = []
    tpr_list = []
    pre_list = []
    recall_list = []
    sp_list = []
    sn_list = []
    mcc_list = []
    trainac_list = []
    trainauc_list = []
    trainsp_list = []
    trainsn_list = []
    trainmcc_list = []
    # 模型预测
#
    dtc = DecisionTreeClassifier()

    rfc = RandomForestClassifier()
    # xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, min_child_weight=1, max_depth=6, gamma=0.0001, subsample=0.9,colsample_bytree=0.6, reg_alpha=0.1)
    svc2 = SVC(kernel='rbf',probability=True,gamma=0.1)

    xgb = XGBClassifier(learning_rate=0.1,n_estimators=200,max_depth=1,min_child_weight=2,gamma=0.0001,subsample=0.8,colsample_bytree=0.8,reg_alpha=0,scale_pos_weight=1,objective= 'binary:logistic')
    #CT n_estimators = 100, min_child_weight = 3, max_depth = 3, gamma = 0.0001, subsample = 0.8, colsample_bytree = 0.8,
    # MA n_estimators=250, min_child_weight=1, max_depth=2, gamma=0.1,subsample=0.5,colsample_bytree=0.6,
    # SG n_estimators=250, min_child_weight=1, max_depth=2, gamma=0.0001,subsample=0.8,colsample_bytree=0.4,
    # raw n_estimators=100, min_child_weight=3, max_depth=3, gamma=0.0001,subsample=0.8,colsample_bytree=0.8,

    # dtc = DecisionTreeClassifier(max_depth=7, max_features=82, min_samples_split=13)
    #
    # rfc = RandomForestClassifier(max_depth=8, max_features=25, min_samples_split=14,
    #                              n_estimators=250)
    #
    # svc2 = SVC(kernel='rbf', probability=True, C=1000, gamma=0.01)
    # xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, min_child_weight=1, max_depth=1, gamma=0.0001,
    #                     subsample=0.9, colsample_bytree=0.8, reg_alpha=0.1)

    Models = [ dtc, rfc, svc2,xgb]
    for i in range(len(Models)):
        ac,auc,sp,sn,mcc,train_ac,train_au,train_sp,train_sn,train_mc= regression_model(train_data,test_data, train_lable, test_lable,
                                                  Models[i])
#
        ac_list.append(ac)
        auc_list.append(auc)
        # pre_list.append(precisions)
        # recall_list.append(recalls)
        # fpr_list.append(fpr)
        # tpr_list.append(tpr)
        sp_list.append(sp)
        sn_list.append(sn)
        mcc_list.append(mcc)
        trainac_list.append(train_ac)
        trainauc_list.append(train_au)
        trainsp_list.append(train_sp)
        trainsn_list.append(train_sn)
        trainmcc_list.append(train_mc)
        # aulist.append(au)
#
#
    print('ac_listof test data',ac_list)
    # print('auc',auclist)
    # print('sp',sp_list)
    # print('sn',sn_list)
    # print('mcc',mcc_list)

#
    acc_all = ac_list
    auc_all = auc_list
    sp_all = sp_list
    sn_all = sn_list
    mcc_all = mcc_list
    trainacc_all = trainac_list
    trainauc_all = trainauc_list
    trainsp_all = trainsp_list
    trainsn_all = trainsn_list
    trainmcc_all = trainmcc_list

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
a11 = np.array(trainacc_all)
# a0 = np.array(au_all)
a21 = np.array(trainauc_all)
a31 = np.array(trainsp_all)
a41 = np.array(trainsn_all)
a51 = np.array(trainmcc_all)
print(a1)
# print(a0)
print(a2)
print(a3)
print(a4)
print(a5)
# a = np.concatenate((a,a1),axis=0)
a = np.vstack((a1,a2,a3,a4,a5))
a_1 = np.vstack((a11,a21,a31,a41,a51))
a = np.vstack((a,a_1))
a = torch.tensor(a)
# #
np.savetxt('all11.csv',a,delimiter=',')




