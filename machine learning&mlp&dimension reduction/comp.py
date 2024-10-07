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

def predict(y_scores, threholds):
    return (y_scores >= threholds) * 1
def compute_scores(y_true, y_pred):
    p_score = precision_score(y_true, y_pred)
    r_score = recall_score(y_true, y_pred)
    return p_score, r_score
def p_r_curve(y_true, y_scores):
    thresholds = sorted(np.unique(y_scores))
    precisions, recalls = [], []
    for thre in thresholds:
        y_pred = predict(y_scores, thre)
        r = compute_scores(y_true, y_pred)
        precisions.append(r[0])
        recalls.append(r[1])
    # 去掉召回率中末尾重复的情况
    last_ind = np.searchsorted(recalls[::-1], recalls[0]) + 1
    precisions = precisions[-last_ind:]
    recalls = recalls[-last_ind:]
    thresholds = thresholds[-last_ind:]
    precisions.append(1)
    recalls.append(0)
    return precisions, recalls, thresholds
# AUC值的计算
def compute_ap(recall, precision):
    # \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n
    rp = [item for item in zip(recall, precision)][::-1]  # 按recall升序进行排序
    ap = 0
    for i in range(1, len(rp)):
        ap += (rp[i][0] - rp[i - 1][0]) * rp[i][1]
        # print(f"({rp[i][0]} - {rp[i - 1][0]}) * {rp[i][1]}")
    return ap
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
def wave(self, data_x):  # 小波变换
        data_x = deepcopy(data_x)
        if isinstance(data_x, pd.DataFrame):
            data_x = data_x.values
        def wave_(data_x):
            w = pywt.Wavelet('db8')  # 选用Daubechies8小波
            maxlev = pywt.dwt_max_level(len(data_x), w.dec_len)
            coeffs = pywt.wavedec(data_x, 'db8', level=maxlev)
            threshold = 0.04
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            datarec = pywt.waverec(coeffs, 'db8')
            return datarec

        tmp = None
        for i in range(data_x.shape[0]):
            if (i == 0):
                tmp = wave_(data_x[i])
            else:
                tmp = np.vstack((tmp, wave_(data_x[i])))
        return tmp

def pro(data,me):
    sdata = deepcopy(data)
    temp2 = np.tile(me, sdata.shape[0]).reshape(
        (sdata.shape[0], sdata.shape[1]))
    return sdata - temp2

def baseline_correction(data, window_size=11, poly_order=3):
    # 平滑数据
    smoothed_data = savgol_filter(data, window_size, poly_order, axis=1)

    # 计算平均光谱
    mean_spectrum = np.mean(smoothed_data, axis=0)

    # 拟合多项式到平均光谱
    X = np.arange(len(mean_spectrum)).reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=poly_order, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, mean_spectrum)
    baseline = lin_reg.predict(X_poly)

    # 去除基线
    corrected_data = smoothed_data - baseline.reshape(1, -1)

    return corrected_data

# # 导入数据集切割训练与测试数据
tvi = np.empty((660,1))
npci = np.empty((660,1))
pri = np.empty((660,1))
nir = np.empty((660,8))
nvid = np.empty((660,80))
rvi = np.empty((660,80))
vb = np.empty((660,10))
data = pd.read_csv(f'shuju111.csv', header=None)
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

min = np.min(rvi,axis=1).reshape(660,1)
# print(min)
max = np.max(rvi,axis=1).reshape(660,1)
print(min.shape)
nvid = np.array(nvid).reshape(660,80)
# rvi = np.array((rvi-min)/(max-min)).reshape(650,80)
print(rvi.shape)
# 0.5[120(750-550)-200（670-550）】
tvi = 0.5*(120*(features[:,400]-features[:,200])-200*(features[:,320]-features[:,200]))
tvi = np.array(tvi).reshape(660,1)
print(tvi)
# 680-430 / 680+430
npci = (features[:,330]-features[:,80])/(features[:,330]+features[:,80])
npci =  np.array(npci).reshape(660,1)
# 531-570 / 531+570
pri = (features[:,181]-features[:,220])/(features[:,181]+features[:,220])
pri = pri.reshape(660,1)

dat = np.hstack((nvid,rvi,npci,pri,tvi,lables.reshape(660,1)))
# print(all.shape)
np.savetxt('Index1.csv',dat,delimiter=',')
'''


acc_all = np.empty((1, 4))
auc_all = np.empty((1, 4))
sp_all = np.empty((1,5))
sn_all = np.empty((1,5))
mcc_all = np.empty((1,4))
au_all = np.empty((1,5))
for b in range(1):
    # data_train, data_test, label_train, label_test = train_test_split(features,lables,test_size=0.3,shuffle=True)
    # # 模型训练与拟合linear  rbf  poly
    t0 = time()
    data = pd.read_csv(f'da.csv', header=None)
    data = data.values
    print(data.shape)
    # np.random.shuffle(dat)
    # dat = torch.tensor(data, dtype=torch.float)
    # np.savetxt(f"d29.csv", dat, delimiter=",")
    # print(data)

    lables = data[:, -1]
    # data_D = SG(data[:,:-1])
    data_D = data[:,:-1]
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
    train_data, test_data = features[165:], features[:165]
    train_lable, test_lable = lables[165:], lables[:165]
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



    # train_data,mean = mean_centralization(train_data)
    # test_data = pro(test_data,mean)
    # print(train_lable.shape)
    # print(train_data.shape)
    # print(test_data.shape)
    #

    # ica = FastICA(n_components=20).fit(train_data)

    # joblib.dump(value=pca, filename=f'pca_model{b}.m')
#   预处理方法
#     ac_list = []
#     auclist = []
#     aulist = []
#     fpr_list = []
#     tpr_list = []
#     pre_list = []
#     recall_list = []
#     sp_list = []
#     sn_list = []
#     mcc_list = []
#     # 模型预测
# #
#     dtc = DecisionTreeClassifier()
#
#     rfc = RandomForestClassifier()
#
#     svc = SVC(kernel='linear',probability=True)
#     svc2 = SVC(kernel='rbf',probability=True)
#     # svc2 = SVC(kernel='linear')
#     xgb = XGBClassifier()
# #     # adj_params, fixed_params = xgboost_parameters()
# #     # model_adjust_parameters(adj_params, fixed_params)
#     Models = [ dtc, rfc, svc,svc2,xgb]
#     for i in range(len(Models)):
#         ac,auc,precisions,recalls,fpr,tpr,sp,sn,mcc= regression_model(train_data,test_data, train_lable, test_lable,
#                                                   Models[i])
# #
#         ac_list.append(ac)
#         auclist.append(auc)
#         pre_list.append(precisions)
#         recall_list.append(recalls)
#         fpr_list.append(fpr)
#         tpr_list.append(tpr)
#         sp_list.append(sp)
#         sn_list.append(sn)
#         mcc_list.append(mcc)
#         # aulist.append(au)
# #
# #
#     print('ac_listof test data',ac_list)
#     # print('auc',auclist)
#     # print('sp',sp_list)
#     # print('sn',sn_list)
#     # print('mcc',mcc_list)
#
# #
#     acc_all[b] = ac_list
#     auc_all[b] = auclist
#     sp_all[b] = sp_list
#     sn_all[b] = sn_list
#     mcc_all[b] = mcc_list
#     # au_all[b] = aulist
#     # print('fpr',fpr_list)
#     # print('tpr',tpr_list)
#     # print('pre',pre_list)
#     # print('recall',recall_list)
#     # tpr_all[b] = tpr_list
#     # pre_all[b] = pre_list
#     # recall_all[b] = recall_list
#
#
# a1 = np.array(acc_all)
# # a0 = np.array(au_all)
# a2 = np.array(auc_all)
# a3 = np.array(sp_all)
# a4 = np.array(sn_all)
# a5 = np.array(mcc_all)
#
# print(a1)
# # print(a0)
# print(a2)
# print(a3)
# print(a4)
# print(a5)
# # a = np.concatenate((a,a1),axis=0)
# a = np.vstack((a1,a2,a3,a4,a5))
# a = torch.tensor(a)
# # #
# np.savetxt('all1.csv',a,delimiter=',')
'''




