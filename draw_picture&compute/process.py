from copy import deepcopy

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib import gridspec,rcParams
from matplotlib.ticker import FormatStrFormatter
from proplot import rc
from scipy import signal
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

def SS(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after StandScaler :(n_samples, n_features)
       """
    return StandardScaler().fit_transform(data)

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
# standard Normal Variate transform  标准正态变量变换
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
    # data_std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    # print(data_std)
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # print(data_average)
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return  data_snv

def mean_centralization(sdata):
    """
    均值中心化
    """
    sdata = deepcopy(sdata)
    temp1 = np.mean(sdata, axis=0)
    temp2 = np.tile(temp1, sdata.shape[0]).reshape(
        (sdata.shape[0], sdata.shape[1]))
    return sdata - temp2, temp1


# 设置

matplotlib.rcdefaults()
print(rcParams["font.family"])
rcParams["font.size"] = 15
rcParams["axes.labelsize"] = 15
rcParams["axes.titlesize"] = 15
print(rcParams["axes.titlesize"])
# rcParams["font.family"] = ''
# 计算650个曲线
'''
data = dataset = pd.read_csv('dataa9.csv', delimiter=",",header=None)
# data = pd.DataFrame(data)
data = data.values
print(data)

# data['lable'] = [0,1,1]
print(data.shape)
c = ['y','Wavelength','Reflectance','sample']

a = np.arange(350,2502)
a = [str(b) for b in a]
a[2151] = 'y'
# dataname =
lab = data[:,-1]
lables = lab.reshape(-1, 1)
pro = [0,SNV,MA,SG,mean_centralization]
# gs = gridspec.GridSpec(2, 6) # 创立2 * 6 网格
# gs.update(wspace=0.8)
# # 对第一行进行绘制
# ax1 = plt.subplot(gs[0, :2]) # gs(哪一行，绘制网格列的范围)
# ax2 = plt.subplot(gs[0, 2:4])
# ax3 = plt.subplot(gs[0, 4:6])
# # 对第二行进行绘制
# ax4 = plt.subplot(gs[1, 1:3])
# ax5 = plt.subplot(gs[1, 3:5])

# for i in range(5):
# data_D = data[:, :-1]
# data_D1 = pro[1](data[:, :-1])
# data_D2 = pro[2](data[:, :-1])
# data_D3 = pro[3](data[:, :-1])
data_D4,aaa = pro[4](data[:, :-1])
#
# data = np.hstack((data_D, lables))
# data = pd.DataFrame(data)
# data.columns = a
# print(data)

# data1 = np.hstack((data_D1, lables))
# data1 = pd.DataFrame(data1)
# data1.columns = a
# print(data1)
#
# data2 = np.hstack((data_D2, lables))
# data2 = pd.DataFrame(data2)
# data2.columns = a
# print(data2)

# data3 = np.hstack((data_D3, lables))
# data3 = pd.DataFrame(data3)
# data3.columns = a
# print(data3)
# #
data4 = np.hstack((data_D4, lables))
data4 = pd.DataFrame(data4)
data4.columns = a
print(data4)

# data10=data.melt(id_vars=['y'])
# print(data10)
# data10['sample'] = list(range(650))*2151
# data10.columns = c
# print(data10)
#
# data11=data1.melt(id_vars=['y'])
# print(data11)
# data11['sample'] = list(range(650))*2151
# data11.columns = c
# print(data11)
# #
# data12=data2.melt(id_vars=['y'])
# print(data12)
# data12['sample'] = list(range(650))*2151
# data12.columns = c
# print(data12)
# #
# data13=data3.melt(id_vars=['y'])
# print(data13)
# data13['sample'] = list(range(650))*2151
# data13.columns = c
# print(data13)
# #
data14=data4.melt(id_vars=['y'])
print(data14)
data14['sample'] = list(range(650))*2151
data14.columns = c
print(data14)
fig, ax4= plt.subplots(figsize=(8,7.2))
# ax = sns.lineplot(x="Wavelength", y="Reflectance", hue="y",units="sample", estimator=None, lw=0.8,data=data10,alpha=0.3)
# ax1 = sns.lineplot(x="Wavelength", y="Reflectance", hue="y",units="sample", estimator=None, lw=0.8,data=data11,alpha=0.3)
# ax2 = sns.lineplot(x="Wavelength", y="Reflectance", hue="y",units="sample", estimator=None, lw=0.8,data=data12,alpha=0.3)
# ax3 = sns.lineplot(x="Wavelength", y="Reflectance", hue="y",units="sample", estimator=None, lw=0.8,data=data13,alpha=0.3)
ax4 = sns.lineplot(x="Wavelength", y="Reflectance", hue="y",units="sample", estimator=None, lw=0.8,data=data14,alpha=0.3)
# sns.lineplot(x="wavelength", y="reflectance", hue="y",lw=0.8,data=data11,alpha=0.5,ax=ax1,errorbar='sd')

# ax1.set_xlim([-50,2200])
# ax1.set_xticks([0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200],[350,550,750,950,1150,1350,1550,1750,1950,2150,2350,2550])
plt.xlim([-50,2200])
plt.xticks([0,400,800,1200,1600,2000],[350,750,1150,1550,1950,2350]) # 指定轴组标签，tick
# plt.xticks([0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200],[350,550,750,950,1150,1350,1550,1750,1950,2150,2350,2550])
ax4.get_legend().remove()
# 设置y轴刻度标签保留两位小数
y_formatter = FormatStrFormatter('%1.2f')
ax4.yaxis.set_major_formatter(y_formatter)

# 展示图形
# plt.show()

# plt.legend(loc = 'best')
# plt.xticks([100,250,500,750,1000,1250,1500,1750,2000,2250,2500],[100,350,600,850,1100,1350,1600,1850,2100,2350,2600]) # 指定轴组标签，tick
plt.savefig('CT.png', dpi=800)
plt.show()

'''

#计算auc，mcc,acc

# fig, ax2 = plt.subplots(figsize=(7, 5))
# sns.boxplot(x="type", y="value", data=df_final,
#               hue="variable",width=0.9, linewidth=1.0, palette=['#55A0FD','#F8403E','#C6C6C6'],
#             saturation=1.0,fliersize=0)
# for label in ax2.get_xticklabels():
#    label.set_rotation(90)
# plt.rcParams['xtick.direction'] = 'in'

f, ax = plt.subplots(figsize=(8, 7.2))
data = dataset = pd.read_csv('mcc.csv', delimiter=",",header=0)
data = pd.DataFrame(data)
print(data)
cor = sns.hls_palette(5)

ax = sns.barplot(x="Preprocessing", y="Mcc", hue="model",data=data,ax = ax,linewidth = 2,palette=sns.hls_palette(5,l=0.75,s=0.70))
bars = plt.gca().patches
bars = np.array(bars)
print(bars[1])
# print(bars.shape)
# 设置边框颜色
cor = sns.hls_palette(5,l=0.5)
print(cor[2])
# data = pd.DataFrame(data)
i = 0
# print(len(ax.patches))
bars[0].set_edgecolor(cor[0], )
bars[1].set_edgecolor(cor[0], )
bars[2].set_edgecolor(cor[0], )
bars[3].set_edgecolor(cor[0], )
bars[4].set_edgecolor(cor[0], )
bars[5].set_edgecolor(cor[1], )
bars[6].set_edgecolor(cor[1], )
bars[7].set_edgecolor(cor[1], )
bars[8].set_edgecolor(cor[1], )
bars[9].set_edgecolor(cor[1], )
bars[10].set_edgecolor(cor[2], )
bars[11].set_edgecolor(cor[2], )
bars[12].set_edgecolor(cor[2], )
bars[13].set_edgecolor(cor[2], )
bars[14].set_edgecolor(cor[2], )
bars[15].set_edgecolor(cor[3], )
bars[16].set_edgecolor(cor[3], )
bars[17].set_edgecolor(cor[3], )
bars[18].set_edgecolor(cor[3], )
bars[19].set_edgecolor(cor[3], )
bars[20].set_edgecolor(cor[4], )
bars[21].set_edgecolor(cor[4], )
bars[22].set_edgecolor(cor[4], )
bars[23].set_edgecolor(cor[4], )
bars[24].set_edgecolor(cor[4], )
# for i in range(25):
#     i = int(i/5)
#     bars[i].set_edgecolor(cor[i])
#     # print(int(i/5))
#     if i/5 == 0:
#         ax.patches[i].set_edgecolor('black')
#     else:
#         ax.patches[i].set_edgecolor('blue')

#     print(bars[i])
    # print(cor[i])
    # if i % 5 == 0:
    #     bar.set_edgecolor(cor[0])
    # if i % 5 == 1:
    #     bar.set_edgecolor(cor[1])
    # if i % 5 == 2:
    #     bar.set_edgecolor(cor[2])
    # if i % 5 == 3:
    #     bar.set_edgecolor(cor[3])
    # if i % 5 == 4:
    #     bar.set_edgecolor(cor[4])

    # i = i + 1
# sns.set(font_scale =3)
ax.set_xlabel('Preprocessing',fontsize=15)
# plt.xticks(fontsize=14)
plt.ylim([0.859,0.961])
plt.yticks([0.84,0.88,0.92,0.96,1.00]) # 指定轴组标签，tick
# plt.ylim([0.90,1.001])
# plt.yticks([0.92,0.94,0.96,0.98,1],fontsize=14) # 指定轴组标签，tick
ax.get_legend().remove()
# plt.legend(loc = 'best')
plt.savefig('mcc.png', dpi=800)
plt.show()



# roc面积
'''
dataset1 = pd.read_csv('compare2/blockcbam_0.csv', delimiter=",",header=None)
dataset1 = dataset1.values
y_true1 = dataset1[3,:]
y_pred1 = dataset1[2,:]
y_soce1 = dataset1[1,:]

dataset2 = pd.read_csv('compare2/blockno2.csv', delimiter=",",header=None)
dataset2 = dataset2.values
y_true2 = dataset2[3,:]
y_pred2 = dataset2[2,:]
y_soce2 = dataset2[1,:]

dataset3 = pd.read_csv('compare2/vgg1_4.csv', delimiter=",",header=None)
dataset3 = dataset3.values
y_true3 = dataset3[3,:]
y_pred3 = dataset3[2,:]
y_soce3 = dataset3[1,:]

dataset4 = pd.read_csv('compare2/resnet/res1_4.csv', delimiter=",",header=None)
dataset4 = dataset4.values
y_true4 = dataset4[3,:]
y_pred4 = dataset4[2,:]
y_soce4 = dataset4[1,:]

fpr1, tpr1, _ = roc_curve(y_true1, y_soce1)
roc_auc1 = auc(fpr1, tpr1)
fpr2, tpr2, _ = roc_curve(y_true2, y_soce2)
roc_auc2 = auc(fpr2, tpr2)
fpr3, tpr3, _ = roc_curve(y_true3, y_soce3)
roc_auc3 = auc(fpr3, tpr3)
fpr4, tpr4, _ = roc_curve(y_true4, y_soce4)
roc_auc4 = auc(fpr4, tpr4)
# 将数据转换为DataFrame格式
data = pd.DataFrame({ 'False Positive Rate': np.concatenate([fpr1, fpr2,fpr3,fpr4]),
                      'True Positive Rate': np.concatenate([tpr1, tpr2,tpr3,tpr4]),
                      'Model': ['Block_cbam'] * len(fpr1) + ['Block'] * len(fpr2)+['Vgg'] * len(fpr3) + ['Resnet'] * len(fpr4) })
fig, ax = plt.subplots(figsize=(8, 7.2))
plt.ylim([0.8,1.01])
plt.yticks([0.8,0.85,0.90,0.95,1])
plt.xlim([-0.05,1.02])
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
sns.lineplot(ax=ax,x="False Positive Rate", y="True Positive Rate", hue="Model", estimator=None, lw=3,data=data,alpha=0.5,palette=['#55A0FD','#F8403E','#0a090b','#8d37e6'])# sns.set(font_scale =3)
plt.legend(loc = 'best',labels=['DeepBGS(AUC={:.4f})'.format(roc_auc1),'DeepBGS-NoCBAM(AUC={:.4f})'.format(roc_auc2),'Vgg(AUC={:.4f})'.format(roc_auc3),'Resnet(AUC={:.4f})'.format(roc_auc4)])
plt.savefig('roc.png', dpi=800)
# ax.get_legend().remove()
plt.show()
'''

'''
# resnet_train compare
data = dataset = pd.read_csv('res.csv', delimiter=",",header=0)
data = pd.DataFrame(data)
print(data)
fig, ax = plt.subplots(figsize=(8, 7.2))
sns.lineplot(ax= ax,x="echo", y="value", hue="class",data=data,estimator=None, lw=2,alpha=0.5,palette=['#55A0FD','#F8403E','#0a090b'])
# sns.set(font_scale =3)
ax.set_xlabel('Echo')
ax.set_ylabel('Value')

# plt.xticks(fontsize=14)
plt.xlim([-0.05,20.1])
plt.xticks([0,5,10,15,20])
# plt.ylim([0.859,0.961])
plt.yticks(fontsize=14) # 指定轴组标签，tick
# plt.ylim([0.90,1.001])
# plt.yticks([0.92,0.94,0.96,0.98,1],fontsize=14) # 指定轴组  标签，tick
plt.legend(loc = 'best',labels=['TrainLoss','Acc','Mcc'])

# ax.get_legend().remove()
# plt.legend(loc = 'best')
plt.savefig('res50.png', dpi=800)
plt.show()
'''

'''
# 热图

matplotlib.rcdefaults()
print(rcParams["font.family"])
rcParams["font.size"] = 15
rcParams["axes.labelsize"] = 15
rcParams["axes.titlesize"] = 15
print(rcParams["axes.titlesize"])
X = np.arange(50,350,50)
# Y坐标数据
Y = np.arange(50,350,50)
print(X)

df= pd.read_csv('acc_step_dimension.csv',header=0)
# print(df)
data = df.values
print(data)
# 使用figure对象
# fig = plt.figure(figsize=(8,4.8))

# ax = plt.axes(projection='3d')
# # fig.add_axes(ax)
# X, Y = np.meshgrid(X, Y)
# ax.plot_surface(X,Y,data,cmap='rainbow')
# print(fig.get_size_inches())
# plt.show()

f, ax = plt.subplots(figsize=(8,7.2))
tick = np.arange(0.96,1,0.01).astype(float)
dict ={"ticks":tick}

#heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
#参数annot=True表示在对应模块中注释值
# 参数linewidths是控制网格间间隔
#参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
#参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
sns.heatmap(data, ax=ax,vmin=0.96,vmax=1,cmap='YlOrRd',annot=True,linewidths=2,cbar=True, fmt=".4f",cbar_kws= dict,xticklabels=['50','100','150','200','250','300'] , #x轴方向刻度标签开关、赋值，可选“auto”, bool, list-like（传入列表）, or int,
            yticklabels=['50', '100', '150', '200', '250', '300'])#y轴方向刻度标签开关、同x轴)

# ax.set_title('ACC') #plt.title('热图'),均可设置图片标题
ax.set_ylabel('Window_size')  #设置纵轴标签
ax.set_xlabel('Step')  #设置横轴标签

#设置坐标字体方向，通过rotation参数可以调节旋转角度
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.savefig('acc_step_dimesion2.png', dpi=800)

plt.show()
'''