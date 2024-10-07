from copy import deepcopy

import joblib
from sklearn.metrics import matthews_corrcoef
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.nn import AdaptiveAvgPool2d
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np

from scipy import signal
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
def SG(data, w=11, p=2,d=0):
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

def process(data):
    d = np.empty((data.shape[0], 40, 200))
    s = 50
    w = 200
    for i in range(data.shape[0]):
        for j in range(40):
            for k in range(200):
                d[i][j][k] = data[i, j * s + k]
    return d

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

def mean_centralization(sdata):
    """
    均值中心化
    """
    sdata = deepcopy(sdata)
    temp1 = np.mean(sdata, axis=0)
    temp2 = np.tile(temp1, sdata.shape[0]).reshape(
        (sdata.shape[0], sdata.shape[1]))
    return sdata - temp2, temp1


def pro(data, me):
    sdata = deepcopy(data)
    temp2 = np.tile(me, sdata.shape[0]).reshape(
        (sdata.shape[0], sdata.shape[1]))
    return sdata - temp2

class ChannelAttention(nn.Module):  # Channel attention module
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        if channels>=ratio:
           hidden_channels = channels // ratio
        else:
            hidden_channels = 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # global avg pool
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # global max pool
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=False),  # 1x1conv代替全连接
            nn.ReLU(inplace=True),  # relu
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=False)  # 1x1conv代替全连接
        )
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        return self.sigmoid(
            self.mlp(x_avg) + self.mlp(x_max)
        )  #


class SpatialAttention(nn.Module):  # Spatial attention module
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, (1,3), 1, (0,1), bias=False)  # 3x3conv
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # 在通道维度上进行avgpool
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # 在通道维度上进行maxpool
        return self.sigmoid(
            self.conv(torch.cat([x_avg, x_max],dim=1))
        )


class CBAM(nn.Module):  # Convolutional Block Attention Module
    def __init__(self, channels, ratio=16 ):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, ratio)  # Channel attention module
        self.spatial_attention = SpatialAttention()  # Spatial attention module

    def forward(self, x):
        f1 = self.channel_attention(x) * x
        f2 = self.spatial_attention(f1) * f1
        return f2


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.att = CBAM(1)
        self.fc1 = nn.Linear(2151,512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x= self.att(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_step(model,features, labels):
    # 正向传播求损失
    features = features.to(device)
    labels = labels.to(device)
    predictions = model.forward(features)
    # print(predictions.shape)
    labe = labels.clone().detach().reshape(batch_size)
    # print(predictions.shape)
    # print(labels.shape)
    loss = loss_function(predictions, labe)
    # loss = loss_function(predictions, labels)
    # 反向传播求梯度
    loss.backward()
    # 参数更新
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(),predictions
def train_model(model, epochs):
    for epoch  in range(1, epochs+1):
        to = 0
        cor = 0
        model.train()
        list_loss = []
        al_train = np.empty((3, 495))
        al_train = torch.tensor(al_train, dtype=torch.float)
        for features, labels in dl_train:

            lossi,outd = train_step(model,features, labels)
            labe = labels.clone().detach().reshape(batch_size).to(device)
            _, predicted = torch.max(outd.data, 1)
            # print(outd)
            m = nn.Softmax(dim=1)
            outputs_softmax = m(outd)
            # print(outputs_softmax[:, 0])
            al_train[0, to:to + batch_size] = outputs_softmax[:, 1]
            al_train[1, to:to + batch_size] = predicted
            al_train[2, to:to + batch_size] = labe
            # print(al_train[2])
            to += labe.size(0)
            cor += (predicted == labe).sum().item()
            list_loss.append(lossi)
        loss = np.mean(list_loss)
        train_acc.append( cor / to)
        al_train = al_train.detach().numpy()
        mcc = matthews_corrcoef(al_train[2,:],al_train[1,:])
        train_mcc.append(mcc)
        c.append(loss)
        if epoch % 10 == 0:
            print('epoch={} | loss={} '.format(epoch,loss))
        if epoch > 5:
            t = torch.tensor(c[epoch - 2:])
            smaller_values = t < c[epoch - 3]
            if smaller_values.any() == False:
                for k in range(epoch - 2, epoch+1):
                    if k % 5 == 0:
                        print('epoch={} | loss={} '.format(k, c[k]))

                break

a=[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
proo = [0,SNV,MA,SG,CT]

for h in range(1):
    c = []
    all = np.empty((3, 165))
    all = torch.tensor(all, dtype=torch.float)
    train_acc = []
    train_mcc = []
# 读取CSV数据
    data = pd.read_csv('shuju111.csv',header=None)

    data = data.values
    # np.random.shuffle(data)
    # dat = torch.tensor(data, dtype=torch.float)
    #
    # np.savetxt(f"dat{h}.csv", dat, delimiter=",")

    print(data.shape)
    # data_D = preprocessing.StandardScaler().fit_transform(data[:, :-1])
    pe = proo[1]
    if pe == 0:
        data_D = data[:, :-1]
    else:
        data_D = pe(data[:, :-1])

    features = data_D
    # features = torch.tensor(features)
    x = features
    lab = data[:,-1]
    y = torch.tensor(lab,dtype=torch.long)
    # 划分训练集和测试集
    train_data, test_data = x[165:], x[:165]
    train_lable, test_lable = y[165:],y[:165]
    # train_data,mean = mean_centralization(train_data)
    # test_data = pro(test_data,mean)
    train_data = torch.tensor(train_data,dtype=torch.float).reshape(495, 1, 1, 2151)
    test_data = torch.tensor(test_data,dtype=torch.float).reshape(165, 1, 1, 2151)
    print(train_lable.shape)
    #%%
    # 构建迭代器
    batch_size = 11
    # ds = TensorDataset(x, y)
    # dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
    ds_train = TensorDataset(train_data, train_lable)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0,shuffle=True)
    ds_test = TensorDataset(test_data, test_lable)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=0,shuffle=True)
    # 查看第一个batch
    x, y = next(iter(dl_train))
    print(x.shape)
    print(y.shape)
    # 自定义训练方式
    model = MLP()
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 测试一个batch0
    features, labels = next(iter(dl_train))
    loss, pr = train_step(model, features, labels)
    c.append(loss)
    print(loss)
    train_model(model, 150)

    # torch.save(model.state_dict(),f'resnetmodel{h}.pth')
    lo = torch.tensor(c)
    np.savetxt(f"lossmlp2.csv", lo, delimiter=",")
    traacc = torch.tensor(train_acc)
    np.savetxt(f"tranaccmlp2.csv", train_acc, delimiter=",")
    tramcc = torch.tensor(train_mcc)
    np.savetxt(f"tranmccmlp2.csv", train_mcc, delimiter=",")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in dl_test:
            features = features.to(device)
            labels = labels.to(device)
            out = model.forward(features)
            labels = labels.reshape(batch_size)
            _, predicted = torch.max(out.data, 1)
            m = nn.Softmax(dim=1)
            outputs_softmax = m(out)
            # print(outputs_softmax[:, 0])
            # all[0, total:total + batch_size] = outputs_softmax[:, 0]
            all[0, total:total + batch_size] = outputs_softmax[:, 1]
            all[1, total:total + batch_size] = predicted
            all[2, total:total + batch_size] = labels
            # print(predicted)
            # print(all[2])
            # print(labels)
            # print(all[3])
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f}%')
        a.append(100 * correct / total)
    # print(all)
    np.savetxt(f"mlp_snvv2.csv", all, delimiter=",")

print(a)
print(sum(a) / len(a))
da = torch.tensor(a)
