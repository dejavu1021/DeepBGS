import joblib
from sklearn.metrics import roc_auc_score
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
def process(data, s=50, w=50):

    a = int((2150-w)/s)+1
    if 2150 % s != 0:
        a=a+1
    print(a)
    d = np.empty((data.shape[0], a, w))

    for i in range(data.shape[0]):
        for j in range(a):
            for k in range(w):
                if (j*s+w) >2150:
                    d[i][j] = data[i, 2150-w:]
                else:
                    d[i][j][k] = data[i, j * s + k]
    d = d.reshape((data.shape[0], 1, a, w))
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

        self.conv = nn.Conv2d(2, 1, 3, 1, 1, bias=False)  # 3x3conv
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

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bd1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bd2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bd3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention = CBAM(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bd1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bd2(x)
        x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.bd3(x)
        # x = self.relu3(x)
        # x = self.attention(x)

        x = self.pool(x)
        # x = self.attention(x)

        return x
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # left
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.c = VGGBlock(1,16)
        self.c1 = VGGBlock(16,32)
        self.c2 = VGGBlock(32,64)
        self.c3 = VGGBlock(64,128)
        self.c4 = VGGBlock(128,256)
        self.at = CBAM(16)
        self.fc = nn.Sequential(
            nn.Linear(3200 , 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2)
        )

    def forward(self, x):

        x1 = self.conv_1(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        # x1 = self.c(x)
        # x1 = self.at(x1)
        x1 = self.c1(x1)
        x1 = self.c2(x1)
        x1 = self.c3(x1)
        # x1 = self.c4(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        return x1


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

def test_step(model,features, labels):
    # 正向传播求损失
    features = features.to(device)
    labels = labels.to(device)
    out = model.forward(features)
    # print(predictions.shape)
    labe = labels.clone().detach().reshape(batch_size)
    # print(predictions.shape)
    # print(labels.shape)
    loss = loss_function(out, labe)
    _, predicted = torch.max(out.data, 1)
    cor = (predicted == labels).sum().item()
    # loss = loss_function(predictions, labels)
    # 反向传播求梯度
    loss.backward()
    # 参数更新
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(),cor

def train_model(model, epochs):
    for epoch  in range(1, epochs+1):
        to = 0
        cor = 0
        model.train()
        list_loss = []
        for features, labels in dl_train:

            lossi,outd = train_step(model,features, labels)
            labe = labels.clone().detach().reshape(batch_size).to(device)
            _, predicted = torch.max(outd.data, 1)
            to += labe.size(0)
            cor += (predicted == labe).sum().item()
            list_loss.append(lossi)
        loss = np.mean(list_loss)
        train_acc.append(100 * cor / to)
        c.append(loss)
        if epoch % 10 == 0:
            print('epoch={} | loss={} '.format(epoch,loss))
        if epoch > 5:
            t = torch.tensor(c[epoch - 2:])
            smaller_values = t < c[epoch - 3]
            if smaller_values.any() == False:
                for k in range(epoch - 3, epoch+1):
                    if k % 5 == 0:
                        print('epoch={} | loss={} '.format(k, c[k]))

                break

a=[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
pro = [0,SNV]
ss = [50,100,150,200,250,300]
for h in range(5):
    c = []
    train_acc = []
    sss = ss[3]
    all = np.empty((4, 130))
    all = torch.tensor(all, dtype=torch.float)
# 读取CSV数据
    data = pd.read_csv('../result.csv',header=None)

    data = data.values
    # np.random.shuffle(data)
    # dat = torch.tensor(data, dtype=torch.float)
    #
    # np.savetxt(f"dataa{h}.csv", dat, delimiter=",")

    print(data.shape)
    # data_D = preprocessing.StandardScaler().fit_transform(data[:, :-1])
    pe = pro[1]
    if pe == 0:
        data_D = data[:, :-1]
    else:
        data_D = pe(data[:, :-1])
    data_D = np.array(data_D)
    features = data_D[:, 1:]
    features = torch.tensor(features)
    features = features.float()
    lab = data[:,-1]
    y = torch.tensor(lab,dtype=torch.long).reshape(-1,1,1)
    # features = process(features,s=200,w=200)
    # x = torch.tensor(features, dtype=torch.float)
    # lab = data[:,-1]
    # y = torch.tensor(lab,dtype=torch.long).reshape(-1,1,1)
    # train_data, test_data = x[:352], x[352:]
    # train_lable, test_lable = y[:352], y[352:]
    # print(train_lable.shape)

    # 划分训练集和测试集
    x = features
    # train_data, test_data = x[:352], x[352:]
    # train_lable, test_lable = y[:352], y[352:]
    train_data, test_data = x[:520], x[520:]
    train_lable, test_lable = y[:520],y[520:]
    train_data = process(train_data,s=150,w=sss)
    test_data = process(test_data,s=150,w=sss)
    #%%
    train_data = torch.tensor(train_data, dtype=torch.float)
    test_data = torch.tensor(test_data, dtype=torch.float)
    # 构建迭代器
    batch_size = 10
    ds_train = TensorDataset(train_data, train_lable)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0, shuffle=True)
    ds_test = TensorDataset(test_data, test_lable)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=0, shuffle=True)

    # 查看第一个batch
    x, y = next(iter(dl_train))
    print(x.shape)
    print(y.shape)

    # 自定义训练方式
    loss_function = torch.nn.CrossEntropyLoss()
    model = Net()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6,weight_decay=1e-4)

    # 测试一个batch
    features, labels = next(iter(dl_train))
    loss ,pr= train_step(model, features, labels)
    c.append(loss)
    print(loss)
    train_model(model, 150)

    torch.save(model.state_dict(),f'blockno{h}.pth')
    lo = torch.tensor(c)
    # np.savetxt(f"loss_yes{h}.csv", lo, delimiter=",")
    traacc = torch.tensor(train_acc)
    # np.savetxt(f"tranacc_yes{h}.csv", traacc, delimiter=",")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in dl_test:
            features = features.to(device)
            labels = labels.to(device)
            labels = labels.reshape(batch_size)


            out = model.forward(features)
            # print('out',out)
            _, predicted = torch.max(out.data, 1)
            # print('label',labels)
            # print('predicted',predicted)
            m = nn.Softmax(dim=1)
            outputs_softmax = m(out)
            # print('outputs_softmax[:, 1]',outputs_softmax[:, 1])
            y_soce = outputs_softmax[:, 1]
            # auc_score = roc_auc_score(labels, y_soce)

            all[0, total:total + 10] = outputs_softmax[:, 0]
            all[1, total:total + 10] = outputs_softmax[:, 1]
            all[2, total:total + 10] = predicted
            all[3, total:total + 10] = labels
            # print(predicted)
            # print(all[2])
            # print(labels)
            # print(all[3])
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f}%')
        a.append(100 * correct / total)
    # print(all)
    np.savetxt(f"blockno{h}.csv", all, delimiter=",")

print(a)
print(sum(a) / len(a))
da = torch.tensor(a)
np.savetxt("computer2_22.csv", da, delimiter=",")

