import joblib
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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

seq_len = 64
input = 60
class Block(nn.Module):
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
        x = self.attention(x)

        x = self.pool(x)
        # x = self.attention(x)

        return x
class Net(nn.Module):
    def __init__(self,input_size=input, hidden_size=32, num_layers=2, num_classes=2):
        super().__init__()
        # left
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.c = Block(1,16)
        self.c1 = Block(16,32)
        self.c2 = Block(32,64)
        self.c3 = Block(64,128)
        self.c4 = Block(128,256)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bd1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bd2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
        # self.fc = nn.Linear(hidden_size, num_classes)
        self.at = CBAM(16)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size , 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.Linear(64, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        x1 = self.conv_1(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        # x1 = self.c(x)
        # x1 = self.at(x1)
        x1 = self.c1(x1)
        x1 = self.c2(x1)
        x1 = self.c3(x1)
        x1 = self.conv1(x1)
        x1 = self.bd1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bd2(x1)
        x1 = self.relu2(x1)
        # x1 = self.at(x1)
        x1 = x1.view(x1.size(0), seq_len,-1)
        # print(x1.shape)

        out, (lat_hidden, last_cell) = self.lstm(x1, (h0, c0))
        # print(out.shape)
        out = out[:, -1, :]
        x1 = self.fc(out)
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


def train_model(model, epochs):
    for epoch  in range(1, epochs+1):
        to = 0
        cor = 0
        model.train()
        list_loss = []
        for features, labels in train_loader:

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
proo = [0,SNV,MA,SG,CT]
ss = [50,100,150,200,250,300]
best_score = 0
bestvalue = None
for h in range(1):
    b=0
    train_acc = []
    sss = ss[1]
    all = np.empty((10,3, 48))
    all = torch.tensor(all, dtype=torch.float)
# 读取CSV数据
    data = pd.read_csv('shuju111.csv',header=None)

    data = data.values
    pe = proo[1]
    print(data.shape)
    # data_D = preprocessing.StandardScaler().fit_transform(data[:, :-1])
    if pe == 0:
        data_D = data[:, :-1]
    else:
        data_D = pe(data[:, :-1])
    data_D = np.array(data_D)
    features = data_D[:, 1:]
    features = torch.tensor(features)
    x = features.float()
    lab = data[:,-1]
    y = torch.tensor(lab,dtype=torch.long).reshape(-1,1,1)

    # 划分训练集和测试集
    train_data, test_data = x[165:], x[:165]
    train_lable, test_lable = y[165:], y[:165]
    train_data = process(train_data, s=50, w=sss)
    test_data = process(test_data, s=50, w=sss)
    print(train_lable.shape)
    #%%
    train_data = torch.tensor(train_data, dtype=torch.float)
    # 构建迭代器
    batch_size = 8
    ds_train = TensorDataset(train_data, train_lable)
    # dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0,shuffle=True)

    # 自定义训练方式
    loss_function = torch.nn.CrossEntropyLoss()
    # model = Net()
    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=1e-4)

    kfold = KFold(n_splits=10, shuffle=True,random_state=42)
    i = 0
    # # 4. 训练数据
    # 训练和评估模型
    for train_index, test_index in kfold.split(train_data, train_lable,):
        i = i+1
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_lable[train_index], train_lable[test_index]
        print(X_train.shape)
        print(test_index)
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

        features, labels = next(iter(train_loader))
        # if i==1:
        #     for j in range(3):
        #         print(test_index)
        c = []
        model = Net()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss,pre = train_step(model, features, labels)
        c.append(loss)
        print(loss)
        train_model(model, 150)

        # 评估模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                out = model.forward(features)
                labels = labels.reshape(batch_size)
                _, predicted = torch.max(out.data, 1)
                m = nn.Softmax(dim=1)
                outputs_softmax = m(out)

                all[b,0, total:total + batch_size] = outputs_softmax[:, 1]
                all[b,1, total:total + batch_size] = predicted
                all[b,2, total:total + batch_size] = labels

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy: {100 * correct / total:.2f}%')
        acc = 100 * correct / total
        a.append(acc)
        b+=1
    avg_score = np.mean(np.array(a))
    print(avg_score)
    print(a)
    # print(all)
    all = all.numpy()
    all_transposed = all.transpose((1, 0, 2))
    alll = all_transposed.reshape(3, 480)
    alll = torch.tensor(alll)
    np.savetxt("valid/train0_1.csv", alll, delimiter=",")
    # if avg_score > best_score:
    #     best_score = avg_score
    #     bestvalue = pe

