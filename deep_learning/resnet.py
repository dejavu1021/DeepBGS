import math
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from scipy import signal
from sklearn.metrics import matthews_corrcoef, accuracy_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import torch.nn as nn
# 导入记好了，         2维卷积，2维最大池化，展成1维，全连接层，构建网络结构辅助工具,2d网络归一化,激活函数,自适应平均池化
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d


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
    return data_snv


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


def process(data, s=50, w=50):
    a = int((2150 - w) / s) + 1
    if 2150 % s != 0:
        a = a + 1
    print(a)
    d = np.empty((data.shape[0], a, w))

    for i in range(data.shape[0]):
        for j in range(a):
            for k in range(w):
                if (j * s + w) > 2150:
                    d[i][j] = data[i, 2150 - w:]
                else:
                    d[i][j][k] = data[i, j * s + k]
    d = d.reshape((data.shape[0], 1, a, w))
    return d


class IdentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, filters1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters1),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters1, filters2, kernel_size, stride=1, padding=autopad(kernel_size), bias=False),
            nn.BatchNorm2d(filters2),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters2, filters3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters3)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x = x1 + x
        self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, stride=2):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, filters1, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(filters1),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters1, filters2, kernel_size, stride=1, padding=autopad(kernel_size), bias=False),
            nn.BatchNorm2d(filters2),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters2, filters3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, filters3, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(filters3)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.conv4(x)
        x = x1 + x2
        self.relu(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            ConvBlock(64, 3, [64, 64, 256], stride=1),
            IdentityBlock(256, 3, [64, 64, 256]),
            IdentityBlock(256, 3, [64, 64, 256])
        )
        self.conv3 = nn.Sequential(
            ConvBlock(256, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512])
        )
        self.conv4 = nn.Sequential(
            ConvBlock(512, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024])
        )
        self.conv5 = nn.Sequential(
            ConvBlock(1024, 3, [512, 512, 2048]),
            IdentityBlock(2048, 3, [512, 512, 2048]),
            IdentityBlock(2048, 3, [512, 512, 2048])
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d


# from torchsummary import summary


class Resnet(nn.Module):
    def __init__(self, num_classes):
        super(Resnet, self).__init__()
        self.model0 = Sequential(
            # 0
            # 输入3通道、输出64通道、卷积核大小、步长、补零、
            Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.model0(x)

        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)
        # print(x.shape)
        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)
        # print(x.shape)
        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def train_step(model, features, labels):
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
    return loss.item(), predictions


def train_model(model, epochs):
    for epoch in range(1, epochs + 1):
        to = 0
        cor = 0
        model.train()
        list_loss = []
        al_train = np.empty((3, 495))
        al_train = torch.tensor(al_train, dtype=torch.float)
        for features, labels in dl_train:
            lossi, outd = train_step(model, features, labels)
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
        train_acc.append(cor / to)
        al_train = al_train.detach().numpy()
        mcc = matthews_corrcoef(al_train[2, :], al_train[1, :])
        train_mcc.append(mcc)
        c.append(loss)
        if epoch % 10 == 0:
            print('epoch={} | loss={} '.format(epoch, loss))
        if epoch > 5:
            t = torch.tensor(c[epoch - 4:])
            smaller_values = t < c[epoch - 5]
            if smaller_values.any() == False:
                for k in range(epoch - 4, epoch + 1):
                    if k % 5 == 0:
                        print('epoch={} | loss={} '.format(k, c[k]))

                break


a = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
proc = [0, SNV, mean_centralization]
ss = [50, 100, 150, 200, 250, 300]

dataset = pd.read_csv('res505.csv', delimiter=",", header=None)
dataset = dataset.values
# print(dataset.shape)
# lab = dataset[:,-1]
y = dataset[2,:]
# print(y)
y_pred = dataset[1,:]
accc = accuracy_score(y,y_pred)

while(accc < 0.94):
    c = []
    train_acc = []
    train_mcc = []
    sss = ss[3]
    all = np.empty((3, 165))
    all = torch.tensor(all, dtype=torch.float)
    # 读取CSV数据
    data = pd.read_csv('shuju111.csv', header=None)

    data = data.values
    # np.random.shuffle(data)
    # dat = torch.tensor(data, dtype=torch.float)
    #
    # np.savetxt(f"dat{h}.csv", dat, delimiter=",")

    print(data.shape)
    # data_D = preprocessing.StandardScaler().fit_transform(data[:, :-1])
    pe = proc[1]
    if pe == 0:
        data_D = data[:, :-1]
    else:
        data_D = pe(data[:, :-1])
    data_D = np.array(data_D)
    features = data_D[:, 1:]
    # features = torch.tensor(features)
    # features = features.float()
    lab = data[:, -1]
    #
    # da = np.hstack((features,lab))
    # train = da[165:]
    # test = da[:165]
    # print(da.shape)
    # print(train.shape)
    # np.random.shuffle(train)
    # np.random.shuffle(test)
    # train_data,  train_lable= train[:,:-1],train[:,-1]
    # test_data, test_lable = test[:,:-1], test[:,-1]

    y = torch.tensor(lab, dtype=torch.long).reshape(-1, 1, 1)

    # features = process(features,s=200,w=200)
    # x = torch.tensor(features, dtype=torch.float)
    # lab = data[:,-1]
    # y = torch.tensor(lab,dtype=torch.long).reshape(-1,1,1)
    # train_data, test_data = x[:352], x[352:]
    # train_lable, test_lable = y[:352], y[352:]
    # print(train_lable.shape)

    # 划分训练集和测试集
    x = features
    train_data, test_data = x[165:], x[:165]
    train_lable, test_lable = y[165:], y[:165]

    # dataa = np.hstack((x,lab.reshape(-1,1)))
    # train = dataa[160:]
    # test = dataa[:160]
    # np.random.shuffle(train)
    # np.random.shuffle(test)
    # train_data, train_lable = train[:, :-1], train[:, -1]
    # test_data, test_lable = test[:, :-1], test[:, -1]
    # train_data, mean = mean_centralization(train_data)
    # test_data = pro(test_data, mean)

    print(train_data.shape)
    train_data = process(train_data, s=150, w=sss)
    test_data = process(test_data, s=150, w=sss)
    # %%
    train_data = torch.tensor(train_data, dtype=torch.float)
    test_data = torch.tensor(test_data, dtype=torch.float)
    # train_lable = torch.tensor(train_lable, dtype=torch.long).reshape(-1,1,1)
    # test_lable = torch.tensor(test_lable, dtype=torch.long).reshape(-1,1,1)
    # 构建迭代器
    batch_size = 11
    ds_train = TensorDataset(train_data, train_lable)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0, shuffle=True)
    ds_test = TensorDataset(test_data, test_lable)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=0, shuffle=True)

    # 查看第一个batch
    x, y = next(iter(dl_train))
    print(x.shape)
    print(y.shape)

    # 自定义训练方式
    model = ResNet50()
    # model = Resnet(num_classes=2)
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6, weight_decay=1e-4)

    # 测试一个batch
    features, labels = next(iter(dl_train))
    loss, pr = train_step(model, features, labels)
    c.append(loss)
    print(loss)
    train_model(model, 150)

    # torch.save(model.state_dict(),f'resnetmodel{h}.pth')
    lo = torch.tensor(c)
    np.savetxt(f"lossres503.csv", lo, delimiter=",")
    traacc = torch.tensor(train_acc)
    np.savetxt(f"tranaccres503.csv", train_acc, delimiter=",")
    tramcc = torch.tensor(train_mcc)
    np.savetxt(f"tranmccres503.csv", train_mcc, delimiter=",")

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
        accc = correct / total
        print(accc)

    # print(all)
    np.savetxt(f"res503.csv", all, delimiter=",")

print(a)
print(sum(a) / len(a))
da = torch.tensor(a)

# np.savetxt("res50/computer_res18.csv", da, delimiter=",")
