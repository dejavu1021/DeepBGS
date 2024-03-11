import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from scipy import signal
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import torch.nn as nn

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

class VGG(nn.Module):
    def __init__(self,num_classes=2):
        super(VGG,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),



        )
        self.classifier = nn.Sequential(
            nn.Linear(6400,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes),
        )
    def forward(self,x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
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
for h in range(1):
    c = []
    train_acc = []
    sss = ss[3]
    all = np.empty((4, 130))
    all = torch.tensor(all, dtype=torch.float)
# 读取CSV数据
    data = pd.read_csv('result.csv',header=None)

    data = data.values
    # np.random.shuffle(data)
    # dat = torch.tensor(data, dtype=torch.float)
    #
    # np.savetxt(f"dat{h}.csv", dat, delimiter=",")

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
    model = VGG()
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6,weight_decay=1e-4)

    # 测试一个batch
    features, labels = next(iter(dl_train))
    loss ,pr= train_step(model, features, labels)
    c.append(loss)
    print(loss)
    train_model(model, 150)

    # torch.save(model.state_dict(),f'vggmodel{h}.pth')
    lo = torch.tensor(c)
    # np.savetxt(f"loss{h}.csv", lo, delimiter=",")
    traacc = torch.tensor(train_acc)
    # np.savetxt(f"tranacc{h}.csv", traacc, delimiter=",")
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
    # np.savetxt(f"vgg1_{h}.csv", all, delimiter=",")

print(a)
print(sum(a) / len(a))
da = torch.tensor(a)
# np.savetxt("computer_vg1.csv", da, delimiter=",")