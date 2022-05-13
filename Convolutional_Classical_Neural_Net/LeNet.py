import paddle
import paddle.nn as N
import numpy as np
# 处理的图像是28*28*1，只有一个通道所以选择二维
# Conv2D:卷积层 MaxPool2D:最大池化层 Linear:线性变化层
from paddle.nn import Conv2D, MaxPool2D, Linear, Conv3D

# 组网
import paddle.nn.functional as F


# 定义LeNet网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_class=1):
        super(LeNet, self).__init__()
        self.layer1 = N.Sequential(
            # 创建第一个卷积层和池化层
            N.Conv2D(in_channels=1, out_channels=6, kernel_size=5),
            N.Sigmoid(),
            N.MaxPool2D(kernel_size=2, stride=2),

        )
        self.layer2 = N.Sequential(
            # 尺寸的逻辑：池化层为改变通道数：当前通道数为6
            # 创建第二个卷积层
            N.Sigmoid(),
            N.Conv2D(in_channels=6, out_channels=16, kernel_size=5),
            N.MaxPool2D(kernel_size=2, stride=2)
        )
        self.layer3 = N.Sequential(
            # 创建第三个卷积层
            N.Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        )
        self.layer4 = N.Sequential(
            # 尺寸的逻辑：输入层将数据拉平[B,C, H, W] -> [B, C*H*W]
            # 输入size是[28,28],经过三次卷积和两次池化之后，C*H*W=120
            N.Linear(in_features=120, out_features=64),
            N.Sigmoid(),
            # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
            N.Linear(in_features=64, out_features=num_class)
        )

    def forward(self, x):
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 拉直操作
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.layer4(x)

        return x


# # 输入数据形状是[N, 1, H, W]
# # 这里用np.random创建一个随机数组作为输入数据
# x = np.random.randn()
#
# # 创建LeNet实例，指定模型名称和分类的类别数目
# model = LeNet(num_class=10)
#
# # 通过调用LeNet从基类继承的sublayers()函数，
# # 查看LeNet中所包含的子层
# print(model.sublayers())
#
# # 将x转换为tensor类型
# x = paddle.to_tensor(x)
# # 查看经过LeNet-5的每一层作用之后，输出数据的形状
# for item in model.sublayers():
#     # item是LeNet类中的一个子层
#     # 查看经过子层之后的输出数据形状
#     try:
#         x = item(x)
#     except:
#         x = paddle.reshape(x, [x.shape[0], -1])
#         x = item(x)
#     if len(item.parameters()) == 2:
#         # 查看卷积和全连接层的数据和参数的形状，
#         # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
#         print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
#     else:
#         # 池化层没有参数
#         print(item.full_name(), x.shape)

import os
import random
import paddle
import numpy as np
from paddle.vision.transforms import ToTensor
from paddle.vision.datasets import MNIST

# 存放精确度
accuracies = []
# 存放损失值
losses = []

# 定义训练过程
# model：数据模型
# opt:优化函数
# train_loader:训练集
# valid_loader:验证集
def train(model, opt, train_loader, valid_loader):
    # 开启gpu训练
    use_gpu = True
    # paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
    print("start training...")
    # 如果模型中有BN层和Dropout，需要在训练时添加model.train(),在测试时添加model.eval()。
    # 其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差
    # 训练集，计算损失函数
    # 验证集，计算精确度
    model.train()
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 数据特征
            img = data[0]
            # 数据标签
            label = data[1]
            # 计算模型输出
            logits = model(img)
            # 计算损失函数，交叉熵
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            # 损失均值
            avg_loss = paddle.mean(loss)
            # 每2000次输出一次
            if batch_id % 2000 == 0:
                print("epoch:{}, batch_id:{}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))
            # 一定要写
            # 反向传播
            avg_loss.backward()
            # 最小化loss，更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

        model.eval()

        for batch_id, data in enumerate(valid_loader()):
            img = data[0]
            label = data[1]
            logits = model(img)

            # 获得预测值
            pred = F.softmax(logits)

            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)

            # 获得精确度
            acc = paddle.metric.accuracy(pred, label)

            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        model.train()

    # 保存模型参数
    paddle.save(model.state_dict(), '../mnist.pdparams')


# 创建模型
model = LeNet(num_class=10)
# model = AlexNet.Alex(num_classes=10)

# 设置迭代轮数
EPOCH_NUM = 5

# 设置优化器为Moementum，学习率为0.001
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())

# 定义数据读取起
train_loader = paddle.io.DataLoader(MNIST(mode='train', transform=ToTensor()), batch_size=10, shuffle=True)
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=10)

# 启动训练过程
train(model, opt, train_loader, valid_loader)
