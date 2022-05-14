import os
import random
import Convolutional_Classical_Neural_Net.LeNet as L
import paddle
import numpy as np
from paddle.vision.transforms import ToTensor
from paddle.vision.datasets import MNIST
import paddle.nn.functional as F
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
model = L.LeNet(num_class=10)
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
