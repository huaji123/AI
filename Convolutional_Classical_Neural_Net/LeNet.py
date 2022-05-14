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

