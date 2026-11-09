import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据：生成 x 和对应的真实 y (y = sin(x))
# 在 -π 到 π 之间生成 2000 个点
x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 2000), dim=1)
# 计算 sin(x)，并加上一点点随机噪声（模拟真实世界的数据不完美）
y = torch.sin(x) + 0.1 * torch.rand(x.size())

# 2. 搭建神经网络
# 输入是 1 维（x），隐藏层 10 个神经元，输出是 1 维（预测的 y）
net = nn.Sequential(
    nn.Linear(1, 10),   # 全连接层：1个输入特征 -> 10个隐藏神经元
    nn.ReLU(),          # 激活函数：引入非线性，让网络能拟合曲线
    nn.Linear(10, 1)    # 全连接层：10个隐藏神经元 -> 1个输出特征
)

# 3. 定义优化器和损失函数
# Adam 是常用的优化器，lr (学习率) 设为 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# MSELoss 是均方误差，衡量预测值和真实值的差距
loss_func = nn.MSELoss()

# 4. 开始训练！
print("开始训练...")
for t in range(1000): # 训练 1000 次
    prediction = net(x)                  # 1. 前向传播：把 x 喂给网络，得到预测值
    loss = loss_func(prediction, y)      # 2. 计算误差：预测值和真实 y 的差距
    
    optimizer.zero_grad()                # 3. 清空上一步的残余梯度
    loss.backward()                      # 4. 反向传播：计算网络参数应该怎么改
    optimizer.step()                     # 5. 更新参数：真正改变网络里的权重
    
    if t % 100 == 0:
        print(f'训练步数: {t}, 损失值(Loss): {loss.item():.4f}')

print("训练完成！")

# 5. 画图看看学得怎么样
# 关闭科学计算模式，防止画图卡住
plt.figure(figsize=(8, 5))
# 画出加了噪声的真实数据（蓝色散点）
plt.scatter(x.data.numpy(), y.data.numpy(), s=10, color='blue', label='True Data (sin x)')
# 画出神经网络预测出的曲线（红色实线）
plt.plot(x.data.numpy(), prediction.data.numpy(), color='red', linewidth=3, label='Neural Network Prediction')

plt.title('PyTorch Learning: Fit sin(x)')
plt.legend()
plt.grid(True)
plt.show()