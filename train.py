import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder

# 如果有GPU可用，优先使用GPU, 否则使用CPU运算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为PyTorch张量,并将每个像素值从[0, 255]范围内缩放到[0.0, 1.0]
])

# 自动下载并加载MNIST数据集，并应用定义好的预处理步骤
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 用于批量加载数据，每批数据包含32个样本，并在每个epoch随机打乱数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = Autoencoder() # 实例化自编码器（AE）模型
model.to(device) # 如果cuda可用，则将模型从CPU移动到GPU上进行计算

criterion = nn.MSELoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3) # 优化器，学习率为0.001

num_epochs = 20 # 训练周期
lowest_loss = float('inf')  # 初始化最低损失为正无穷，用于跟踪保存最好的模型

for epoch in range(num_epochs):
    total_loss = 0.0
    for data in train_loader:
        # 自编码器只要图像数据，不需要标签. 
        # 图像张量形状：[batch_size, 1, 28, 28]
        # 其中，batch_size由train_loader指定，28为Mnist数据集图像的宽和高，1代表图像只有一个通道，是黑白图片。（彩色图片有RGB三个通道）
        img, _ = data 
        img = img.to(device) # 将数据从CPU移动到指定的设备
        
        img = img.view(img.size(0), -1) # 将图像数据展平 形状：[batch_size, 28 * 28]
        output = model(img)  # 通过模型前向传播得到重建的输出 形状：[batch_size, 28 * 28]
        loss = criterion(output, img)  # 计算重建图像与原图之间的损失
        
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新模型参数
        
        total_loss += loss.item() # 累加损失

    avg_loss = total_loss / len(train_loader)  # 计算这个epoch的平均损失

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # 如果损失是目前为止最低的，保存模型
    if avg_loss < lowest_loss:
        lowest_loss = avg_loss
        torch.save(model.state_dict(), 'best.pth')  # 保存模型
        print(f'New lowest average loss {lowest_loss:.4f} at epoch {epoch+1}, model saved.')