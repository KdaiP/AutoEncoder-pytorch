import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder

# 如果有GPU可用，优先使用GPU, 否则使用CPU运算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# 定义数据预处理步骤：与训练相同。
transform = transforms.Compose([
    transforms.ToTensor(), 
])

# 实例化Autoencoder模型并移动到相应的设备上。
model = Autoencoder().to(device)
# 加载预训练的模型权重。
model.load_state_dict(torch.load('./best.pth'))

# 将模型设置为评估模式。
model.eval()

# 自动下载并加载MNIST数据集，并应用定义好的预处理步骤
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 使用DataLoader来批量加载数据。
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 使用torch.inference_mode()上下文管理器来进行推理，以获得更好的性能。
with torch.inference_mode():
    # 初始化两个空的Tensor，用于存储所有图像的特征和标签
    features = torch.empty((0, 3), device=device)
    labels = torch.empty((0), device=device)
    
    # 提取训练集所有图像的特征
    for data in train_loader:
        img, label = data 
        
        # 将图像和标签移动到指定设备上
        # non_blocking=True：启用异步传输，提升传输效率
        img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
        
        img = img.view(img.size(0), -1) # 将图像数据展平 形状：[batch_size, 28 * 28]
        feature = model.encoder(img)  # 通过模型的encoder，得到压缩的图像特征 形状：[batch_size, 3]
        
        # 将当前批次的特征向量拼接到之前的特征向量上，dim=0表示沿着批次（batch）维度拼接
        features = torch.cat((features, feature), dim=0)
        # 将当前批次的标签拼接到之前的标签上，同样是沿着批次（batch）维度拼接
        labels = torch.cat((labels, label), dim=0)

# 将特征和标签数据从设备（如GPU）转回CPU，并转换为NumPy数组，以便于使用matplotlib进行可视化
features = features.cpu().numpy()
labels = labels.cpu().numpy()

# 创建一个matplotlib图形
fig = plt.figure()
# 添加一个3D子图
ax = fig.add_subplot(111, projection='3d')

# 使用3D散点图可视化特征，颜色根据标签变化，这里使用的是tab10颜色映射
scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap='tab10')

# 添加图例，显示不同标签的数据点代表的数字
legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
ax.add_artist(legend1)

# 设置3D散点图的X、Y、Z坐标轴标签
ax.set_xlabel('X Feature')
ax.set_ylabel('Y Feature')
ax.set_zlabel('Z Feature')

# 显示图形
plt.show()