import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder

# 定义数据预处理步骤：与训练相同。
transform = transforms.Compose([
    transforms.ToTensor(), 
])

# 实例化Autoencoder模型。
model = Autoencoder()
# 加载预训练的模型权重。
model.load_state_dict(torch.load('./best.pth'))

# 将模型设置为评估模式。
model.eval()

# 自动下载并加载MNIST数据集，并应用定义好的预处理步骤
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 使用DataLoader来批量加载数据。这里设置batch_size=1表示每批只处理一个图像，
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# 使用迭代器从数据加载器中取出一个批次的数据。
dataiter = iter(train_loader)
images, labels = next(dataiter)
img = images[0].view(images[0].size(0), -1) # 每个batch中，只取第一个图像 形状：[1, 28*28]

# 使用torch.inference_mode()上下文管理器来进行推理，以获得更好的性能。
with torch.inference_mode():
    output = model(img)

# 将tensor转换为numpy数组
img = img.view(28, 28).numpy() # 形状：[28, 28]
output = output.view(28, 28).numpy() # 形状：[28, 28]

# 使用matplotlib保存重构后的图像。
plt.imsave('reconstructed_image.png', output, cmap='gray')

# 绘制原始图像和重构图像
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img, cmap='gray') # 显示原始图像。
axes[0].set_title('Original Image')
axes[1].imshow(output, cmap='gray') # 显示重构图像。
axes[1].set_title('Reconstructed Image')
plt.show()