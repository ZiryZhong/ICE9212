import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 设置随机种子以确保结果可重复
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 使用预训练的ResNet模型
resnet = models.resnet50(pretrained=True)

# 冻结所有参数，除了最后一层全连接层
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc.requires_grad = True

# 更改最后一层的输出类别数
num_classes = 10  # 请根据您的数据集修改类别数
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder('path_to_train_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练模型
num_epochs = 10  # 根据需要修改
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存训练好的模型
torch.save(resnet.state_dict(), 'resnet_model.pth')
