import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
from PIL import Image
import numpy as np


def test_metrics(weight):
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载测试数据集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # 加载预训练的ResNet模型
    model = torchvision.models.resnet18(pretrained=True)
    num_classes = 10  # 请根据您的数据集修改类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weight))  # 加载您之前训练好的模型权重

    # 设置模型为评估模式
    model.eval()

    # 初始化变量来存储真实标签和模型的预测
    true_labels = []
    predicted_labels = []

    # 遍历测试数据集并进行预测
    with torch.no_grad():
        for inputs, labels in test_loader:  # test_dataloader 包含测试数据
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())  # 将真实标签添加到列表
            predicted_labels.extend(predicted.cpu().numpy())  # 将模型的预测添加到列表

    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Accuracy: {accuracy:.4f}')

    # 计算精确率和召回率
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')


def test_image(weight):
    
    # 加载预训练的ResNet模型
    model = torchvision.models.resnet18(pretrained=True)
    num_classes = 10  # 请根据您的数据集修改类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weight))  # 加载您之前训练好的模型权重

    # 设置模型为评估模式
    model.eval()    

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试图像（替换为你要测试的图像路径）
    image_path = 'path_to_your_image.jpg'
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度

    # 使用模型进行预测
    with torch.no_grad():
        output = model(image)

    # 获取类别标签
    predicted_class = output.argmax().item()

    # 加载CIFAR-10类别名称映射文件
    cifar10_class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # 输出预测结果
    print(f'Predicted class: {cifar10_class_names[predicted_class]}')

if __name__ == '__main__':

    weight = "./resnet_model_10_epoch.pth"
    test_metrics(weight)