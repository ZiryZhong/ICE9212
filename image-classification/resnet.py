import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter   
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
import shutil
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 设置随机种子以确保结果可重复
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(torch.cuda.current_device())
print(torch.cuda.is_available())

# 定义模型
resnet = torchvision.models.resnet18(pretrained=True)
 
# 修改模型
resnet.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
resnet.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
num_ftrs = resnet.fc.in_features  # 获取（fc）层的输入的特征数
resnet.fc = nn.Linear(num_ftrs, 10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.1, weight_decay=5e-4)

# 数据预处理和加载
train_transform = transforms.Compose([
            transforms.ToTensor()
            , transforms.RandomCrop(32, padding=4)  # 先四周填充0，在吧图像随机裁剪成32*32
            , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ])

valid_transform = transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

Batch_Size = 64
# train_dataset = datasets.ImageFolder('path_to_train_data', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 加载训练数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=Batch_Size, shuffle=True)

# # 加载测试数据集
valid_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=Batch_Size, shuffle=False)


# 训练模型
num_epochs = 100  # 根据需要修改
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
resnet.to(device)
log_train = open("./log/train.log","w")
log_valid = open("./log/valid.log","w")

# tensorboard 可视化
writer = SummaryWriter('./tensorboard')
shutil.rmtree("./tensorboard")
l = len((train_loader))
l_v = len(valid_loader)

truth_list = []
pred_list = []

for epoch in range(num_epochs):
    pbar = tqdm.tqdm(total = l)
    resnet.train()
    loss = 0
    val_loss = 0
    running_loss = 0
    running_corrects = 0
    truth_list.clear()
    pred_list.clear()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pbar.update(1)

        preds = outputs.argmax(dim=1)
        # 计算损失值
        # print(images.size(0))
        truth_list.append(labels.cpu().detach().numpy())
        pred_list.append(preds.cpu().detach().numpy())

        running_loss += loss.item() * images.size(0)  # loss计算的是平均值，所以要乘上batch-size，计算损失的总和
        # running_corrects += (preds == labels).sum()  # 计算预测正确总个数
    
    y_true = np.concatenate(truth_list)
    y_pred = np.concatenate(pred_list)

    epoch_train_accuracy = accuracy_score(y_true,y_pred)
    epoch_train_precision, epoch_train_recall, epoch_train_f1 = precision_recall_fscore_support(y_true,y_pred,average='macro', zero_division='warn')[:-1]
    epoch_train_loss = running_loss / len(train_loader) / Batch_Size # 当前轮的总体平均损失值
    # epoch_train_acc = float(running_corrects) / len(train_loader) / Batch_Size  # 当前轮的总正确率

    running_loss = 0
    running_corrects = 0
    
    truth_list.clear()
    pred_list.clear()
    resnet.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(total = l_v)
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(images)
            val_loss = criterion(outputs, labels)
            pbar.update(1)

            preds = outputs.argmax(dim=1)

            truth_list.append(labels.cpu().detach().numpy())
            pred_list.append(preds.cpu().detach().numpy())
            # 计算损失值
            running_loss += val_loss.item() * images.size(0)  # loss计算的是平均值，所以要乘上batch-size，计算损失的总和
            # running_corrects += (preds == labels).sum()  # 计算预测正确总个数

    y_true = np.concatenate(truth_list)
    y_pred = np.concatenate(pred_list)

    epoch_valid_accuracy = accuracy_score(y_true,y_pred)
    epoch_valid_precision, epoch_valid_recall, epoch_valid_f1 = precision_recall_fscore_support(y_true,y_pred,average='macro', zero_division='warn')[:-1]
    epoch_valid_loss = running_loss / len(valid_loader) / Batch_Size # 当前轮的总体平均损失值
    # epoch_valid_acc = float(running_corrects) / len(valid_loader) / Batch_Size # 当前轮的总正确率

    # 写入记录文件
    writer.add_scalars("LOSS", {"train_loss": epoch_train_loss,"valid_loss": epoch_valid_loss}, epoch)
    writer.add_scalars("ACC", {"train_accuracy": epoch_train_accuracy,"valid_accuracy": epoch_valid_accuracy}, epoch)
    writer.add_scalars("PRECISION", {"train_precision": epoch_train_precision,"valid_precision": epoch_valid_precision}, epoch)
    writer.add_scalars("RECALL", {"train_recall": epoch_train_recall,"valid_recall": epoch_valid_recall}, epoch)
    writer.add_scalars("F1", {"train_f1": epoch_train_f1,"valid_f1": epoch_valid_f1}, epoch)

    if epoch % 10 == 0:
        torch.save(resnet.state_dict(), './weights/resnet_model_' + str(epoch) +"_"+ str(epoch_valid_loss) + '_epoch.pth')
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train_Loss: {epoch_train_loss}, Valid_Loss: {epoch_valid_loss}\
          , Train_ACC: {epoch_train_accuracy}, Valid_ACC: {epoch_valid_accuracy}')
    
    log_train.write(f'{epoch_train_loss},{epoch_train_accuracy},{epoch_train_precision},{epoch_train_recall},{epoch_train_f1}')
    log_valid.write(f'{epoch_valid_loss},{epoch_valid_accuracy},{epoch_valid_precision},{epoch_valid_recall},{epoch_valid_f1}')

# 保存训练好的模型
# torch.save(resnet.state_dict(), 'resnet_model_final.pth')
log_train.close()
log_valid.close()