import os
import clip
import torch

import numpy as np
from torchvision.datasets import CIFAR100, CIFAR10
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from clip_linear_probing_model import LinearModel, MyDataSet

def do_linear_probing(DATASET,EPOCH,LOAD_WEIGHTS,MODEL_SAVE_PATH):

	# Load the model
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# model, preprocess = clip.load('ViT-B/32', device)
	model, preprocess = clip.load('RN50', device)
	
	# Load the dataset
	if DATASET == "CIFAR100":
		linear_model = LinearModel(100).to(device)  #实例化

		train = CIFAR100("./data", download=False, train=True, transform=preprocess)
		test = CIFAR100("./data", download=False, train=False, transform=preprocess)
	elif DATASET == "CIFAR10":
		linear_model = LinearModel(10).to(device)  #实例化

		train = CIFAR10("../image-classification/data", download=False, train=True, transform=preprocess)
		test = CIFAR10("../image-classification/data", download=False, train=False, transform=preprocess)

	def get_features(dataset):
		all_features = []
		all_labels = []
		
		with torch.no_grad():
			for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
				features = model.encode_image(images.to(device))

				all_features.append(features)
				all_labels.append(labels)

		return torch.cat(all_features), torch.cat(all_labels)

	# Calculate the image features
	train_features, train_labels = get_features(train)
	test_features, test_labels = get_features(test)

	train_features = torch.tensor(train_features).to(device) 
	train_labels = torch.tensor(train_labels).to(device)
	test_features = torch.tensor(test_features).to(device)
	test_labels = torch.tensor(test_labels).to(device)

	train_dataset = MyDataSet(train_features, train_labels)
	test_dataset = MyDataSet(test_features, test_labels)
	train_dataloader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=False)
	test_dataloader = DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)

	
	# if LOAD_WEIGHTS:
	# 	linear_model.load_state_dict(torch.load("./linear_probing_weights/weight.pt"))
	
	#定义loss和优化方法
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.01, momentum=0.9, nesterov=True)   #进行优化梯度下降

	epochs = EPOCH
	step_size = 200
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.5, last_epoch=-1)

	for epoch in range(epochs):
		loss = 0
		val_loss = 0
		for features, labels in train_dataloader:
		#计算grads和cost
			y_pred = linear_model(features)   #x_data输入数据进入模型中
			loss = criterion(y_pred.to(torch.double), labels.to(torch.long))
			optimizer.zero_grad() #梯度清零
			loss.backward() #反向传播
			optimizer.step()  #优化迭代
		
		with torch.no_grad():
			for features, labels in test_dataloader:
				y_pred = linear_model(features)   #x_data输入数据进入模型中
				val_loss = criterion(y_pred.to(torch.double), labels.to(torch.long))
		
		print('epoch = ', epoch+1, loss.item(),val_loss.item())
		
		scheduler.step()

	torch.save(linear_model.state_dict(), MODEL_SAVE_PATH)

	predictions = linear_model(test_features)
	predictions = torch.argmax(predictions,dim=1)
	accuracy = np.mean((test_labels == predictions).cpu().numpy().astype(float)) * 100.

	print(f"Accuracy = {accuracy:.3f}")

if __name__ == "__main__":
	
	EPOCH = 3000
	do_linear_probing("CIFAR10",EPOCH,False,"./linear_probing_weights/lp_weights_cifar10_RN.pt")
	do_linear_probing("CIFAR100",EPOCH,False,"./linear_probing_weights/lp_weights_cifar100_RN.pt")
