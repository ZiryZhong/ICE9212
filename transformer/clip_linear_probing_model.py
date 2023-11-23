import torch
from torch.utils.data import Dataset

# Perform linear regression
class LinearModel(torch.nn.Module):
	def __init__(self, type_cnt):
		super(LinearModel, self).__init__()  #初始父类
		self.linear = torch.nn.Linear(1024, type_cnt)  #输入维度和输出维度都为1
	
	def forward(self, x):
		y_pred = self.linear(x.to(torch.float32))
		return y_pred

class MyDataSet(Dataset):

	def __init__(self, datas, labels) -> None:
		super().__init__()
		self.datas = datas
		self.labels = labels
		self.len = len(labels)

	def __getitem__(self, index):
		return self.datas[index], self.labels[index]
	
	def __len__(self):
		return self.len