from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
import clip
import torch
from tqdm import tqdm
from clip_linear_probing_model import LinearModel
import torch.nn as nn

def do_linear_probing_test(DATASET,WEIGHTS,LOG_PATH):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)
    
    f = open(LOG_PATH,"w")

    # Load the dataset
    if DATASET == "CIFAR100":
        linear_model = LinearModel(100).to(device)
        linear_model.load_state_dict(torch.load(WEIGHTS))
        test = CIFAR100("./data", download=False, train=False, transform=preprocess)
    elif DATASET == "CIFAR10":
        linear_model = LinearModel(10).to(device)
        linear_model.load_state_dict(torch.load(WEIGHTS))
        test = CIFAR10("../image-classification/data", download=False, train=False, transform=preprocess)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        true_labels = []
        predicted_labels = []
        for images, labels in tqdm(DataLoader(test, batch_size=100)):
            features = model.encode_image(images.to(device))
            pred = linear_model(features)
            loss = criterion(pred.to(torch.double), labels.to(device).to(torch.long))
            pred = torch.argmax(pred,dim=1)
            
            true_labels.extend(labels.cpu().numpy())  # 将真实标签添加到列表
            predicted_labels.extend(pred.cpu().numpy())  # 将模型的预测添加到列表

    epoch_test_accuracy = accuracy_score(true_labels,predicted_labels)
    epoch_test_precision, epoch_test_recall, epoch_test_f1 = precision_recall_fscore_support(true_labels,predicted_labels,average='macro', zero_division='warn')[:-1]

    print("accuracy:{},precision:{},recall:{},f1:{}\n".format(epoch_test_accuracy, epoch_test_precision,epoch_test_recall,epoch_test_f1))
    f.write("accuracy:{},precision:{},recall:{},f1:{}\n".format(epoch_test_accuracy, epoch_test_precision,epoch_test_recall,epoch_test_f1))
    f.close()

if __name__ == "__main__":
    
    WEIGHT_PATH = "./linear_probing_weights/lp_weights_cifar100_RN.pt"
    do_linear_probing_test("CIFAR100",WEIGHT_PATH,"./logs/cifar100-linear-probing-RN.logs")
    WEIGHT_PATH = "./linear_probing_weights/lp_weights_cifar10_RN.pt"
    do_linear_probing_test("CIFAR10",WEIGHT_PATH,"./logs/cifar10-linear-probing-RN.logs")