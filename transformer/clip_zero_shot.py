import os
import clip
import torch
from torchvision.datasets import CIFAR100, CIFAR10
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from clip_linear_probing_model import MyDataSet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CUDA_LAUNCH_BLOCKING=1
def do_zero_shot(DATASET,OUTPUT_PATH):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # model, preprocess = clip.load('ViT-B/32', device)
    model, preprocess = clip.load('RN50', device)


    f = open(OUTPUT_PATH, "w")
    
    # Download the dataset
    if DATASET == "CIFAR100":
        test_set = CIFAR100(root="./data", download=False, train=False, transform=preprocess)
    elif DATASET == "CIFAR10":
        test_set = CIFAR10(root="../image-classification/data", download=False, train=False, transform=preprocess)
    
    dataloader = DataLoader(dataset=test_set,batch_size=32,shuffle=False)
    true_labels = []
    predicted_labels = []
    pbar = tqdm(total = len(dataloader))

    # Prepare the inputs
    for image_input, class_id in dataloader:
        # image, class_id = test_set[i]
        # image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_set.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input.to(device))
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[:].topk(1)
        
        true_labels.extend(class_id.cpu().numpy())
        predicted_labels.extend(indices.cpu().numpy())
        pbar.update(1)
        
    epoch_test_accuracy = accuracy_score(true_labels,predicted_labels)
    epoch_test_precision, epoch_test_recall, epoch_test_f1 = precision_recall_fscore_support(true_labels,predicted_labels,average='macro', zero_division='warn')[:-1]
    
    print("accuracy:{},precision:{},recall:{},f1:{}\n".format(epoch_test_accuracy, epoch_test_precision,epoch_test_recall,epoch_test_f1))
    f.write("accuracy:{},precision:{},recall:{},f1:{}\n".format(epoch_test_accuracy, epoch_test_precision,epoch_test_recall,epoch_test_f1))
    f.close()
    # # Print the result
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")


def do_zero_shot_back(DATASET,OUTPUT_PATH):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    f = open(OUTPUT_PATH, "w")
    
    # Download the dataset
    if DATASET == "CIFAR100":
        test_set = CIFAR100(root="./data", download=False, train=False)
    elif DATASET == "CIFAR10":
        test_set = CIFAR10(root="../image-classification/data", download=False, train=False, transform=preprocess)
    
    dataloader = DataLoader(dataset=test_set,batch_size=32,shuffle=False)
    true_labels = []
    predicted_labels = []
    pbar = tqdm(total = len(test_set))

    # Prepare the inputs
    for i in range(0,len(test_set)):
        image, class_id = test_set[i]
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_set.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input.to(device))
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        
        # true_labels.extend(class_id.cpu().numpy())
        # predicted_labels.extend(indices.cpu().numpy())
        true_labels.append(class_id)
        predicted_labels.append(indices.cpu())
        pbar.update(1)
        
    epoch_test_accuracy = accuracy_score(true_labels,predicted_labels)
    epoch_test_precision, epoch_test_recall, epoch_test_f1 = precision_recall_fscore_support(true_labels,predicted_labels,average='macro', zero_division='warn')[:-1]
    
    print("accuracy:{},precision:{},recall:{},f1:{}\n".format(epoch_test_accuracy, epoch_test_precision,epoch_test_recall,epoch_test_f1))
    f.write("accuracy:{},precision:{},recall:{},f1:{}\n".format(epoch_test_accuracy, epoch_test_precision,epoch_test_recall,epoch_test_f1))
    f.close()
    # # Print the result
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

if __name__ == "__main__":

    do_zero_shot("CIFAR100","./logs/cifar100-zero-shot-RN.log")
    do_zero_shot("CIFAR10","./logs/cifar10-zero-shot-RN.log")