import torch
import torchvision
import os
import time

from models import FireClassifier, BACKBONES, transform
from utils import accuracy_gpu


# # Load saved model

weight_path = "weights/resnet50-epoch-1-val_acc=0.99-test_acc=-1.00.pt"
device = torch.device("cuda:0")

model = torch.load(weight_path)
model = model.to(device)
model.eval()


# # Define datasets

dataset_paths = {
    "afd_train": "/media/tomek/BIG2/datasets/FIRE/aerial_fire_dataset/train",
    "afd_test": "/media/tomek/BIG2/datasets/FIRE/aerial_fire_dataset/test/",
    "dunnings_train": "/media/tomek/BIG2/datasets/FIRE/dunnings/fire-dataset-dunnings/images-224x224/train",
    "dunnings_test": "/media/tomek/BIG2/datasets/FIRE/dunnings/fire-dataset-dunnings/images-224x224/test",
    "combined_train": "/media/tomek/BIG2/datasets/FIRE/combined_dunnings_afd/train",
    "combined_test": "/media/tomek/BIG2/datasets/FIRE/combined_dunnings_afd/test"
}

# Define transform for data preprocessing

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.4005, 0.3702, 0.3419), std=(0.2858, 0.2749, 0.2742)),
    ]
)

def average(l):
    return sum(l)/len(l)


# External test set

test_dunnings = torchvision.datasets.ImageFolder(root=dataset_paths['combined_test'],
                                                 transform=transform)

test = torch.utils.data.DataLoader(
    test_dunnings,
    batch_size=8,
    num_workers=0,
    shuffle=True
)

test_acc = []
with torch.no_grad():
    for i, data in enumerate(test):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(inputs)
        pred = outputs.squeeze() > 0.5
        acc = torch.sum(pred == labels).double()/pred.numel()
        acc = float(acc)
        test_acc.append(acc)
        print(f'testing batch {i}/{len(test)} batch accuracy: {acc:.4f} cumulative: {average(test_acc):.4f}')


print(f"Combined: {average(test_acc)}")


# AFD test set
test_afd = torchvision.datasets.ImageFolder(root=dataset_paths['afd_test'],
                                            transform=transform)

test_afd.class_to_idx = {'positive': 1, 'negative': 0} # class mapping
test = torch.utils.data.DataLoader(
    test_afd,
    batch_size=8,
    num_workers=0,
    shuffle=True
)

test_acc = []
with torch.no_grad():
    for i, data in enumerate(test):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(inputs)
        pred = ~ (outputs.squeeze() > 0.5)
        acc = torch.sum(pred == labels).double()/pred.numel()
        acc = float(acc)
        test_acc.append(acc)
        print(f'testing batch {i}/{len(test)} batch accuracy: {acc:.4f} cumulative: {average(test_acc):.4f}')

print(f"AFD: {average(test_acc)}")