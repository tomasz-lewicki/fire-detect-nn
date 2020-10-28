from model import load_dataset, accuracy
from model import Model

import torch
import torchvision
import os
import time

def average(l):
    return sum(l)/len(l)

weight_path = 'weights/resnet50-epoch-29-val_acc=0.99-test_acc=-1.00.pt'

dataset_paths = {'mine': '/home/013855803/fire_aerial2k_dataset/',
                 'dunnings': '/home/013855803/fire-dataset-dunnings/images-224x224/train',
                 'dunnings_test': '/home/013855803/fire-dataset-dunnings/images-224x224/test'}

tr = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                            torchvision.transforms.ToTensor()])

test_dataset = torchvision.datasets.ImageFolder(root=dataset_paths['dunnings'],
                                                transform=tr)

#test_dataset.class_to_idx = {'fire': 1, 'nofire': 0} # for dunnings

test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True
)

device = torch.device("cuda:0")
test_acc = []

print('loading model...')
model = torch.load(weight_path)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

for i, data in enumerate(test):
    print(f'testing batch {i}/{len(test)}')
    inputs = data[0].to(device)
    labels = (~ torch.tensor(data[1], dtype=torch.bool)).to(device)

    outputs = model(inputs)
    a = accuracy(outputs, labels)
    test_acc.append(a)
    print(average(test_acc))

