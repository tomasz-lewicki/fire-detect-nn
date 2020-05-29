#!/usr/bin/env python
# coding: utf-8

import sys 
import numpy as np
import torch
import torchvision

sys.path.append('../firedetect')
from dataset import load_dataset


model = torch.load('../firedetect/weights/resnet50-epoch-1-valid_acc=0.9802-test_acc=0.63.pt')

dataset_paths = {'mine': '/home/013855803/fire_aerial2k_dataset/',
                 'dunnings': '/home/013855803/fire-dataset-dunnings/images-224x224/train',
                 'dunnings_test':  "/home/tomek/projects/fire-detect-nn/data/fire-dataset-dunnings/images-224x224/test",}


def accuracy_gpu(pred, truth):
    agreeing = pred.eq(truth)
    acc = agreeing.sum().double()/agreeing.numel()
    return float(acc)

tr = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                            torchvision.transforms.ToTensor()])

test_dataset = torchvision.datasets.ImageFolder(root=dataset_paths['dunnings_test'],
                                                transform=tr)

# test_dataset.class_to_idx = {'fire': 1, 'nofire': 0} # for dunnings

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=0,
    shuffle=False
)

device = torch.device("cuda:0")
test_acc = []

with torch.no_grad():
    model = model.to(device)
    model.eval()

    for i, data in enumerate(test_loader):
        print(f'testing batch {i}/{len(test_loader)}')
        inputs = data[0].to(device)
        labels = torch.tensor(data[1], dtype=torch.bool).to(device)

        scores = model(inputs)
        pred = scores.squeeze() > 0.5

        a = accuracy_gpu(pred, labels)
        test_acc.append(a)
        print(np.mean(test_acc))

