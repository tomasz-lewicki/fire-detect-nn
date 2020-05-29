#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import torchvision

from model import Model
from dataset import load_dataset

def accuracy_gpu(pred, truth):
    agreeing = pred.eq(truth)
    acc = agreeing.sum().float()/agreeing.numel()
    return float(acc)

BATCH_SIZE = 32
EPOCHS = 10

BACKBONES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "densenet121",
    "mobilenet",
]

dataset_paths = {
    "mine": "/home/013855803/fire_aerial2k_dataset/",
    "dunnings": "/home/tomek/projects/fire-detect-nn/data/fire-dataset-dunnings/images-224x224/train",
    "dunnings_test": "/home/tomek/projects/fire-detect-nn/data/fire-dataset-dunnings/images-224x224/test",
}

train_loader, valid_loader = load_dataset(
    dataset_paths["dunnings"], batch_size=BATCH_SIZE
)

tr = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
)

# test_dataset = torchvision.datasets.ImageFolder(
#     root=dataset_paths["dunnings_test"], transform=tr
# )


# test_loader = torch.utils.data.DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
#     num_workers=4,
#     shuffle=False)


test_dataset = torchvision.datasets.ImageFolder(root=dataset_paths['dunnings_test'],
                                                transform=tr)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True
)

print(f"Loaded {len(train_loader)} training batches with {len(train_loader) * BATCH_SIZE} samples")
print(f"Loaded {len(valid_loader)} validation batches with {len(valid_loader) * BATCH_SIZE} samples")
print(f"Loaded {len(test_loader)} training batches with {len(test_loader) * BATCH_SIZE} samples")

# Can be useful if we're retraining many times on the entire dataset
# completely memory extravagant but I have 256GB of RAM to use :)
# train, valid = list(train), list(valid)

device = torch.device("cuda:0")
is_validating = True
is_testing = False

history = {
    "train_samples": [],
    "train_acc": [],
    "valid_acc": [],
    "test_acc": [],
    "loss": [],
}
bbone = 'resnet50'
m = Model(backbone=bbone)
m = m.to(device)

criterion = torch.nn.BCELoss()

for epoch in range(EPOCHS):  # epochs

    optimizer = torch.optim.Adam(
        m.parameters(), lr=1e-4 if epoch < 5 else 1e-5, weight_decay=1e-3
    )

    running_loss = []
    running_acc = []

    # epoch training
    m.train()
    for i, data in enumerate(train_loader):

        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].to(device)
        labels = data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        scores = m(inputs)
        loss = criterion(scores[:, 0], labels.type_as(scores[:, 0]))
        loss.backward()
        optimizer.step()

        pred = (scores >= 0.5).squeeze()
        acc = accuracy_gpu(pred, labels)
        # print statistics

        running_loss.append(loss.item())
        running_acc.append(acc)

        if i % 20 == 0:
            print(
                f"epoch: {epoch+1:02d}, \
                batch: {i:03d}, \
                loss: {np.mean(running_loss):.3f}, \
                training accuracy: {np.mean(running_acc):.3f}"
                )

            history["loss"].append(np.mean(running_loss))
            history["train_samples"].append(epoch * len(train_loader) + i)
            history["train_acc"].append(np.mean(running_acc))

        # del outputs, inputs, labels

    #########################################
    # on epoch end:
    m.eval()
    if is_validating:
        valid_acc = []
        # epoch validation
        
        for i, data in enumerate(valid_loader):

            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            labels = data[1].to(device)

            with torch.no_grad():
                scores = m(inputs).squeeze()
                pred = scores > 0.5
                acc = accuracy_gpu(pred, labels)
            
            valid_acc.append(acc)

        va = round(np.mean(valid_acc), 4)
        print(f"validation accuracy {va}")
        history["valid_acc"].append(va)
    else:
        va = -1

    if is_testing:
        test_acc = []
        # epoch validation

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # data has list entries: [inputs, labels]
                inputs = data[0].to(device)
                labels = data[1].to(device)
                
                scores = m(inputs).squeeze()
                pred = scores > 0.5
                
                acc = accuracy_gpu(pred, labels)
                test_acc.append(acc)
    
        tst = np.mean(test_acc)
        print(f"test_accuracy {tst}")
        history["test_acc"].append(tst)
    else:
        tst = -1

    fname = f"weights/{bbone}-epoch-{epoch}-valid_acc={va:.2f}-test_acc={tst:.2f}.pt"
    torch.save(m, fname)
    print(f"Saved {fname}")

print(f"Finished Training: {bbone}")



import matplotlib.pyplot as plt

# for history in histories:
plt.figure()
plt.plot(history["train_samples"], history["loss"])

plt.plot(history["train_samples"], history["train_acc"])


plt.scatter(
    np.array(history["train_samples"]) / len(train_loader), history["train_acc"]
)
plt.scatter(list(range(10)), history["valid_acc"])
plt.scatter(list(range(10)), history["test_acc"])
