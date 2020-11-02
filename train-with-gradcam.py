import json

import numpy as np
import torch
import torchvision

from datasets.afd import make_afd_loaders
from datasets.dunnings import make_dunnings_test_loader, make_dunnings_train_loader
from datasets.combo import make_combo_train_loaders

from models import FireClassifierCAM
from utils import accuracy_gpu

BATCH_SIZE = 32
EPOCHS = 10
DECREASE_LR_AFTER = 3
PRINT_EVERY = 1  # batches
EVAL_EVERY = 1000

bbone = "final-densenet"

dataset_paths = {
    "afd_train": "/home/tomek/pro/aerial_fire_dataset/train",
    # "afd_test": "home/tomek/pro/aerial_fire_dataset/test/",
    "dunnings_train": "/media/tomek/BIG2/datasets/FIRE/dunnings/fire-dataset-dunnings/images-224x224/train",
    # "dunnings_test": "/media/tomek/BIG2/datasets/FIRE/dunnings/fire-dataset-dunnings/images-224x224/test",
    # HDD
    # "combined_train": "/media/tomek/BIG2/datasets/FIRE/combined_dunnings_afd/train",
    # "combined_test": "media/tomek/BIG2/datasets/FIRE/combined_dunnings_afd/test,
    # SSD
    "combined_train": "/home/tomek/pro/combined_dunnings_afd/train",
    # "combined_test": "/home/tomek/pro/combined_dunnings_afd/test
}


train, val = make_combo_train_loaders(
    dataset_paths["combined_train"], batch_size=BATCH_SIZE
)

print(f"Loaded {len(train)} training batches with {len(train) * BATCH_SIZE} samples")
print(f"Loaded {len(val)} val batches with {len(val) * BATCH_SIZE} samples")

print(f"Training {bbone}")

device = torch.device("cuda:0")
do_val = True
do_test = False

history = {
    "train_samples": [],
    "train_acc": [],
    "train_loss": [],
    "val_acc": [],
    "test_acc": [],
}

# bbone = "resnet50"
m = FireClassifierCAM(imagenet_init=True)
m.train()
m = m.to(device)

criterion = torch.nn.BCELoss()

for epoch in range(EPOCHS):

    optimizer = torch.optim.Adam(
        m.parameters(),
        lr=1e-4 if epoch < DECREASE_LR_AFTER else 1e-5,
        weight_decay=1e-3
    )

    running_loss = []
    running_acc = []

    # epoch training

    for i, data in enumerate(train):
        m.train()
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

        if i % PRINT_EVERY == 0:
            print(
                f"epoch: {epoch+1:02d}, \
                batch: {i:03d}, \
                loss: {np.mean(running_loss):.3f}, \
                training accuracy: {np.mean(running_acc):.3f}"
            )

            history["train_samples"].append(epoch * len(train) + i)
            history["train_acc"].append(np.mean(running_acc))
            history["train_loss"].append(np.mean(running_loss))

        # del outputs, inputs, labels

        if i % EVAL_EVERY == 0:
            m.eval()
            val_acc = []
            # epoch val

            for i, data in enumerate(val):

                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0].to(device)
                labels = data[1].to(device)

                with torch.no_grad():
                    scores = m(inputs).squeeze()
                    pred = scores > 0.5
                    acc = accuracy_gpu(pred, labels)

                val_acc.append(acc)

            va = round(np.mean(val_acc), 4)
            print(f"val accuracy {va}")
            history["val_acc"].append(va)

    #########################################
    # on epoch end:
    m.eval()
    if do_val:
        val_acc = []
        # epoch val

        for i, data in enumerate(val):

            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            labels = data[1].to(device)

            with torch.no_grad():
                scores = m(inputs).squeeze()
                pred = scores > 0.5
                acc = accuracy_gpu(pred, labels)

            val_acc.append(acc)

        va = round(np.mean(val_acc), 4)
        print(f"val accuracy {va}")
        history["val_acc"].append(va)
    else:
        va = -1

    if do_test:
        test_acc = []
        # epoch val

        with torch.no_grad():
            for i, data in enumerate(test):
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

    fname = (
        f"weights/{bbone}-epoch-{epoch+1}-val_acc={va:.4f}-test_acc={tst:.2f}.pt"
    )
    torch.save(m, fname)
    print(f"Saved {fname}")

    with open("log.json", "w") as f:
        s = json.dumps(history)
        f.write(s)

print(f"Finished Training: {bbone}")
