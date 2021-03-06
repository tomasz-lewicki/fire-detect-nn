import os
import json

import numpy as np
import torch
import torchvision

from datasets.afd import make_afd_loaders
from datasets.dunnings import make_dunnings_test_loader, make_dunnings_train_loader
from datasets.combo import make_combo_train_loaders

from models import FireClassifier, BACKBONES
from utils import accuracy_gpu

BATCH_SIZE = 32
EPOCHS = 10
DECREASE_LR_AFTER = 3
PRINT_EVERY = 100  # batches
EVAL_EVERY = 100

# Datasets are in data/ folder by default
file_path = os.path.realpath(__file__)
DATASETS_ROOT = os.path.dirname(file_path) + "/data"

DATASETS = {
    # AFD only
    "afd_train": DATASETS_ROOT + "/aerial_fire_dataset/train",
    "afd_test": DATASETS_ROOT + "/aerial_fire_dataset/test/",

    # Dunnings only
    "dunnings_train": DATASETS_ROOT + "/dunnings/fire-dataset-dunnings/images-224x224/train",
    "dunnings_test": DATASETS_ROOT + "/FIRE/dunnings/fire-dataset-dunnings/images-224x224/test",

    # AFD + Dunnings
    "combined_train": DATASETS_ROOT + "/combined_dunnings_afd/train",
    "combined_test": DATASETS_ROOT + "/combined_dunnings_afd/test"
}


# dunnings_train = make_dunnings_train_loader(
#     DATASETS["dunnings_train"], batch_size=BATCH_SIZE
# )

# test = make_dunnings_test_loader(DATASETS["dunnings_test"], batch_size=BATCH_SIZE)



# print(f"Loaded {len(test)} test batches with {len(test) * BATCH_SIZE} samples")

# Can be useful if we're retraining many times on the entire dataset
# completely memory extravagant but I have 256GB of RAM to use :)
# train, val = list(train), list(val)

for bbone in BACKBONES:

    BATCH_SIZE = 16 if bbone == 'VGG16' else 32
    train, val = make_combo_train_loaders(
        DATASETS["combined_train"], batch_size=BATCH_SIZE
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
    m = FireClassifier(backbone=bbone)
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
