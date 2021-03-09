# Aerial Fire Dataset +

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Dataset parameters
IMG_SHAPE = (224, 224)
RGB_MEAN = (0.4005, 0.3702, 0.3419)
RGB_STD = (0.2858, 0.2749, 0.2742)

# Example transform
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(IMG_SHAPE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ]
)

def make_combo_train_loaders(
    directory,
    val_frac=0.1,
    batch_size=16,
    random_seed=4822,
    shuffle=True,
):

    entire_dataset = torchvision.datasets.ImageFolder(
        root=directory,
        transform=transform,
        # What's the right way to use this???
        # class_to_idx = {'fire': 1, 'nofire': 0} # class mapping
    )

    n_all = len(entire_dataset)
    n_valid = int(np.floor(val_frac * n_all))
    indices = list(range(n_all))

    np.random.seed(random_seed)

    if shuffle:
        np.random.shuffle(indices)

    train_idxs_list, test_idxs_list = indices[n_valid:], indices[:n_valid]

    train_loader = torch.utils.data.DataLoader(
        entire_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idxs_list),
    )

    test_loader = torch.utils.data.DataLoader(
        entire_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(test_idxs_list),
    )

    return train_loader, test_loader
