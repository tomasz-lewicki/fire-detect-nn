# AFD: Aerial Fire Dataset

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def make_afd_loaders(directory='~/fire_aerial2k_dataset/',
                 val_frac = 0.1,
                 batch_size = 16,
                 random_seed = None,
                 shuffle=True):
    
    tr = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                torchvision.transforms.ToTensor()])
    
    entire_dataset = torchvision.datasets.ImageFolder(root=directory,
                                                      transform=tr)

    n_all = len(entire_dataset)
    n_valid = int(np.floor(val_frac * n_all))
    indices = list(range(n_all))

    if random_seed:
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
