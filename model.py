import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

class Model(torch.nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Model, self).__init__()

        #  we'll have to substitute the classifier here
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = torch.nn.Linear(512,1)

        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = torch.nn.Linear(2048,1) 

        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone.fc = torch.nn.Linear(512,1)

        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.backbone.fc = torch.nn.Linear(2048,1) 

        elif backbone == 'mobilenet':
            self.backbone = models.mobilenet.mobilenet_v2(pretrained=pretrained)
            self.backbone.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=False),
                torch.nn.Linear(in_features=1280, out_features=1, bias=True)
                )
            
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            self.backbone.classifier = torch.nn.Linear(in_features=1024, out_features=1, bias=True)
        
        else: raise NotImplementedError
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.sigmoid(x)

        return x

def load_dataset(directory='~/fire_aerial2k_dataset/',
                     val_frac = 0.1,
                     batch_size = 16,
                     random_seed = 4822):
    
    tr = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                torchvision.transforms.ToTensor()])
    
    entire_dataset = torchvision.datasets.ImageFolder(root=directory,
                                                      transform=tr)

    n_all = len(entire_dataset)
    n_valid = int(np.floor(val_frac * n_all))
    indices = list(range(n_all))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idxs_list, test_idxs_list = indices[n_valid:], indices[:n_valid]

    train_loader = torch.utils.data.DataLoader(
        entire_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idxs_list),
    )
    
    test_loader = torch.utils.data.DataLoader(
        entire_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(test_idxs_list),
    )
    
    
    return train_loader, test_loader

def accuracy(pred, truth):
    agreeing = (pred.transpose(0,1)[0] >= 0.5).eq(truth >= 0.5)
    acc = float(agreeing.sum())/agreeing.numel()
    return acc