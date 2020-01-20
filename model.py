import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

class Model(torch.nn.Module):
    def __init__(self, backbone='resnet50'):
        super(Model, self).__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = torch.nn.Linear(512,1)

        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = torch.nn.Linear(2048,1) #  we're substituting the classifier here
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.sigmoid(x)

        return x


def load_dataset(folder_path, batch_size=16):
    tr = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                    torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root=folder_path,
                                                    transform=tr)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    return train_loader

def accuracy(pred, truth):
    agreeing = (pred.transpose(0,1)[0] >= 0.5).eq(truth >= 0.5)
    acc = float(agreeing.sum())/agreeing.numel()
    return acc