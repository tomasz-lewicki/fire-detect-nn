import torch
import torchvision.models as models

class FireClassifier(torch.nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(FireClassifier, self).__init__()

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
