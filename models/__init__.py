import torch
import torchvision
import torchvision.models as models

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.4005, 0.3702, 0.3419), std=(0.2858, 0.2749, 0.2742)
        ),
    ]
)

BACKBONES = [
    "resnet50",
    "densenet121",
    "resnet18",
    "resnet34",
    "resnet101",
    "VGG16",
    "mobilenet",
]


class FireClassifier(torch.nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True):
        super(FireClassifier, self).__init__()

        #  we'll have to substitute the classifier here
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = torch.nn.Linear(512, 1)

        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = torch.nn.Linear(2048, 1)

        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone.fc = torch.nn.Linear(512, 1)

        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            self.backbone.fc = torch.nn.Linear(2048, 1)

        elif backbone == "mobilenet":
            self.backbone = models.mobilenet.mobilenet_v2(pretrained=pretrained)
            self.backbone.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=1280, out_features=1, bias=True),
            )

        elif backbone == "VGG16":
            self.backbone = models.vgg16(pretrained=pretrained)
            self.backbone.classifier = torch.nn.Sequential(
                torch.nn.Linear(25088, 4096),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(4096, 1),
            )

        elif backbone == "InceptionV3":
            pass

        elif backbone == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            self.backbone.classifier = torch.nn.Linear(
                in_features=1024, out_features=1, bias=True
            )
            

        else:
            raise NotImplementedError

        self.sigmoid = torch.nn.Sigmoid() # instead of softmax (since this is binary classification)

    def forward(self, x):
        x = self.backbone(x)
        x = self.sigmoid(x)

        return x

class FireClassifierCAM(torch.nn.Module):
    def __init__(self, imagenet_init=True):
        super(FireClassifierCAM, self).__init__()

        self.backbone = models.densenet121(pretrained=imagenet_init).features
        self.g_avg_pool = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = torch.nn.Linear(in_features=1024, out_features=1, bias=True)
        self.sigmoid = torch.nn.Sigmoid() 
        self._gradients = None
        
    def _activations_hook(self, grad):
        self._gradients = grad

    def forward(self, x):
        x = self.backbone(x) # conv. activations
        
        # keep gradients after last layer of the backbone
        if self.train and x.requires_grad:
            _ = x.register_hook(self._activations_hook)
        
        x = self.g_avg_pool(x)
        x = x.squeeze()
        x = self.classifier(x)
        
        x = self.sigmoid(x) #score (0-1)
        self._score = x # safe score tensor for backprop during gradCAM

        return x
    
    @property
    def gradients(self):
        # has to be called after output.backward()
        return self._gradients
    
    def activations(self, inputbatch):
        return self.backbone(inputbatch)
    
    def gradCAM(self, img_tensor):
        inputb = img_tensor.unsqueeze(dim=0)
        score = self(inputb)
        self._score.backward()
        
        pooled_gradients = torch.mean(self._gradients, dim=[0, 2, 3])
        activations = self.activations(inputb).detach()

        for i in range(1024):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = heatmap.cpu().numpy()
        
        return heatmap