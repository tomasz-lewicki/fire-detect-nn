import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
model = torch.load("../weights/resnet50-epoch-8-valid_acc=0.97-test_acc=-1.00.pt")

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

torch.save(model_trt.state_dict(), 'resnet50-epoch-8-valid_acc=0.97-test_acc=-1.00-tensorrt.pt')