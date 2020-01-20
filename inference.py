import torch
from model import Model

model = Model(backbone='resnet18')
model = torch.load('firedetection-resnet18.pt')
device = torch.device("cuda:0")
model = model.to(device)

def avgerage(l):
    return sum(l)/len(l)

for param in model.parameters():
    param.requires_grad = False

import time
times = []

for _ in range(50):
    tens = torch.rand((1,3,224,224)).to(device)
    start = time.time()

    result = model(tens)

    timing = time.time()-start
    print(timing)
    times.append(timing)


print(f'average inference: {avgerage(times)}')