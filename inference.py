import torch
import os
import time

from model import Model

weight_paths = os.listdir('weights')

def avgerage(l):
    return sum(l)/len(l)

device = torch.device("cuda:0")
fps_results = {}

for p in weight_paths:
    backbone_name = p.split('-')[0]
    print(f'trying {backbone_name}')
    model = Model(backbone=backbone_name)
    model = torch.load('weights/'+p)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    times = []
    tens = torch.rand((1,3,224,224)).to(device)
    model(tens)

    for _ in range(50):
        tens = torch.rand((1,3,224,224)).to(device)
        start = time.time()

        result = model(tens)

        timing = time.time()-start
        print(timing)
        times.append(timing)

    fps = round(1/avgerage(times), 2)

    fps_results[backbone_name] = fps
    print(f'average fps with {backbone_name} : {fps}')