import torch
import unittest

import sys

sys.path.append('..')
from models import FireClassifier

class TestModels(unittest.TestCase):

    def test_backbone_avail(self):
        BACKBONES = ['resnet18','resnet34','resnet50','resnet101', 'densenet121']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for b in BACKBONES:
            print(f"Testing {b}...")
            model = FireClassifier(backbone=b)
            model = model.to(device)
            self.assertTrue(model(torch.ones(1,3,224,224).to(device)))
            
if __name__ == '__main__':
    unittest.main()