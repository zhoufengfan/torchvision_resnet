import torch.nn as nn
import torchvision

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50=torchvision.models.resnet50()

    def forward(self):

