import torch
import torch.nn.functional as F
from torch import nn

from torchvision.models import vgg16_bn
from collections import OrderedDict
from ..builder import BACKBONES
from .base_backbone import BaseBackbone

@BACKBONES.register_module()
class VGG16(BaseBackbone):
    def __init__(self,):
        super().__init__()
    
        bb_net = list(vgg16_bn(pretrained=False).children())[0]
        bb_convs = OrderedDict({
            'conv1': bb_net[:6],
            'conv2': bb_net[6:13],
            'conv3': bb_net[13:23],
            'conv4': bb_net[23:33],
            'conv5': bb_net[33:43]
        })
        self.ics = [512, 512, 256, 128, 64]
        self.encoder = nn.Sequential(bb_convs)
        
    def forward(self, x):
        return self.encoder(x)