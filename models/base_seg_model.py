# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import lightning.pytorch as pl

from . import build_Seg_model
from . import builder

class BaseSEG(pl.LightningModule):
    def __init__(self, backbone,loss):
        super().__init__()
        self.backbone = builder.build_Seg_model(backbone)


    #@abstractmethod
    def training_step(self, batch, **kwargs):
        """Defines the computation performed at training."""
        cosal_im = batch['cosal_img']
        sal_im = batch['sal_img']
        if isinstance(sal_im,torch.Tensor):
            img = torch.cat((cosal_im,sal_im),dim=0)
        else:
            img = cosal_im
        out = self.backbone(img)
        return out


    
    
    # @abstractmethod
    # def validation_step(self, batch, **kwargs)
    #     """Defines the computation performed at training."""
    
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)