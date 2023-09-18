# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.optim.lr_scheduler as lrs
import cv2

from . import builder


def resize(input, target_size=(224, 224)):
    return F.interpolate(
        input, (target_size[0], target_size[1]), mode="bilinear", align_corners=True
    )


class BaseSEG(pl.LightningModule):
    def __init__(self, backbone, head, train_set, **kwargs):
        super().__init__()
        self.backbone = builder.build_Seg_model(backbone)
        head["in_channels"] = self.backbone.ics[::-1]
        self.head = builder.build_head(head)
        self.train_set = train_set

    # @abstractmethod
    def training_step(self, batch, **kwargs):
        """Defines the computation performed at training."""
        cosal_im = batch["cosal_img"]
        sal_im = batch["sal_img"]
        cosal_batch = cosal_im.shape[0]
        if isinstance(sal_im, torch.Tensor):
            img = torch.cat((cosal_im, sal_im), dim=0)
        else:
            img = cosal_im
        img = sal_im
        out = self.backbone(img)
        sals = self.head(out)
        SISMs = sals  # [cosal_batch:, ...]
        loss = self.head.get_loss(SISMs, batch["sal_gt"])
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, *args,**kwargs):
        import os

        img = batch["cosal_img"][0:1, ...]
        path = batch["path"]
        im = cv2.imread(path[0])
        out = self.backbone(img)
        sals = self.head(out).detach().cpu()
        sal = resize(sals, im.shape[:2]).squeeze().numpy()
        # 把sal和im加权到一起并输出
        #sal = sal * 255
        sal = sal.astype("uint8")
        #sal = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
        im[sal>0.5]=255
        cv2.imwrite(f"workdir/{os.path.basename(path[0])}", im)

    # @abstractmethod
    # def validation_step(self, batch, **kwargs)
    #     """Defines the computation performed at training."""

    def configure_optimizers(self):
        if "weight_decay" in self.train_set:
            weight_decay = self.train_set["weight_decay"]
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.head.parameters(),
                },
                {"params": self.backbone.parameters(), "lr": 1e-5},
            ],
            lr=self.train_set["lr"],
            weight_decay=weight_decay,
        )

        if "lr_scheduler" not in self.train_set:
            return optimizer
        else:
            if self.train_set["lr_scheduler"] == "step":
                scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.train_set["step"],
                    gamma=self.train_set["decay_rate"],
                )
            elif self.train_set["lr_scheduler"] == "cosine":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.train_set["T_max"],
                    eta_min=self.train_set["min_lr"],
                )
            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]
