# Copyright (c) OpenMMLab. All rights reserved.
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
    def __init__(self, backbone, aux_head, neck,head,train_set, **kwargs):
        super().__init__()
        self.backbone = builder.build_Seg_model(backbone)
        aux_head["in_channels"] = self.backbone.ics[::-1]
        neck["in_channels"] = self.backbone.ics
        head["in_channels"] = self.backbone.ics
        
        
        self.aux_head = builder.build_head(aux_head)
        
        self.neck = builder.build_neck(neck)
        self.train_set = train_set
        self.head = builder.build_head(head)

    # @abstractmethod
    def training_step(self, batch, **kwargs):
        """Defines the computation performed at training."""
        cosal_im = batch["cosal_img"]
        sal_im = batch["sal_img"]
        group_num = batch["group_num"]
        assert sum(group_num)==cosal_im.shape[0],group_num
        cosal_batch = cosal_im.shape[0]
        if isinstance(sal_im, torch.Tensor):
            img = torch.cat((cosal_im, sal_im), dim=0)
        else:
            img = cosal_im
            
        feat = self.backbone(img)
        ALL_SISMs = self.aux_head(feat)
        
        SISMs = ALL_SISMs[:cosal_batch, ...]
        SISMs_sup = ALL_SISMs[cosal_batch:, ...]
        maps = batch["cosal_gt"]
        maps = maps.unsqueeze(1).float()
        aux_loss = self.aux_head.get_loss(SISMs_sup, batch["sal_gt"])
            
        cmprs_feat = self.neck(feat,cosal_batch)
        
        pred_list,map = self.head(feat,cmprs_feat,SISMs,maps,cosal_batch,group_num)
        #print(len(pred),pred[-1].shape)
        cosal_loss = self.head.get_loss(pred_list,batch["cosal_gt"])
        loss = 0.9*cosal_loss+0.1*aux_loss
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cosal_loss", cosal_loss, on_step=True,prog_bar=True, logger=True)
        self.log("aux_loss", aux_loss, on_step=True,prog_bar=True, logger=True)
        return loss
    
    
    def validation_step(self,batch,*args,**kwargs):

        img = batch["cosal_img"]
        group_num = batch["group_num"]
        assert sum(group_num)==img.shape[0]
        cosal_batch = img.shape[0]
        feat = self.backbone(img)
        SISMs = self.aux_head(feat)
              
        cmprs_feat = self.neck(feat,cosal_batch)
        
        pred = SISMs
        for _ in range(3):
            
            pred_list,map = self.head(feat,cmprs_feat,SISMs,pred,cosal_batch,group_num)
            pred = map
        cosal_loss = self.head.get_loss(pred_list[-1:],batch["cosal_gt"])
        val_iou = 1-cosal_loss
        self.log("val_iou", val_iou, on_epoch=True, logger=True)
        return 0

    def predict_step(self, batch, *args,**kwargs):
        import os

        img = batch["cosal_img"]
        group_num = batch["group_num"]
        paths = batch["path"]
        #im = cv2.imread(path)
        
        cosal_batch = img.shape[0]
        feat = self.backbone(img)
        SISMs = self.aux_head(feat)
        
              
        cmprs_feat = self.neck(feat,cosal_batch)
        pred = self.head(feat,cmprs_feat,SISMs,cosal_batch,group_num)
              
        sals = pred.detach().cpu()
        for sal,path in zip(sals,paths):
            im = cv2.imread(path)
            
            sal = resize(sal, im.shape[:2]).squeeze().numpy()
            sal = sal * 255
            sal = sal.astype("uint8")
            #sal = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
            im[sal>127,0]=255
            os.makedirs("workdir", exist_ok=True)
            cv2.imwrite(f"workdir/{os.path.basename(path)}", im)


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
            
            elif self.train_set["lr_scheduler"] == "multistep":
                scheduler = lrs.MultiStepLR(
                    optimizer,
                    milestones=self.train_set["milestones"],
                    gamma=self.train_set["gamma"],
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
