from torch import nn
import torch.nn.functional as F

from ..builder import HEADS
import torch



def resize(input, target_size=(224, 224)):
    return F.interpolate(
        input, (target_size[0], target_size[1]), mode="bilinear", align_corners=True
    )
    
def IoU_loss(preds_list, gt):
    preds = torch.cat(preds_list, dim=1)
    #preds = preds.squeeze(1)
    N,C,H,W = preds.shape
    min_tensor = torch.where(preds < gt, preds, gt)    # shape=[N, C, H, W]
    max_tensor = torch.where(preds > gt, preds, gt)    # shape=[N, C, H, W]
    min_sum = min_tensor.view(N,C, H * W).sum(dim=2)  # shape=[N, C]
    max_sum = max_tensor.view(N,C, H * W).sum(dim=2)  # shape=[N, C]
    loss = 1 - (min_sum / max_sum).mean()


    return loss 


class Res(nn.Module):
    def __init__(self, in_channel):
        super(Res, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channel, in_channel, 3, 1, 1))

    def forward(self, feats):
        feats = feats + self.conv(feats)
        feats = F.relu(feats, inplace=True)
        return feats

"""
Cosal_Module:
    Given features extracted from the VGG16 backbone,
    exploit SISMs to build intra cues and inter cues.
"""
class Cosal_Module(nn.Module):
    def __init__(self, H, W):
        super(Cosal_Module, self).__init__()
        self.cosal_feat = Cosal_Sub_Module(H, W)
        self.conv = nn.Sequential(nn.Conv2d(256, 128, 1), Res(128))

    def forward(self, feats, SISMs,group_num):
        # Get foreground co-saliency features.
        split_feats = torch.split(feats,group_num,dim=0)
        split_SISMs = torch.split(SISMs,group_num,dim=0)
        cosal_enhanced_feats=[]
        for feat_batch,SISM_batch in zip(split_feats,split_SISMs):
            fore_cosal_feats = self.cosal_feat(feat_batch,SISM_batch)

            # Get background co-saliency features.
            back_cosal_feats = self.cosal_feat(feat_batch,1.0-SISM_batch)

            # Fuse foreground and background co-saliency features
            # to generate co-saliency enhanced features.
            cosal_enhanced_feats_batch = self.conv(torch.cat([fore_cosal_feats, back_cosal_feats], dim=1))
            cosal_enhanced_feats.append(cosal_enhanced_feats_batch)
        return torch.cat(cosal_enhanced_feats,dim=0)


"""
Cosal_Sub_Module:
  * The core module of CoRP!
"""
class Cosal_Sub_Module(nn.Module):
    def __init__(self, H, W):
        super(Cosal_Sub_Module, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(32, 128, 1), Res(128))

    def forward(self, feats, SISMs):
        N, C, H, W = feats.shape
        HW = H * W

        # Resize SISMs to the same size as the input feats.
        SISMs = resize(SISMs, [H, W])  # shape=[N, 1, H, W], SISMs are the saliency maps generated by saliency head.

        # NFs: L2-normalized features.
        NFs = F.normalize(feats, dim=1)  # shape=[N, C, H, W]

        # Co_attention_maps are utilized to filter more background noise.
        def get_co_maps(co_proxy, NFs):
            correlation_maps = F.conv2d(NFs, weight=co_proxy)  # shape=[N, N, H, W]

            # Normalize correlation maps.
            correlation_maps = F.normalize(correlation_maps.reshape(N, N, HW), dim=2)  # shape=[N, N, HW]
            co_attention_maps = torch.sum(correlation_maps , dim=1)  # shape=[N, HW]

            # Max-min normalize co-attention maps.
            min_value = torch.min(co_attention_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(co_attention_maps, dim=1, keepdim=True)[0]
            co_attention_maps = (co_attention_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[N, HW]
            co_attention_maps = co_attention_maps.view(N, 1, H, W)  # shape=[N, 1, H, W]
            return co_attention_maps

        # Use co-representation to obtain co-saliency features.
        def get_CoFs(NFs, co_rep):
            SCFs = F.conv2d(NFs, weight=co_rep)
            return SCFs

        # Find the co-representation proxy.
        co_proxy = F.normalize((NFs * SISMs).mean(dim=3).mean(dim=2), dim=1).view(N, C, 1, 1)  # shape=[N, C, 1, 1]

        # Reshape the co-representation proxy to compute correlations between all pixel embeddings and the proxy.
        r_co_proxy = F.normalize((NFs * SISMs).mean(dim=3).mean(dim=2).mean(dim=0), dim=0)
        r_co_proxy = r_co_proxy.view(1, C)
        all_pixels = NFs.reshape(N, C, HW).permute(0, 2, 1).reshape(N*HW, C)
        correlation_index = torch.matmul(all_pixels, r_co_proxy.permute(1, 0))

        # Employ top-K pixel embeddings with high correlation as co-representation.
        ranged_index = torch.argsort(correlation_index, dim=0, descending=True).repeat(1, C)
        co_representation = torch.gather(all_pixels, dim=0, index=ranged_index)[:32, :].view(32, C, 1, 1)

        co_attention_maps = get_co_maps(co_proxy, NFs)  # shape=[N, 1, H, W]
        CoFs = get_CoFs(NFs, co_representation)  # shape=[N, 32, H, W]
        co_saliency_feat = self.conv(CoFs * co_attention_maps)  # shape=[N, 128, H, W]

        return co_saliency_feat



class Prediction(nn.Module):
    def __init__(self, in_channel):
        super(Prediction, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_channel, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        pred = self.pred(feats)
        return pred
    
    
    

class Decoder_Block(nn.Module):
    def __init__(self, in_channel):
        super(Decoder_Block, self).__init__()
        self.cmprs = nn.Conv2d(in_channel, 32, 1)
        self.merge_conv = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                                        nn.Conv2d(96, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pred = Prediction(32)

    def forward(self, low_level_feats, cosal_map, SISMs, old_feats):
        _, _, H, W = low_level_feats.shape

        cosal_map = resize(cosal_map, [H, W])
        SISMs = resize(SISMs, [H, W])
        old_feats = resize(old_feats, [H, W])

        # Predict co-saliency maps with the size of H*W.
        cmprs = self.cmprs(low_level_feats)
        new_feats = self.merge_conv(torch.cat([cmprs * cosal_map,
                                               cmprs * SISMs,
                                               old_feats], dim=1))
        new_cosal_map = self.pred(new_feats)
        return new_feats, new_cosal_map


@HEADS.register_module()
class cosal_Decoder(nn.Module):
    def __init__(self, in_channels):
        super(cosal_Decoder, self).__init__()
        
        self.Co6 = Cosal_Module(7, 7)
        self.Co5 = Cosal_Module(14, 14)
        self.Co4 = Cosal_Module(28, 28)
        self.Co3 = Cosal_Module(56, 56)
        
        
        self.merge_co_56 = Res(128)
        self.merge_co_45 = Res(128)
        self.merge_co_34 = nn.Sequential(Res(128), nn.Conv2d(128, 32, 1))
        
        
        self.get_pred_4 = Prediction(32)
        self.refine_2 = Decoder_Block(in_channels[3])
        self.refine_1 = Decoder_Block(in_channels[4])

    def forward(self, feat,cmprs_feat,SISMs,bs_group,group_num):
        conv3_cmprs,conv4_cmprs,conv5_cmprs,conv6_cmprs = cmprs_feat
        conv1_2,conv2_2 = feat[:2]
        cosal_feat_6 = self.Co6(conv6_cmprs, maps,group_num)  # shape=[N, 128, 7, 7]
        cosal_feat_5 = self.Co5(conv5_cmprs, maps,group_num)  # shape=[N, 128, 14, 14]
        cosal_feat_4 = self.Co4(conv4_cmprs, maps,group_num)  # shape=[N, 128, 28, 28]
        cosal_feat_3 = self.Co3(conv3_cmprs, maps,group_num)  # shape=[N, 128, 28, 28]
        # Merge co-saliancy features and predict co-saliency maps with size of 28*28 (i.e., "cosal_map_4").
        feat_56 = self.merge_co_56(cosal_feat_5 + resize(cosal_feat_6, [14, 14]))  # shape=[N, 128, 14, 14]
        feat_45 = self.merge_co_45(cosal_feat_4 + resize(feat_56, [28, 28]))  # shape=[N, 128, 28, 28]
        feat_34 = self.merge_co_34(cosal_feat_3 + resize(feat_45, [56, 56]))  # shape=[N, 128, 56, 56]
        cosal_map_4 = self.get_pred_4(feat_34)  # shape=[N, 1, 56, 56]
        # Obtain co-saliency maps with size of 224*224 (i.e., "cosal_map_1") by progressively upsampling.
        feat_23, cosal_map_2 = self.refine_2(conv2_2[:bs_group, ...], cosal_map_4, SISMs, feat_34)
        _, cosal_map_1 = self.refine_1(conv1_2[:bs_group, ...], cosal_map_4, SISMs, feat_23)
        maps = cosal_map_1


        return [resize(cosal_map_4),  resize(cosal_map_2), resize(cosal_map_1)]
        
        

    def get_loss(self,sal,gt):
        return IoU_loss(sal,gt)


