import torch
import torch.nn.functional as F
from torch import nn
from typing import *
from einops import rearrange, einsum
from src.model.base import BaseSegmenter, MODELS
from src.model.module.auto_feature_pyramid import AutoFeaturePyramid
from src.model.module.manual_extractor import ManualFeatureExtractor

from src.loss.utils import MaskSmoother

from typing import *


class NaiveSegmenterPL(BaseSegmenter):
    """a naive toy segmenter built for test"""

    def setup_model(self):
        return NaiveSegmenter(
            backbone=self.model_cfg.backbone,
            shot=self.cfg.exp.n_shot,
        )
        
    def call_loss_func(self, out: List, mask_q, mask_s):
        """TO-BE-OVERRIDDEN"""
        call_ls = lambda x, y: self.loss_func(x, y)
        def sum_dict(dicts, weights):
            return {key: sum([d[key] * w for w, d in zip(weights, dicts)]) for key in dicts[0].keys()}
        
        if len(out) > 1:
            l0, d0 = call_ls(out[0], mask_q)
            l1, d1 = call_ls(out[1], mask_s)
            
            total_loss = l0 + 0.3 * l1
            loss_dict = sum_dict((d0, d1), (1, 0.3))
        else:
            total_loss, loss_dict = call_ls(out[0], mask_q)
        
        return total_loss, loss_dict


class NaiveSegmenter(torch.nn.Module):
    """A naive segmenter, using sam encoder"""
    def __init__(self, backbone='sam_vit_b', shot=1, n_attn_blocks=5, *args, **kwargs):
        super().__init__()
        self.backbone = ManualFeatureExtractor(backbone)
        self.shot = shot
        self.temperature = 10.0

    def forward(
        self, 
        img_s_list: List[torch.Tensor], 
        mask_s_list: List[torch.Tensor], 
        img_q: torch.Tensor, 
        mask_q: torch.Tensor,
    ) -> List[torch.tensor] | torch.Tensor:
        H, W = img_q.shape[-2:]
        # from ipdb import set_trace as st
        # st()
        # feature maps of support images, [(b, c, h, w)] * n_shot
        feature_s_list = [self.backbone(img_s) for img_s in img_s_list]
        
        # feature map of query image, (b, c, h, w)
        feature_q = self.backbone(img_q)
        # h, w = feature_q.shape[-2:]
        proto_bg, proto_fg = self.get_prototype_pair(feature_s_list, mask_s_list)
        pred_q = self.get_logits(feature_q, proto_bg, proto_fg)
        
        # proto_bg_from_q, proto_fg_from_q = self.get_prototype_pair(feature_q, pred_q.argmax(dim=1))
        # pred_s = self.get_logits(feature_s_list, proto_bg_from_q, proto_fg_from_q)
        
        out = [F.interpolate(pred_q, (H, W), mode='bilinear')]  # , F.interpolate(pred_s, (H, W), mode='bilinear')]
        return out

    def get_logits(self, features, proto_bg, proto_fg):
        # assert False, f'{features[0].shape, proto_bg.shape}'
        if isinstance(features, (list, tuple)):            
            return self.temperature * torch.cat(
                [torch.stack([self.cos_sim(proto_bg, f), self.cos_sim(proto_fg, f)], dim=1) for f in features], dim=0)
        else:
            return self.temperature * torch.stack(
                [self.cos_sim(proto_bg, features), self.cos_sim(proto_fg, features)], dim=1)
    
    def get_prototype_pair(
        self, 
        features: List[torch.Tensor] | torch.Tensor, 
        masks: List[torch.Tensor] | torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        def _process_one_pair(feature, mask):
            mask = F.interpolate(mask[:, None].float(), feature.shape[-2:], mode='bilinear')
            p_bg = self.masked_average_pooling(feature, 1 - mask)
            p_fg = self.masked_average_pooling(feature, mask)
            return p_bg, p_fg
        
        if isinstance(features, (list, tuple)) and isinstance(masks, (list, tuple)):
            proto_bg, proto_fg = [], []
            for feature, mask in zip(features, masks):
                p_bg, p_fg = _process_one_pair(feature, mask)
                proto_bg.append(p_bg)
                proto_fg.append(p_fg)
                
            return sum(proto_bg) / len(proto_bg), sum(proto_fg) / len(proto_fg)
        
        else:
            return _process_one_pair(features, masks)
        
    def masked_average_pooling(self, x, mask):
        return ((x * mask).sum(dim=(-1, -2), keepdim=True) / (mask.sum(dim=(-1, -2), keepdim=True)+ 10e-5))
    
    def cos_sim(self, x, y):
        # x: (b c 1 1), y: (b c h w)
        return einsum(F.normalize(x, dim=1), F.normalize(y, dim=1), 'b c h w, b c h w -> b h w')
    
    

# pred_q, pred_s = [], []
# for feature_s, mask_s in zip(feature_s_list, mask_s_list):
#     q, v = rearrange(feature_s, 'b c h w -> b (h w) c'), rearrange(feature_q, 'b c h w -> b (h w) c')
    
#     k = F.interpolate(mask_s[:, None].float(), (h, w), mode='bilinear')  # (b, 1, h, w)
#     k = rearrange(k, 'b c h w -> b (h w) c').repeat(1, 1, 256)
#     p = self.masked_average_pooling(q, k)
#     for attn in self.attn:
#         v = attn(q, p, v, self.pe)
#         q = attn(v, p, q, self.pe)
        
#     pred_q.append(v)
#     pred_s.append(q)

# pred_q = self.classifier(sum(pred_q) / len(pred_q))
# pred_s = self.classifier(sum(pred_s) / len(pred_s))

# pred_q = rearrange(pred_q, 'b (h w) c -> b c h w', h=h)
# pred_s = rearrange(pred_s, 'b (h w) c -> b c h w', h=h)
# print(pred_q.max(), pred_q.min(), pred_q[0, :, :5, :5])
# return [F.interpolate(pred_q, (H, W), mode='bilinear'), F.interpolate(pred_s, (H, W), mode='bilinear')]