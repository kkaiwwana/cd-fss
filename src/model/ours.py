import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from einops import einsum, rearrange
from typing import *

from src.model.base import BaseSegmenter, MODELS
from src.model.module.auto_feature_pyramid import AutoFeaturePyramid
from src.model.module.manual_extractor import ManualFeatureExtractor


class EncoderOnlySegmenterPL(BaseSegmenter):
    """a naive toy segmenter built for test"""

    def setup_model(self):
        return EncoderOnlySegmenter(
            backbone=instantiate(self.model_cfg.backbone),
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
            
            total_loss = l0 + l1
            loss_dict = sum_dict((d0, d1), (1, 0.3))
        else:
            total_loss, loss_dict = call_ls(out[0], mask_q)
        
        return total_loss, loss_dict


class EncoderOnlySegmenter(torch.nn.Module):
    def __init__(self, backbone, shot=1,**kwargs):
        super().__init__()
        
        self.backbone = backbone
        self.refine = True
        self.shot = shot
        self.temp = 10.0

    def fuse_feature_pyramid(self, feature_pyramid: dict, target_size: int, target_c: int):
        def _interpolate(x):
            return F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=True)
        
        return torch.cat([_interpolate(feature) for feature in feature_pyramid.values()], dim=1)

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):
        h, w = img_q.shape[-2:]

        feature_s_list = []
        for img_s in img_s_list:
            feature_s_list.append(self.backbone(img_s))
        feature_s_ls = torch.cat(feature_s_list, dim=0)
        feature_q = self.backbone(img_q)
        
        if self.training:
            out_q, out_s = self.meta_forward(feature_s_list, mask_s_list, feature_q, mask_q)
            return [out_q, out_s]
        else:
            out_q = self.meta_forward(feature_s_list, mask_s_list, feature_q, mask_q)
            return [out_q]
        
    def meta_forward(self, feature_s_list, mask_s_list, feature_q, mask_q):
        h, w  = mask_q.shape[-2:]  # raw image size 
        
        upsample = lambda x: F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        
        feature_s_ls = torch.cat(feature_s_list, dim=0)
        proto_bg_s, proto_fg_s = self.get_prototype_pair(feature_s_list, mask_s_list)
        proto_bg_q, proto_fg_q = self.get_prototype_pair(feature_q, mask_q)

        out_query, out_support =  [], []

        if self.training:
            q_out, q_out_ssp, s_out, new_FP, new_BP = self.iter_BFP(proto_fg_s, proto_bg_s, feature_s_ls, feature_q, self.refine)
            
            loss = 0.3 * (2 - self.cos_sim(new_FP, proto_fg_s).mean() - self.cos_sim(new_BP, proto_bg_s).mean())
            
            loss.backward(retain_graph=True)
            
            self_out = self.get_logits(feature_q, proto_bg_q, proto_fg_q)     
            supp_out = self.get_logits(feature_s_list, proto_bg_s, proto_fg_s)
            
            out_query.append(upsample(q_out) * 0.3)
            out_query.append(upsample(q_out_ssp) * 0.5)
            out_query.append(upsample(self_out) * 0.2)
            
            out_support.append(upsample(s_out) * 0.6)
            out_support.append(upsample(supp_out) * 0.4)
            
            return sum(out_query), sum(out_support)
        
        else:
            q_out, q_out_ssp = self.iter_BFP(proto_fg_s, proto_bg_s, feature_s_ls, feature_q, self.refine)
            out_query.append(upsample(q_out) * 0.4)
            out_query.append(upsample(q_out_ssp) * 0.6)

            return sum(out_query)

    def get_logits(self, features, proto_bg, proto_fg):
        if isinstance(features, (list, tuple)):            
            return torch.cat(
                [torch.stack([self.cos_sim(proto_bg, f), self.cos_sim(proto_fg, f)], dim=1) for f in features], dim=0) * self.temp
        else:
            return torch.stack([self.cos_sim(proto_bg, features), self.cos_sim(proto_fg, features)], dim=1) * self.temp
    
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
        return ((x * mask).sum(dim=(-1, -2), keepdim=True) / (mask.sum(dim=(-1, -2), keepdim=True) + 10e-5))
    
    def cos_sim(self, x, y):
        # x: (b c 1 1), y: (b c h w)
        return einsum(F.normalize(x, dim=1), F.normalize(y, dim=1), 'b c h w, b c h w -> b h w')
    
    def iter_BFP(self, FP, BP, feature_s_ls, feature_q, refine=True):
        ###### input FP and BP are support prototype
        ###### SSP on query side
        ### find the most similar part in query feature
        q_out = self.get_logits(feature_q, BP, FP)
        ### SSP in query feature
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, q_out)
        ### update prototype for query prediction
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        ### use updated prototype to search target in query feature
        q_out_ssp = self.get_logits(feature_q, BP_1, FP_1)

        ###### SSP on support side
        if self.training:
            ### duplicate query prototype for support SSP if shot > 1
            if self.shot > 1:
                FP_nshot = FP.repeat_interleave(self.shot, dim=0)
                FP_1 = FP_1.repeat_interleave(self.shot, dim=0)
                BP_1 = BP_1.repeat_interleave(self.shot, dim=0)
            ### find the most similar part in support feature list
            supp_out_0 = self.get_logits(feature_s_ls, BP_1, FP_1)
            ### SSP in support feature list
            SSFP_supp, SSBP_supp, ASFP_supp, ASBP_supp = self.SSP_func(feature_s_ls, supp_out_0)
            ### update prototype for support prediction
            if self.shot > 1:
                FP_supp = FP_nshot * 0.5 + SSFP_supp * 0.5
            else:
                FP_supp = FP * 0.5 + SSFP_supp * 0.5

            BP_supp = SSBP_supp * 0.3 + ASBP_supp * 0.7
            ### use updated prototype to search target in support feature list
            s_out = self.get_logits(feature_s_ls, BP_supp, FP_supp)

            ### process prototype if shot > 1
            if self.shot > 1:
                for i in range(FP_supp.shape[0]//self.shot):
                    for j in range(self.shot):
                        # print("each FP_supp", FP_supp[i*self.shot+j])
                        if j == 0:
                            FP_supp_avg = FP_supp[i*self.shot+j]
                            BP_supp_avg = BP_supp[i*self.shot+j]
                        else:
                            FP_supp_avg = FP_supp_avg + FP_supp[i*self.shot+j]
                            BP_supp_avg = BP_supp_avg + BP_supp[i*self.shot+j]

                    FP_supp_avg = FP_supp_avg/self.shot
                    BP_supp_avg = BP_supp_avg/self.shot
                    FP_supp_avg = FP_supp_avg.reshape(1,FP_supp.shape[1],FP_supp.shape[2],FP_supp.shape[3])
                    BP_supp_avg = BP_supp_avg.reshape(1,BP_supp.shape[1],BP_supp.shape[2],BP_supp.shape[3])
                    if i == 0:
                        new_FP_supp = FP_supp_avg
                        new_BP_supp = BP_supp_avg
                    else:
                        new_FP_supp = torch.cat((new_FP_supp,FP_supp_avg), dim=0)
                        new_BP_supp = torch.cat((new_BP_supp,BP_supp_avg), dim=0)

                FP_supp = new_FP_supp
                BP_supp = new_BP_supp          

        if self.training:
            return q_out, q_out_ssp, s_out, FP_supp, BP_supp
        else:
            return q_out, q_out_ssp
    
    def SSP_func(self, feature_q, out):
        bs, c = feature_q.shape[0:2]
        pred_1 = out.softmax(1).view(bs, 2, -1)
        pred_fg, pred_bg = pred_1[:, 1].contiguous(), pred_1[:, 0].contiguous()
        fg_ls, bg_ls = [], []
        fg_local_ls, bg_local_ls = [], []
        fg_thres, bg_thres = 0.7, 0.6
        
        for epi in range(bs):
            cur_feat = feature_q[epi].view(c, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(c, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(c, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local