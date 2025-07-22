
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from einops import einsum, rearrange
from typing import *

from src.model.base import BaseSegmenter, MODELS
from src.model.module.auto_feature_pyramid import AutoFeaturePyramid
from src.model.module.manual_extractor import ManualFeatureExtractor


class SSPSegmenterPL(BaseSegmenter):
    """a naive toy segmenter built for test"""

    def setup_model(self):
        return SSP_MatchingNet(
            backbone=ManualFeatureExtractor(self.model_cfg.backbone),
            shot=self.cfg.exp.n_shot,
        )
    
    def call_loss_func(self, out: List, mask_q, mask_s):
        """TO-BE-OVERRIDDEN"""
        call_ls = lambda x, y: self.loss_func(x, y)
        def sum_dict(dicts, weights):
            return {key: sum([d[key] * w for w, d in zip(weights, dicts)]) for key in dicts[0].keys()}
        
        if len(out) > 2:
            
            l0, d0 = call_ls(out[0], mask_q)
            l1, d1 = call_ls(out[1], mask_q)
            l2, d2 = call_ls(out[2], mask_q)
            l3, d3 = call_ls(out[3], mask_s)
            
            total_loss = l0 + l1 + l2 + 0.2 * l3
            loss_dict = sum_dict((d0, d1, d2, d3), (1, 1, 1, 0.2))
        else:
            total_loss, loss_dict = call_ls(out[0], mask_q)
        
        return total_loss, loss_dict



class SSP_MatchingNet(torch.nn.Module):
    def __init__(self, backbone, shot, *args, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.refine = True
        self.shot = shot
        self.temp = 10.0

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):
        h, w = img_q.shape[-2:]

        feature_s_list = []
        for k in range(len(img_s_list)):
            feature_s_list.append(self.backbone(img_s_list[k]))
        # feature_s_ls = torch.cat(feature_s_list, dim=0)
        feature_q = self.backbone(img_q)
        

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        supp_out_ls = []
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            if self.training:
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0

                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        # measure the similarity of query features to fg/bg prototypes
        out_0 = self.similarity_func(feature_q, FP, BP)

        ##################### Self-Support Prototype (SSP) #####################
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)
        
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

        out_1 = self.similarity_func(feature_q, FP_1, BP_1)

        ##################### SSP Refinement #####################
        if self.refine:
            SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1)

            FP_2 = FP * 0.5 + SSFP_2 * 0.5
            BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7

            FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
            BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3

            out_2 = self.similarity_func(feature_q, FP_2, BP_2)

            out_2 = out_2 * 0.7 + out_1 * 0.3

        #out_0 = F.interpolate(out_0, size=(h, w), mode="bilinear", align_corners=True)
        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        if self.refine:
            out_2 = F.interpolate(out_2, size=(h, w), mode="bilinear", align_corners=True)
            out_ls = [out_2, out_1]
        else:
            out_ls = [out_1]

        fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)

            out_ls.append(self_out)
            out_ls.append(supp_out)

        import time
        torch.save(fg_q, f'temp_tensor_out/{str(time.time())}.pt')
        
        return out_ls


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

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature