import torch
import wandb
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from functools import reduce
from copy import deepcopy

from typing import *
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

from src.model.utils import WanbSyncLogger
from src.model.module.auto_feature_pyramid import AutoFeaturePyramid
from src.model.module.manual_extractor import ManualFeatureExtractor
from src.callback.checkpoint import Checkpoint

OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
}

SCHEDULERS = {
    'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
}

# for auto feature pyramid extractor
MODELS = {
    
}


class MatchingNetIFA(pl.LightningModule):
    """a naive toy segmenter built for test"""
    def __init__(
        self,
        loss_func,
        metrics,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig = None,
        cfg: DictConfig = None,
        checkpointing: Checkpoint = None,
    ):
        super().__init__()
        self.loss_func = loss_func
        self.metrics_list = torch.nn.ModuleList([metrics, metrics.clone()])
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.checkpointing = checkpointing
        self.cfg = cfg

        self.save_hyperparameters(
            cfg, ignore=['loss_func', 'metrics', 'model_cfg', 'optimizer_cfg', 'scheduler_cfg'])
        
        self.model = IFA_MatchingNet(
            backbone=model_cfg.backbone,
            refine=model_cfg.refine,
            use_auto_extractor=model_cfg.use_auto_extractor,
            shot=cfg.experiment.n_shot,
        )
        
        self._logger = None
        self._skip_val_set2 = False
    
    def call_model_train(self, *args, **kwargs):
        """TO-BE-OVERRIDDEN"""
        return self.model(*args, **kwargs)

    def call_model_eval(self, *args, **kwargs):
        """TO-BE-OVERRIDDEN"""
        return self.model(*args, **kwargs)
    
    def call_loss_func(self, out: List, mask_q, mask_s):
        """TO-BE-OVERRIDDEN"""
        call_ls = lambda x, y: self.loss_func(x, y)
        def sum_dict(dicts, weights):
            return {key: sum([d[key] * w for w, d in zip(weights, dicts)]) for key in dicts[0].keys()}
        
        l0, d0 = call_ls(out[0], mask_q)
        l1, d1 = call_ls(out[1], mask_q)
        
        l2, d2 = call_ls(out[2], mask_q) if len(out) > 2 else (0, {k: 0 for k in d0.keys()})
        l3, d3 = call_ls(out[3], mask_s) if len(out) > 2 else (0, {k: 0 for k in d0.keys()})
        l4, d4 = call_ls(out[4], mask_s) if len(out) > 2 else (0, {k: 0 for k in d0.keys()})

        if self.model_cfg.refine:
            total_loss = l0 + l1 + l2 + l3 * 0.2 + l4 * 0.4
            loss_dict = sum_dict((d0, d1, d2, d3, d4), (1, 1, 1, 0.2, 0.4))
        else:
            # disable l3
            total_loss = l0 + l1 + l2 + l3 * 0 + l4 * 0.4
            loss_dict = sum_dict((d0, d1, d2, d3, d4), (1, 1, 1, 0.0, 0.4))
    
        return total_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        img_q, mask_q = batch['query_image'], batch['query_mask']
        img_s_list  = batch['support_images'].unbind(dim=1)
        mask_s_list = batch['support_masks'].unbind(dim=1)
        out_ls = self.call_model_train(img_s_list, mask_s_list, img_q, mask_q)
        mask_s = torch.cat(mask_s_list, dim=0).long()
        
        total_loss, output_dict = self.call_loss_func(out_ls, mask_q, mask_s)
        
        self.metrics_list[0].update(out_ls[0], mask_q, query_image=img_q)
    
        if self.trainer is not None:
            self._logger.log({f'train-losses/total_loss': total_loss.detach()}, 
                                on_step=True, on_epoch=True, step=self.trainer.global_step)
            self._logger.log({f'train-losses/{k}': v.detach() for k, v in output_dict.items()}, 
                                on_step=True, on_epoch=True, step=self.trainer.global_step)
            self._logger.add_step()

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 1 and self._skip_val_set2:
            return
        
        img_q, mask_q = batch['query_image'], batch['query_mask']
        img_s_list  = batch['support_images'].unbind(dim=1)
        mask_s_list = batch['support_masks'].unbind(dim=1)
        out_ls = self.call_model_train(img_s_list, mask_s_list, img_q, mask_q)
        mask_s = torch.cat(mask_s_list, dim=0).long()

        total_loss, output_dict = self.call_loss_func(out_ls, mask_q, mask_s)
        
        self.metrics_list[dataloader_idx].update(out_ls[0], mask_q, query_image=img_q)
        
        if self.trainer is not None:
            self._logger.log({f'val-losses-ds{dataloader_idx}/total_loss': total_loss.detach()}, 
                                on_step=False, on_epoch=True, step=self.trainer.global_step)
            self._logger.log({f'val-losses-ds{dataloader_idx}/{k}': v.detach() for k, v in output_dict.items()}, 
                                on_step=False, on_epoch=True, step=self.trainer.global_step)
        return {'losses': total_loss}

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')

    @rank_zero_only
    def on_validation_epoch_end(self):
        self._log_epoch_metrics('val')
        self._logger.add_epoch()
        self._logger.log({}, on_epoch=True)
        self._logger.sync_epoch()

    def on_train_start(self):
        if self._logger is None:
            self._logger = WanbSyncLogger(run=wandb.run, log_every_n_steps=self.trainer.log_every_n_steps)
        self._skip_val_set2 = self.trainer.val_dataloaders[0].dataset.__class__ == \
            self.trainer.val_dataloaders[1].dataset.__class__
    
    def _log_epoch_metrics(self, prefix: str):
        # other metrics do not require dl & model, but they accept kwargs anyway.
        metrics_to_log = self.metrics_list if (prefix == 'val' and not self._skip_val_set2) else self.metrics_list[:1]
        for i, metrics in enumerate(metrics_to_log):
            res = metrics.compute()
            if prefix == 'val' and i == len(metrics_to_log) - 1:
                self.checkpointing(res, self, self.trainer.current_epoch)
            for metric_name, result in res.items():
                if result is not None:
                    if isinstance(result, dict):
                        for term, values in result.items():
                            self._logger.log({f'{prefix}-metrics-ds{i}/{metric_name}': values}, step=self.trainer.global_step)
                    else:
                        self._logger.log({f'{prefix}-metrics-ds{i}/{metric_name}': result}, step=self.trainer.global_step)

            metrics.reset()

    def configure_optimizers(self):
        assert self.optimizer_cfg.name in OPTIMIZERS.keys(), \
            f'The specified optimizer: {self.optimizer_cfg.name} is not supported. Supported ones: {OPTIMIZERS.keys()}'
        assert self.scheduler_cfg.name in SCHEDULERS.keys(), \
            f'The specified scheduler: {self.scheduler_cfg.name} is not supported. Supported ones: {SCHEDULERS.keys()}'

        optimizer = OPTIMIZERS[self.optimizer_cfg.name](self.model.parameters(), **self.optimizer_cfg.optimizer_params)
        scheduler = SCHEDULERS[self.scheduler_cfg.name](optimizer, **self.scheduler_cfg.scheduler_params)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    

class IFA_MatchingNet(torch.nn.Module):
    def __init__(self, backbone='sam_vit_b', refine=False, use_auto_extractor=False, shot=1):
        super().__init__()
        self.use_auto_extractor = use_auto_extractor
        if use_auto_extractor:
            self.backbone = AutoFeaturePyramid(
                n_layers=8, build_channel_adapter=True, target_channels=[128]*8,input_size=896, 
                froze_shallow_percent=0.2,
            )(MODELS[backbone])
        else:
            self.backbone = ManualFeatureExtractor(backbone)
        
        self.refine = refine
        self.shot = shot
        self.iter_refine = False

    def fuse_feature_pyramid(self, feature_pyramid: dict, target_size: int, target_c: int):
        def _interpolate(x):
            return F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=True)
        
        return torch.cat([_interpolate(feature) for feature in feature_pyramid.values()], dim=1)

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):
        h, w = img_q.shape[-2:]

        # feature maps of support images
        feature_s_list = []
        for k in range(len(img_s_list)):
            if self.use_auto_extractor:
                feature_pyramid = self.backbone(img_s_list[k])
                feature = self.fuse_feature_pyramid(feature_pyramid, target_size=h // 8)
            else:
                feature = self.backbone(img_s_list[k])
            feature_s_list.append(feature)

        feature_s_ls = torch.cat(feature_s_list, dim=0)
        # feature map of query image
        if self.use_auto_extractor:
            feature_q = self.fuse_feature_pyramid(self.backbone(img_q), target_size=h // 8)
        else:
            feature_q = self.backbone(img_q)
            
        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list, feature_bg_list, supp_out_ls = [], [], []

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

        if self.training:

            ### iter = 1 (BFP)
            if self.refine:
                out_refine, out_1, supp_out_1, new_FP, new_BP = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            else:
                out_1, supp_out_1, new_FP, new_BP = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
                
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
            supp_out_1 = F.interpolate(supp_out_1, size=(h, w), mode="bilinear", align_corners=True)
            ### iter = 2
            out_2, supp_out_2, new_FP, new_BP = self.iter_BFP(new_FP, new_BP, feature_s_ls, feature_q, self.iter_refine)
            out_2 = F.interpolate(out_2, size=(h, w), mode="bilinear", align_corners=True)
            supp_out_2 = F.interpolate(supp_out_2, size=(h, w), mode="bilinear", align_corners=True)
            ### iter = 3
            out_3, supp_out_3, new_FP, new_BP = self.iter_BFP(new_FP, new_BP, feature_s_ls, feature_q, self.iter_refine)
            out_3 = F.interpolate(out_3, size=(h, w), mode="bilinear", align_corners=True)
            supp_out_3 = F.interpolate(supp_out_3, size=(h, w), mode="bilinear", align_corners=True)

        else:
            if self.refine:
                out_refine, out_1 = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            else:
                out_1 = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        if self.refine:
            out_refine = F.interpolate(out_refine, size=(h, w), mode="bilinear", align_corners=True)
            out_ls = [out_refine, out_1]
        else:
            out_ls = [out_1]

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
            ### iter = 1 (BFP)
            out_ls.append(supp_out_1)
            ### iter = 2
            out_ls.append(out_2)
            out_ls.append(supp_out_2)
            ### iter = 3
            out_ls.append(out_3)
            out_ls.append(supp_out_3)

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
    
    def iter_BFP(self, FP, BP, feature_s_ls, feature_q, refine=True):
        ###### input FP and BP are support prototype
        ###### SSP on query side
        ### find the most similar part in query feature
        out_0 = self.similarity_func(feature_q, FP, BP)
        ### SSP in query feature
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)
        ### update prototype for query prediction
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        ### use updated prototype to search target in query feature
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)
        ###### Refine (only for the 1st iter)
        if refine:
            ### use updated prototype to find the most similar part in query feature again
            SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1)
            ### update prototype again for query regine
            FP_2 = FP * 0.5 + SSFP_2 * 0.5
            BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7
            FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
            BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3
            ### use updated prototype to search target in query feature again
            out_refine = self.similarity_func(feature_q, FP_2, BP_2)
            out_refine = out_refine * 0.7 + out_1 * 0.3

        ###### SSP on support side
        if self.training:
            ### duplicate query prototype for support SSP if shot > 1
            if self.shot > 1:
                FP_nshot = FP.repeat_interleave(self.shot, dim=0)
                FP_1 = FP_1.repeat_interleave(self.shot, dim=0)
                BP_1 = BP_1.repeat_interleave(self.shot, dim=0)
            ### find the most similar part in support feature list
            supp_out_0 = self.similarity_func(feature_s_ls, FP_1, BP_1)
            ### SSP in support feature list
            SSFP_supp, SSBP_supp, ASFP_supp, ASBP_supp = self.SSP_func(feature_s_ls, supp_out_0)
            ### update prototype for support prediction
            if self.shot > 1:
                FP_supp = FP_nshot * 0.5 + SSFP_supp * 0.5
            else:
                FP_supp = FP * 0.5 + SSFP_supp * 0.5

            BP_supp = SSBP_supp * 0.3 + ASBP_supp * 0.7
            ### use updated prototype to search target in support feature list
            supp_out_1 = self.similarity_func(feature_s_ls, FP_supp, BP_supp)

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

        if refine:
            if self.training:
                return out_refine, out_1, supp_out_1, FP_supp, BP_supp
            else:
                return out_refine, out_1
        else:
            if self.training:
                return out_1, supp_out_1, FP_supp, BP_supp
            else:
                return out_1