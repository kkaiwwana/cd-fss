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


class BaseSegmenter(pl.LightningModule):
    """Base segmenter using meta learning."""
    def __init__(
        self,
        loss_func,
        metrics,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig = None,
        cfg: DictConfig = None,
        checkpointing: Callable = None,
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
        
        self.model = self.setup_model()
        
        self._logger = None
        self._skip_val_set2 = False
    
    def setup_model(self) -> torch.nn.Module:
        raise NotImplementedError
    
    def call_model_train(self, *args, **kwargs):
        """TO-BE-OVERRIDDEN"""
        return self.model(*args, **kwargs)

    def call_model_eval(self, *args, **kwargs):
        """TO-BE-OVERRIDDEN"""
        return self.model(*args, **kwargs)
    
    def call_loss_func(self, out: Any, mask_q, mask_s):
        """TO-BE-OVERRIDDEN"""
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        img_q, mask_q = batch['query_image'], batch['query_mask']
        img_s_list  = batch['support_images'].unbind(dim=1)
        mask_s_list = batch['support_masks'].unbind(dim=1)
        out_ls = self.call_model_train(img_s_list, mask_s_list, img_q, mask_q)
        mask_s = torch.cat(mask_s_list, dim=0).long()
        
        total_loss, loss_dict = self.call_loss_func(out_ls, mask_q, mask_s)
        
        self.metrics_list[0].update(out_ls[0], mask_q, query_image=img_q, stage='train')
    
        if self.trainer is not None:
            self._log_step_metrics(total_loss, loss_dict, prefix='train')

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        
        if dataloader_idx == 1 and self._skip_val_set2:
            return None
        
        img_q, mask_q = batch['query_image'], batch['query_mask']
        img_s_list  = batch['support_images'].unbind(dim=1)
        mask_s_list = batch['support_masks'].unbind(dim=1)
        out_ls = self.call_model_train(img_s_list, mask_s_list, img_q, mask_q)
        mask_s = torch.cat(mask_s_list, dim=0).long()

        total_loss, loss_dict = self.call_loss_func(out_ls, mask_q, mask_s)
        
        self.metrics_list[dataloader_idx].update(out_ls[0], mask_q, query_image=img_q, stage='val')
        
        if self.trainer is not None:
            self._log_step_metrics(total_loss, loss_dict, prefix='val', dataloader_idx=dataloader_idx)
            
        return {'losses': total_loss}
    
    def _log_step_metrics(self, total_loss, loss_dict, prefix: Literal['train', 'val'], dataloader_idx: int = None):

        if dataloader_idx is not None:
            self._logger.log({f'{prefix}-losses-ds{dataloader_idx}/total_loss': total_loss.detach()}, 
                                    on_step=False, on_epoch=True, step=self.trainer.global_step)
            self._logger.log({f'{prefix}-losses-ds{dataloader_idx}/{k}': v.detach() for k, v in loss_dict.items()}, 
                                    on_step=False, on_epoch=True, step=self.trainer.global_step)
        else:
            self._logger.log({f'{prefix}-losses/total_loss': total_loss.detach()}, 
                                on_step=True, on_epoch=True, step=self.trainer.global_step)
            self._logger.log({f'{prefix}-losses/{k}': v.detach() for k, v in loss_dict.items()}, 
                                on_step=True, on_epoch=True, step=self.trainer.global_step)
        
        if prefix == 'train':
            self._logger.add_step()

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')
        
        if self.trainer.val_dataloaders is not None:
            self._skip_val_set2 = self.trainer.val_dataloaders[0].dataset.__class__ == \
                self.trainer.val_dataloaders[1].dataset.__class__

    def on_validation_epoch_end(self):
        self._log_epoch_metrics('val')

        self._logger.add_epoch()
        self._logger.log({}, on_epoch=True)
        self._logger.sync_epoch()

    def on_train_start(self):
        self.checkpointing = self.checkpointing(None)  # init checkpointing at rank zero.
        if self._logger is None:
            self._logger = WanbSyncLogger(run=wandb.run, log_every_n_steps=self.trainer.log_every_n_steps) 
    
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
                            values = values.mean() if isinstance(values, torch.Tensor) else values
                            self._logger.log(
                                {f'{prefix}-metrics-ds{i}/{metric_name}': values.mean()}, step=self.trainer.global_step)
                    else:
                        result = result.mean() if isinstance(result, torch.Tensor) else result
                        self._logger.log(
                            {f'{prefix}-metrics-ds{i}/{metric_name}': result}, step=self.trainer.global_step)

            metrics.reset()

    def configure_optimizers(self):
        assert self.optimizer_cfg.name in OPTIMIZERS.keys(), \
            f'The specified optimizer: {self.optimizer_cfg.name} is not supported. Supported ones: {OPTIMIZERS.keys()}'
        assert self.scheduler_cfg.name in SCHEDULERS.keys(), \
            f'The specified scheduler: {self.scheduler_cfg.name} is not supported. Supported ones: {SCHEDULERS.keys()}'
        
        from src.model.module.selective_axial_attn import AxialAttention
        p_small_lr = set()
        for k, v in self.model.named_modules():
            if isinstance(v, AxialAttention):
                p_small_lr |= set(v.parameters())

        p_normal_lr = set(self.model.parameters()) - p_small_lr
        
        lr = self.optimizer_cfg.optimizer_params.lr * 1.0
        # del self.optimizer_cfg.optimizer_params.lr
        
        optimizer = OPTIMIZERS[self.optimizer_cfg.name](
            [
                {'params': list(p_small_lr), 'lr': lr},
                {'params': list(p_normal_lr), 'lr': lr}
            ],
            **self.optimizer_cfg.optimizer_params
        )
        scheduler = SCHEDULERS[self.scheduler_cfg.name](optimizer, **self.scheduler_cfg.scheduler_params)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]