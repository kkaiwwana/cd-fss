import torch
# import wandb
import swanlab
import numpy as np
from einops import rearrange
from torchmetrics import Metric
from pytorch_lightning.utilities import rank_zero_only


class Visualizer(Metric):
    
    def __init__(self, n_images_per_epoch, unnormalizer=None, down_sample_output=4):
        super().__init__()
        self.add_state('n_received_images', torch.tensor([0], dtype=torch.uint8))
        self.add_state('pred_masks', [])
        self.add_state('gt_masks', [])
        self.add_state('query_images', [])
        
        self.n_images_per_epoch = n_images_per_epoch
        self.unnormalizer = unnormalizer if unnormalizer is not None else lambda x: x
        if down_sample_output is not None:
            self.down_sampler = lambda x: torch.nn.functional.interpolate(
                x, scale_factor=(1 / down_sample_output), mode='bicubic')
            
            def down_long(X, scale_factor=(1 / down_sample_output)):
                H, W = X.shape[-2:]
                iw = torch.linspace(0, W-1, int(W * scale_factor)).long()
                ih = torch.linspace(0, H-1, int(H * scale_factor)).long()
                return X[:, :, ih[:, None], iw]
                            
            self.down_sampler_long = lambda x: down_long(x)
        else:
            self.down_sampler = self.down_sampler_long = lambda x: x
    
    def update(self, logits, target, query_image, *args, **kwargs):
        if self.n_received_images < self.n_images_per_epoch:
            self.pred_masks.append(logits.argmax(dim=1).detach()[0])
            self.gt_masks.append(target[0])
            self.query_images.append(query_image[0])
            
            self.n_received_images += 1
    
    def compute(self, *args, **kwargs):
        def convert_fmt(arr):
            if len(arr.shape) == 4:
                arr = arr[0]
            elif len(arr.shape) == 2:
                arr = arr[None]
                
            if arr.dtype is torch.float:
                arr = self.unnormalizer(arr)
                arr = self.down_sampler(arr[None])[0]
            else:
                arr = self.down_sampler_long(arr[None])[0]
            
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            
            arr = rearrange(arr, 'c h w -> h w c')
            if arr.shape[-1] == 1:
                arr = arr.repeat(repeats=3, axis=-1)
            
            arr = (arr * 255).astype(np.uint8)
            return arr
        
        def make_row(arr_list, margin=32):
            arr_list_margin = []
            for arr in arr_list:
                arr = convert_fmt(arr)
                arr_list_margin.append(arr)
                arr_list_margin.append(np.ones((arr.shape[-2], 50, 3), dtype=np.uint8) * 255)
                
            return np.concatenate(arr_list_margin, axis=1)
        
        query_row = make_row(self.query_images)
        masks_row = make_row(self.gt_masks)
        preds_row = make_row(self.pred_masks)
        
        out_img = np.concatenate([query_row, masks_row, preds_row], axis=0)
        
        # return wandb.Image(out_img)
        return swanlab.Image(out_img)