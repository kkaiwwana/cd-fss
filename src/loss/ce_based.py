import torch
import torch.nn.functional as F

from einops import rearrange
from torch.nn import CrossEntropyLoss

from src.loss.utils import MaskSmoother
from typing import *

class SmoothTopkCELoss(CrossEntropyLoss):
    
    def __init__(self, k_ratio: float, mask_smoother: MaskSmoother = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_ratio = k_ratio
        if mask_smoother is not None:
            self.smoother = mask_smoother
        else:
            self.smoother = lambda x: x

    @staticmethod
    def unravel_index(indices, shape):
        r"""Converts flat indices into unraveled coordinates in a target shape.
        This is a `torch` implementation of `numpy.unravel_index`.
        Args:
            indices: A tensor of indices, (*, N).
            shape: The targeted shape, (D,).
        Returns:
            unravel coordinates, (*, N, D).
        """
        shape = torch.tensor(shape)
        indices = indices % shape.prod()  # prevent out-of-bounds indices
        coord = torch.zeros(indices.size() + shape.size(), dtype=int)
        for i, dim in enumerate(reversed(shape)):
            coord[..., i] = indices % dim
            indices = indices // dim
        return coord.flip(-1)
        
    def forward(self, input, target):
        raw_loss =  F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        b, h, w = raw_loss.shape
        
        raw_loss = rearrange(raw_loss, 'b h w -> b (h w)')
        k = int(h * w * self.k_ratio)
        v, idx = torch.topk(raw_loss, k)
        raw_loss = rearrange(raw_loss, 'b (h w) -> b h w', h=h)
        
        indices = SmoothTopkCELoss.unravel_index(idx, (h, w))
        
        b_idx = torch.arange(b)[..., None].expand(-1, k).reshape(-1)
        h_idx = indices[..., 0].reshape(-1)
        w_idx = indices[..., 1].reshape(-1)

        mask = torch.zeros_like(raw_loss)
        mask[b_idx, h_idx, w_idx] = 1.0
        soft_mask = self.smoother(mask[:, None])[:, 0]
        loss = raw_loss * soft_mask
        return loss.sum() / soft_mask.sum()