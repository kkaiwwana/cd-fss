r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
import pytorch_lightning as pl

from typing import *
from copy import deepcopy
from tensordict import TensorDict
from hydra.main import DictConfig
from torch.utils.data import DataLoader


from src.data.tiny_crack import TinyCrackDS
from src.data.crack_vision_12k import CrackVision12kDS
from src.data.edm_crack import EdmCrackDS
from src.data.crack_seg import CrackSegDS
from src.data.utils import EpochSubsetSampler


REGISTERED_DATASETS = {
    'tiny_crack': TinyCrackDS,
    'crack_vision_12k': CrackVision12kDS,
    'edm_crack': EdmCrackDS,
    'crack_seg': CrackSegDS,
}


class FSSDataset(pl.LightningDataModule):

    def __init__(self, data_config: DictConfig, loader_config: DictConfig):
        super().__init__()
        self.data_config = data_config
        self.loader_config = loader_config
        self.train_set, self.val_set, self.test_set = (None,) * 3
        self.val_set2 = None
        self.train_shuffle = True
        self.setup()
        
        if 'train_epoch_splits' in self.loader_config.keys() and self.loader_config.train_epoch_splits is not None:
            self.batch_sampler = sampler = EpochSubsetSampler(
                total_size=len(self.train_set), 
                num_splits=self.loader_config.train_epoch_splits, 
                shuffle=self.train_shuffle
            )
        else: self.batch_sampler = None

    def setup(self, stage: str = None):
        self.train_set, self.val_set, self.test_set = REGISTERED_DATASETS[self.data_config['name']](self.data_config)
        
        if 'also_val_at' in self.data_config.keys() and self.data_config['also_val_at'] is not None:
            _, self.val_set2, _ = \
                REGISTERED_DATASETS[self.data_config['also_val_at']['name']](self.data_config['also_val_at'])
        else:
            self.val_set2 = deepcopy(self.val_set)
            self.val_set2.limited_num_val = 1
            
        if 'train_shuffle' in self.loader_config.keys():
            self.train_shuffle = self.loader_config.train_shuffle

    @staticmethod
    def collate_fn_4_tensordict(data_batch: Iterable[TensorDict]):
        out = TensorDict()
        keys = data_batch[0].keys()
        for key in keys:
            out[key]= torch.stack([data[key] for data in data_batch])
        return out
    
    def train_dataloader(self):
        assert self.train_set is not None
        # split massive train epoch to multiple subsets.
        if self.batch_sampler is not None:
            loader = DataLoader(
                self.train_set, 
                **self.loader_config.train_loader_params, 
                sampler=self.batch_sampler, 
                collate_fn=FSSDataset.collate_fn_4_tensordict
            )
        else:
            loader = DataLoader(
                self.train_set, 
                **self.loader_config.train_loader_params, 
                shuffle=self.train_shuffle,
                collate_fn=FSSDataset.collate_fn_4_tensordict
            )
        
        return loader

    def val_dataloader(self):
        assert self.val_set is not None
        val_loaders = [
            DataLoader(
                self.val_set, 
                **self.loader_config.val_loader_params, 
                shuffle=False, 
                collate_fn=FSSDataset.collate_fn_4_tensordict
            ), 
            DataLoader(
                self.val_set2, 
                **self.loader_config.val_loader_params, 
                shuffle=False, 
                collate_fn=FSSDataset.collate_fn_4_tensordict
            )
        ]
        return val_loaders
            

    def test_dataloader(self):
        assert self.test_set is not None
        return DataLoader(
            self.test_set, **self.loader_config, shuffle=False, collate_fn=FSSDataset.collate_fn_4_tensordict)