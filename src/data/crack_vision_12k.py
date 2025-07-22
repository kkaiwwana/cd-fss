import os
import glob
import time
import torch
import logging
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F

from tensordict import TensorDict
from typing import Callable
from torch.utils.data import Dataset


log = logging.getLogger(__name__)


class CrackVision12kDS:
    SPLITS = ['train', 'val', 'test']
    def __new__(cls, data_config):
        return (CrackVision12k(split=split, **data_config) for split in CrackVision12kDS.SPLITS)
        

class CrackVision12k(Dataset):
    
    def __init__(
        self, 
        data_path, 
        transform: Callable, 
        split: int, 
        n_shot: int, 
        limited_num_val=None,
        limited_num_train=None,
        paired_aug=None,
        visual_aug=None,
        deterministic=False,
        **kwargs,
    ):
        self.shot = n_shot
        self.split = split
        self.limited_num_train = limited_num_train
        self.limited_num_val = limited_num_val
        
        self.base_path = os.path.join(data_path, self.split)
        self.img_path = os.path.join(self.base_path, 'IMG')
        self.ann_path = os.path.join(self.base_path, 'GT')
        self.categories = ['1']

        self.class_ids = range(0, 1)
        self.img_metadata_classwise, self.num_images = self.build_img_metadata_classwise()

        self.transform = transform
        self.paired_aug = paired_aug
        self.visual_aug = visual_aug
        
        self.deterministic = deterministic
        if deterministic:
            self.seed = int(time.time())
            log.info(f'Deterministic dataset {self.__class__}:split={self.split}, numpy.random seed set to {self.seed}.')

    def __len__(self):
        if self.split == 'train' and self.limited_num_train:
            return min(self.limited_num_train, self.num_images)
        elif self.split == 'val' and self.limited_num_val:
            return min(self.limited_num_val, self.num_images)
        else:
            return self.num_images

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)
        
        support_masks = torch.tensor(np.stack(support_masks)).float()
        support_masks = F.interpolate(support_masks[None], support_imgs.shape[-2:], mode='nearest')[0].long()

        data = TensorDict(
            support_images=support_imgs, 
            support_masks=support_masks, 
            query_image=query_img, 
            query_mask=query_mask,
        )
        return data

    def load_frame(self, query_name, support_names):
        query_mask = self.read_mask(query_name)
        support_masks = np.stack([self.read_mask(name) for name in support_names])

        support_ids = [os.path.basename(name) for name in support_names]
        support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
        
        query_img = np.array(Image.open(os.path.join(self.img_path, os.path.basename(query_name))).convert('RGB'))
        if self.paired_aug and self.split == 'train':
            _out = self.paired_aug(image=query_img, mask=query_mask) 
            query_img, query_mask = _out['image'], _out['mask']
        
        if self.visual_aug and self.split == 'train':
            query_img = self.visual_aug(image=query_img)['image']
        
        query_img = self.transform(query_img)
        query_mask = F.interpolate(
            torch.tensor(query_mask)[None, None].float(), query_img.shape[-2:], mode='nearest')[0, 0].long()
        
        support_imgs = np.stack([
            self.transform(np.array(Image.open(name).convert('RGB'))) for name in support_names])

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = np.array(Image.open(img_name).convert('L'))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_name = self.categories[class_id]
        
        if self.deterministic:
            np.random.seed(self.seed + idx)
            
        sampled_targets = list(map(
            lambda x: np.bytes_.decode(x, 'utf-8'), 
            np.random.choice(
                self.img_metadata_classwise[class_name], 1 + self.shot, replace=False).tolist()
        ))

        return sampled_targets[0], sampled_targets[1:], class_id

    def build_img_metadata_classwise(self):
        num_images = 0
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata_classwise[cat] += [img_path]
                    num_images += 1
        
        # to continuous np object, avoid fork + copy on write problem when persistent workers enabled.
        for cat in self.categories:
            img_metadata_classwise[cat] = np.array(img_metadata_classwise[cat], dtype=np.bytes_)
        
        return img_metadata_classwise, num_images