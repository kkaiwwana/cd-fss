import os
import glob
import time
import torch
import logging
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F

from copy import deepcopy
from torch.utils.data import Dataset
from typing import Callable, Literal


log = logging.getLogger(__name__)


class EdmCrackDS:
    SPLITS = ['train', 'val', 'test']
    def __new__(cls, data_config):
        return (EdmCrack(split=split, **data_config) for split in EdmCrackDS.SPLITS)
    

class EdmCrack(Dataset):
    """few-shot and out-of-domain dataset, can only use 1 shot"""
    def __init__(
        self, 
        data_path: str, 
        transform: Callable,
        split: Literal['train', 'val', 'test'], 
        n_shot: int, 
        limited_num_train=1, 
        limited_num_val=None,
        paired_aug: Callable = None,
        visual_aug: Callable = None,
        deterministic: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.shot = n_shot
        self.split = split
        
        self.limited_num_train = limited_num_train
        self.limited_num_val = limited_num_val
        
        self.base_path = data_path
        self.img_path = os.path.join(self.base_path, 'images')
        self.ann_path = os.path.join(self.base_path, 'annotations')

        self.categories = ['1']

        self.class_ids = range(0, 1)
        self.img_metadata_classwise, self.num_images = self.build_img_metadata_classwise()

        self.transform = transform
        self.paired_aug = paired_aug
        self.visual_aug = visual_aug
        
        self.deterministic = deterministic
        if deterministic:
            self.seed = 1744941734 # int(time.time())
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
        if self.paired_aug and self.split == 'train':
            _out = self.paired_aug(image=query_img, mask=query_mask) 
            query_img, query_mask = _out['image'], _out['mask']
        
        if self.visual_aug and self.split == 'train':
            query_img = self.visual_aug(image=query_img)['image']
            
        query_img = self.transform(query_img)
        query_mask = F.interpolate(
            torch.tensor(query_mask)[None, None].float(), query_img.size()[-2:], mode='nearest')[0, 0].long()
        
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks = torch.tensor(np.stack(support_masks)).float()
        support_masks = F.interpolate(support_masks[None], support_imgs.shape[-2:], mode='nearest')[0].long()

        data = {
            'support_images': support_imgs, 
            'support_masks': support_masks, 
            'query_image': query_img, 
            'query_mask': query_mask, 
            'class_sample': torch.tensor(class_sample), 
            'support_names': support_names, 
            'query_name': query_name,
        }
        
        return data

    def load_frame(self, query_name, support_names):
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        query_img = np.array(Image.open(os.path.join(self.img_path, os.path.basename(query_name))).convert('RGB'))

        support_ids = [os.path.basename(name) for name in support_names]
        support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
        support_imgs = [np.array(Image.open(name).convert('RGB')) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = np.array(Image.open(img_name).convert('L'))
        mask[mask == 0] = 0
        mask[mask > 0] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]
        
        if self.deterministic:
            np.random.seed(self.seed + idx)
            
        query_name = str(np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0])
        
        _candidates = deepcopy(self.img_metadata_classwise[class_sample])
        _candidates.remove(query_name)
        
        support_names = np.random.choice(_candidates, self.shot, replace=False).tolist()
        del _candidates

        return query_name, support_names, class_id

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.img_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        num_images=0
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata_classwise[cat] += [img_path]
                    num_images += 1
        return img_metadata_classwise, num_images