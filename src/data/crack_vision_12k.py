import os
import glob
import torch
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F

from torch.utils.data import Dataset


class CrackVision12kDS:
    SPLITS = ['train', 'val', 'test']
    def __new__(cls, data_config):
        return (CrackVision12k(split=split, **data_config) for split in CrackVision12kDS.SPLITS)
        

class CrackVision12k(Dataset):
    
    def __init__(self, data_path, transform, split, n_shot, limited_num_val=None, **kwargs):
        self.benchmark = 'crack12k'
        self.shot = n_shot
        self.split = split
        self.limited_num_val = limited_num_val

        self.base_path = os.path.join(data_path, self.split)
        self.img_path = os.path.join(self.base_path, 'IMG')
        self.ann_path = os.path.join(self.base_path, 'GT')
        self.categories = ['1']

        self.class_ids = range(0, 1)
        self.img_metadata_classwise, self.num_images = self.build_img_metadata_classwise()

        self.transform = transform

    def __len__(self):
        return min(self.limited_num_val, self.num_images) if self.split == 'val' and self.limited_num_val else self.num_images

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask[None, None].float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for mask in support_masks:
            mask = F.interpolate(mask[None, None].float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(mask)
        support_masks = torch.stack(support_masks_tmp)

        data = {
            'support_images': support_imgs, 
            'support_masks': support_masks.long(), 
            'query_image': query_img, 
            'query_mask': query_mask.long(), 
            'class_sample': torch.tensor(class_sample), 
            'support_names': support_names, 
            'query_name': query_name,
        }
        
        return data

    def load_frame(self, query_name, support_names):
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        query_id = query_name[:-3] + 'png'
        query_img = Image.open(os.path.join(self.img_path, os.path.basename(query_id))).convert('RGB')

        support_ids = [os.path.basename(name)[:-3] + 'png' for name in support_names]
        support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

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