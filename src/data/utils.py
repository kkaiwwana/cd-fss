import torch
import random
from torch.utils.data import Sampler


class EpochSubsetSampler(Sampler):
    def __init__(self, total_size, num_splits, shuffle=True, seed=42):
        indices = list(range(total_size))
        self.do_shuffle = shuffle
        
        self.epoch_indices = [indices[i::num_splits] for i in range(num_splits)]
        self.seed = seed
        self.epoch = -1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        self.epoch += 1
        indices = list(self.epoch_indices[self.epoch % len(self.epoch_indices)])
        if self.do_shuffle:
            rng = random.Random(self.seed + self.epoch)  # 保证每个 epoch 可复现性但不同顺序
            rng.shuffle(indices)
        return iter(indices)
        

    def __len__(self):
        return len(self.epoch_indices[self.epoch % len(self.epoch_indices)])
    

def interpolate_int_tensor(X, scale_factor):
    H, W = X.shape[-2:]
    iw = torch.linspace(0, W-1, int(W * scale_factor)).long()
    ih = torch.linspace(0, H-1, int(H * scale_factor)).long()
    return X[:, :, ih[:, None], iw]