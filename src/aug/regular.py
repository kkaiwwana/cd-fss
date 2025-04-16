import torch
from torchvision import transforms

class ResNetAug:
    """normalize image for resnet backbone"""
    def __new__(cls, img_mean, img_std, img_size):
        return transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ])
        
class InverseResNetAug:
    def __new__(cls, img_mean, img_std, img_size=None):
        img_mean = torch.tensor(img_mean)
        img_std = torch.tensor(img_std)
        return transforms.Compose([
            transforms.Normalize(- img_mean / img_std, 1 / img_std),
        ])