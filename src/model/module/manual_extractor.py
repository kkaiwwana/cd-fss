import torch
import torchvision
from typing import *


class Resnet50Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights)
        self.layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        self.froze_bn2d(stop_tracking_running_stats=True)
        self.froze_layer(self.layer0)
        self.froze_layer(self.layer1)
    
    def froze_layer(self, layer):
        for p in layer.parameters():
            p.requires_grad = False
        
    def froze_bn2d(self, stop_tracking_running_stats):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                if stop_tracking_running_stats:
                    m.track_running_stats = False
                self.froze_layer(m)
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x
    
    
class SamVitEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint="weights/sam_vit_b_01ec64.pth")
        self.feature_extractor = sam.image_encoder
        self.froze_layer(self.feature_extractor.patch_embed)
        for i, module in enumerate(self.feature_extractor.blocks):
            if i <= 6:
                self.froze_layer(module)
    
    def froze_layer(self, layer):
        for p in layer.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        feat = self.feature_extractor.patch_embed(x) # (b, p, p, dim)
        feat = feat + self.feature_extractor.pos_embed
        
        for module in self.feature_extractor.blocks:
            feat = module(feat)
        
        feat = self.feature_extractor.neck[0](feat.permute(0, 3, 1, 2))
        feat = self.feature_extractor.neck[1](feat)
        
        feat = self.feature_extractor.neck[2](feat)
        feat = self.feature_extractor.neck[3](feat)
        
        return feat
    
    
class ManualFeatureExtractor:
    support_models = {
        'resnet50': Resnet50Extractor,
        'sam_vit_b': SamVitEncoder,
    }
    
    def __new__(cls, name: Literal['resnet50', 'sam_vit_b']):
        return ManualFeatureExtractor.support_models[name]()