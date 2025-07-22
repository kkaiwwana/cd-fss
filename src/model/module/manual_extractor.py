import torch
import torchvision
from typing import *


class Resnet50Extractor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
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
    


from torch import nn
from torch import Tensor
import torch
import math
from typing import *
from einops import einsum

from operator import attrgetter
from src.model.module.selective_axial_attn import AxialAttention


from operator import attrgetter

def set_children_layer(module: torch.nn.Module, layer_name: list[str], layer):
    # recursively set model's target layer to a target model.
    if not list(module.children()) or layer_name is None:
        return layer
    else:
        if layer_name[0].isdigit():
            # index (of Sequential, ModuleList) case
            module[int(layer_name[0])] = set_children_layer(
                module[int(layer_name[0])], layer_name[1:] if len(layer_name) > 1 else None, layer)
        else:
            # named module case
            setattr(module, layer_name[0], set_children_layer(
                    attrgetter(layer_name[0])(module), layer_name[1:] if len(layer_name) > 1 else None, layer))
        return module


class ZeroInitUpSampler(torch.nn.Module):
    def __init__(self, c_in, c_out, num_features):
        super().__init__()
        self.upsampler = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.zero_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=num_features),
            torch.nn.ReLU(),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.zero_init(self.zero_upsampler)
    
    def forward(self, x):
        return self.upsampler(x) + self.zero_upsampler(x)

    @torch.no_grad()
    def zero_init(self, module):
        for p in module.parameters():
            p.fill_(0.0)


class SamVitEncoder(torch.nn.Module):
    def __init__(
        self,
        froze_depth=2,
        use_axial_attn=False,
        use_selector=False,
        selector_config=None,
        row_selector_config=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint="weights/sam_vit_b_01ec64.pth")
        encoder = sam.image_encoder
        
        self.num_features = None
        if use_axial_attn:
            for k, v in encoder.named_modules():
                name = k.split('.')
                if name[-1] == 'attn' and (v.rel_pos_h.shape[0] + 1) / 2 == 64:
                    self.num_features = v.qkv.in_features
                    
                    axial_attn = AxialAttention(
                        dim=self.num_features,
                        dim_index=-1,
                        num_dimensions=2,
                        heads=v.num_heads,
                        use_selector=use_selector,
                        row_selector_config=row_selector_config,
                        selector_config=selector_config,
                    )
                    encoder = set_children_layer(encoder, name, axial_attn)
                    # encoder = set_children_layer(encoder, name, torch.nn.Identity())
        
        self.froze_layer(encoder.patch_embed)
        for i, module in enumerate(encoder.blocks):
            if module.window_size > 0 and i < froze_depth:
                self.froze_layer(module)
        
        self.feature_extractor = torch.nn.Sequential(encoder,)

    
    def inherit_weight(self, original_attn, axial_attn):
        ori_to_q, ori_to_kv = original_attn.qkv.weight.split((self.num_features, self.num_features * 2))
        
        if hasattr(axial_attn.axial_attentions[0].fn, 'to_q'):
            ax_attn_0 = axial_attn.axial_attentions[0].fn  
            ax_attn_1 = axial_attn.axial_attentions[1].fn
        else:
            ax_attn_0 = axial_attn.axial_attentions[0].fn.attn
            ax_attn_1 = axial_attn.axial_attentions[1].fn.attn
        
        
        ax_attn_0.to_q.weight = torch.nn.Parameter(ori_to_q)
        ax_attn_1.to_q.weight = torch.nn.Parameter(ori_to_q)

        
        ax_attn_0.to_kv.weight = torch.nn.Parameter(ori_to_kv)
        ax_attn_1.to_kv.weight = torch.nn.Parameter(ori_to_kv)
        
        ax_attn_0.to_out.load_state_dict(original_attn.proj.state_dict())
        ax_attn_1.to_out.load_state_dict(original_attn.proj.state_dict())
        
        print('inherited!')
        
    
    def froze_layer(self, layer):
        for p in layer.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        return self.feature_extractor(x)

    
class ManualFeatureExtractor:
    support_models = {
        'resnet50': Resnet50Extractor,
        'sam_vit_b': SamVitEncoder,
    }
    
    def __new__(cls, name: Literal['resnet50', 'sam_vit_b'], *args, **kwargs):
        return ManualFeatureExtractor.support_models[name](*args, **kwargs)