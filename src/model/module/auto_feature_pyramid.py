import re
import torch
import torchvision
import numpy as np
from typing import *
from functools import reduce
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


class FeatureExtractor(torch.nn.Module):
    def __init__(self, extractor, channel_adapter: List[torch.nn.Module] = None):
        super().__init__()
        self.extractor = extractor
        self.adapter = torch.nn.ModuleList(channel_adapter)
        

    def forward(self, x):
        if not self.adapter: 
            return self.extractor(x)
        else:
            return {name: conv(fea) for (name, fea), conv in zip(self.extractor(x).items(), self.adapter)}


class AutoFeaturePyramid:
    """given arbitrary model and expected `n-layers`,
    automatically build a feature extractor"""
    
    # matching low-level layers is all you need.
    LAYER_DEPTH = {
        'classifier': None,
        'cat|add|reshape|flatten': None,
        'conv': 1.0,
        'fc|linear|mlp': 0.5,
        'feature': 0.5,  # work for some old network, e.g. alexnet
        'bn|ln|norm': 0.0,  # higher priority when following a conv/norm
        '(re|ge|si)lu|sigmoid|tanh': 0.0,
        '.*': None,
    }
    
    def __init__(
            self, 
            n_layers: int, 
            build_channel_adapter: bool =  False, 
            target_channels: List = None,
            input_size: int = None,
            froze_shallow_percent: int = 0.4,
            verbose: bool = False
    ):
        self.n_nodes = n_layers
        self.verbose = verbose
        self.if_build_channel_adapter = build_channel_adapter 
        self.input_size = input_size
        self.target_channels = target_channels if target_channels else [None] * n_layers
        self.froze_shallow_percent = froze_shallow_percent
        
    def __call__(self, model: torch.nn.Module)-> torch.nn.Module:
        model = AutoFeaturePyramid.froze_shallow_layers(model, frozen_percent=self.froze_shallow_percent)
        train_nodes, eval_nodes = get_graph_node_names(model)
        
        shared_nodes = list(filter(lambda x: x in set(eval_nodes), train_nodes))
        # also remove last layer, which is typically a FC coveting feature to vec
        selected_nodes = AutoFeaturePyramid.sample_nodes(shared_nodes[:-1], self.n_nodes)
        assert selected_nodes, 'No valid nodes found. Please check the model or modify the matching rule.'
        if self.verbose: print(selected_nodes)
        
        extractor = create_feature_extractor(model, return_nodes=selected_nodes)
        
        if not self.if_build_channel_adapter:
            return FeatureExtractor(extractor)
        else:
            conv_adapter = self.build_channel_adapter(extractor=extractor)
            return FeatureExtractor(extractor, conv_adapter)

    def build_channel_adapter(self, extractor):
        device = next(extractor.parameters()).device
        example_input = torch.rand(1, 3, self.input_size, self.input_size, device=device)
        features = extractor(example_input)
        assert len(self.target_channels) == len(features)

        def adapter(c_in, c_out):
            return torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=1, in_channels=c_in, out_channels=c_out),
                torch.nn.BatchNorm2d(num_features=c_out),
                torch.nn.ReLU(),
                torch.nn.Conv2d(kernel_size=1, in_channels=c_out, out_channels=c_out),
            )
        
        conv_11 = [adapter(fea.shape[1], o_chn if o_chn else fea.shape[1]) 
                            for fea, o_chn in zip(features.values(), self.target_channels)]
        return conv_11
    
    @staticmethod
    def froze_shallow_layers(model: torch.nn.Module, frozen_percent=.3):
        def all_modules(m: torch.nn.Module):
            chs = list(m.children())
            return reduce(lambda a, b: a + b, [all_modules(ch) for ch in chs]) if chs else [m]
        
        modules = all_modules(model)
        for i in range(int(len(modules) * min(frozen_percent, 1.0))):
            for p in modules[i].parameters():
                p.requires_grad = False
    
        return model
    
    @staticmethod
    def re_match_k2v(target: str, rules: dict):
        for rule, value in rules.items():
            if re.search(rule, target, flags=re.IGNORECASE):
                return target, value

    @staticmethod
    def try_uniform_sampling(layer_depth, n):
        num_layers = len(layer_depth)
        n = min(num_layers, n)

        # ideal distance
        d_ideal = (layer_depth[-1] - layer_depth[0]) / (n - 1)
        
        # init DP mat
        dp = np.full((num_layers, n + 1), float('inf'))  # save var values as cost
        prev = np.full((num_layers, n + 1), -1)  # record path for backtrack
        dp[:, 1] = 0

        for j in range(2, n + 1):
            for i in range(j - 1, num_layers):
                for k in range(j - 2, i): 
                    cost = dp[k][j-1] + (layer_depth[i] - layer_depth[k] - d_ideal) ** 2
                    if cost < dp[i][j] or (cost == dp[i][j] and k > prev[i][j]):
                        dp[i][j] = cost
                        prev[i][j] = k
                        
        indices = []
        idx = max([i for i in range(num_layers) if dp[i][n] == np.min(dp[:, n])])
        for j in range(n, 0, -1):
            indices.append(idx)
            idx = prev[idx][j]

        # reverse backtrack results
        return reversed(indices)

    @staticmethod
    def sample_nodes(nodes: List[str], n: int) -> List[str]:
        nodes = [AutoFeaturePyramid.re_match_k2v(node, AutoFeaturePyramid.LAYER_DEPTH) for node in nodes]
        nodes = filter(lambda x: x[1] is not None, nodes)
        named_nodes, nodes_depth = [], []
        depth_acc = 0
        for name, weight in nodes:
            named_nodes.append(name)
            depth_acc += weight
            nodes_depth.append(depth_acc)
            
        selected_indices = AutoFeaturePyramid.try_uniform_sampling(nodes_depth, n)
        return {named_nodes[idx]: f'layer{i}' for i, idx in enumerate(selected_indices)}