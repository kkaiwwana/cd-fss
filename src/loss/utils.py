import torch
import numpy as np


class WeightedLoss(torch.nn.Module):
    def __init__(self, loss_func: torch.nn.Module, weight=1.0):
        super().__init__()
        self.loss_func = loss_func
        self.weight = weight

    def __call__(self, y_hat, y, **kwargs):
        return self.loss_func(y_hat, y, **kwargs) * self.weight


class MultipleLosses(torch.nn.ModuleDict):
    def __init__(self, losses: dict):
        super().__init__(losses)  # loss_name: loss_func -> Callable

    def __call__(self, y_hat, y, **kwargs):
        output = {loss_name: loss_func(y_hat, y, **kwargs) for loss_name, loss_func in self.items()}
        total_loss = sum(output.values()) / len(output)
        return total_loss, output
    
    
class MaskSmoother(torch.nn.Module):
    def __init__(self, radius: int = 10, sigma: float = 0.1, depth: int = 1):
        super().__init__()
        self.radius = radius
        self.depth = depth
        self.sigma = sigma

        self.conv_tp = self._init_tp_conv()

    def _init_tp_conv(self):
        k = self.radius * 2 + 1
        weight = torch.nn.Parameter(torch.zeros((1, 1, k, k)), requires_grad=False)
        for x in range(k):
            for y in range(k):
                weight[:, :, x, y] = np.exp(- self.sigma ** 2 * ((x - self.radius) ** 2 + (y - self.radius) ** 2))
        
        weight = weight / weight.sum()
        
        tp_conv = torch.nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=(k, k), stride=1, padding=self.radius, bias=False)

        tp_conv._parameters['weight'] = weight
        
        return tp_conv
        
    @staticmethod
    def _normalize(X):
        # return X
        min_val = X.min(dim=(-1), keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = X.max(dim=(-1), keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (X - min_val) / (max_val - min_val)

    @torch.no_grad
    def forward(self, X):
        for _ in range(self.depth):
            X = MaskSmoother._normalize(self.conv_tp(X))
        return X