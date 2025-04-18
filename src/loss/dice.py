import torch
import torch.nn.functional as F
class DiceLoss(torch.nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        """inputs: (b, class, h, w), targets (b, h, w)
        """
        inputs = F.sigmoid(inputs)       
        intersection = (inputs * targets[:, None]).sum()                   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice