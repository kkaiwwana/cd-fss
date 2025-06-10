import torch
from torchmetrics import Metric


class mIoU(Metric):
    """single class only, and background label is 0."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('intersection', torch.tensor([0.0]))
        self.add_state('union', torch.tensor([1e-8]))
        
    def update(self, logits, targets, *args, **kwargs):
        logits = logits.argmax(dim=1)
        i, u = (logits & targets).sum(), (logits | targets).sum()
        self.intersection += i
        self.union += u
        
    def compute(self, *args, **kwargs):
        return self.intersection / self.union


# class mIoU(Metric):
#     """single class only, and background label is 0."""
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.add_state('iou_list', [])
        
#     def update(self, logits, targets, *args, **kwargs):
#         eps = 10e-4
#         logits = logits.argmax(dim=1)
#         i, u = (logits & targets).sum(dim=(-1, -2)), (logits | targets).sum(dim=(-1, -2))
#         iou = list((i / (u + eps)).unbind(dim=0))
#         self.iou_list.extend(iou)
        
#     def compute(self, *args, **kwargs):
#         return sum(self.iou_list) / len(self.iou_list)