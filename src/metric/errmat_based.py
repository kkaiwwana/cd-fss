import torch
from torchmetrics import Metric


class ErrMatBasedMetrics(Metric):
    """single class only. 0 for bg and 1 for fg."""
    def __init__(self):
        super().__init__()
        self.add_state('TP', torch.tensor([0]))
        self.add_state('FP', torch.tensor([0]))
        self.add_state('FN', torch.tensor([0]))
        self.add_state('TN', torch.tensor([0]))
    
    def update(self, logits, target, *args, **kwargs):
        pred_mask = logits.argmax(dim=1)
        self.TP += (pred_mask & target).sum()
        self.FP += (pred_mask & (1 - target)).sum()
        self.FN += ((1 - pred_mask) & target).sum()
        self.TN += ((1 - pred_mask) & (1 - target)).sum()
    
    def compute(self, *args, **kwargs):
        eps = 10e-5
        return dict(
            precision = self.TP / ((self.TP + self.FP) + eps),
            recall = self.TP / ((self.TP + self.FN) + eps),
            accuracy = (self.TP + self.TN) / ((self.TP + self.TN + self.FP + self.FN) + eps),
            f1_score = 2 * self.TP / ((2 * self.TP + self.FN + self.FP) + eps),
        )