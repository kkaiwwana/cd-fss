from torchmetrics import Metric


class mIoU(Metric):
    """single class only, and background label is 0."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('iou_list', [])
        
    def update(self, logits, targets, *args, **kwargs):
        eps = 10e-4
        logits = logits.argmax(dim=1)
        i, u = (logits & targets).sum(dim=(-1, -2)), (logits | targets).sum(dim=(-1, -2))
        iou = list((i / (u + eps)).unbind(dim=0))
        self.iou_list.extend(iou)
        
    def compute(self, *args, **kwargs):
        return sum(self.iou_list) / len(self.iou_list)