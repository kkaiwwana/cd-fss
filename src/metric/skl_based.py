import torch
from torchmetrics import Metric

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import medial_axis
from scipy.spatial import cKDTree

from multiprocessing import Process, Queue


class SklBasedMetrics(Metric):
    """single class only. 0 for bg and 1 for fg."""
    def __init__(self, radius=5):
        super().__init__()
        self.add_state('tp_pred', [])
        self.add_state('tp_gt', [])
        self.add_state('num_pred', [])
        self.add_state('num_gt', [])
        self.add_state('i', torch.tensor([0.0]))
        self.add_state('u', torch.tensor([1e-8]))

        self.radius = radius
        self.dil_structure = np.ones((2 * radius + 1, 2 * radius + 1))
    
    @staticmethod
    def update_single(out_queue, logits, target, radius, dil_structure) -> None:
        pred_mask = logits.numpy()
        gt_mask = target.numpy()
        gt_skl, pred_skl = medial_axis(gt_mask), medial_axis(pred_mask)
    
        pred_coords = np.column_stack(np.nonzero(pred_skl))
        gt_coords = np.column_stack(np.nonzero(gt_skl))
        
        pred_tree = cKDTree(pred_coords)
        gt_tree = cKDTree(gt_coords)

        pred_match = pred_tree.query_ball_point(gt_coords, r=radius)
        gt_match = gt_tree.query_ball_point(pred_coords, r=radius)

        gt_dil = binary_dilation(gt_skl, structure=dil_structure)
        pred_dil = binary_dilation(pred_skl, structure=dil_structure)

        out = [
            sum([len(p) > 0 for p in pred_match]),
            sum([len(p) > 0 for p in gt_match]),
            len(pred_coords),
            len(gt_coords),
            (pred_dil & gt_dil).sum(), 
            (pred_dil | gt_dil).sum(),
        ]
        out_queue.put(out)
    
    def update(self, logits, target, stage='val', *args, **kwargs):
        # using multi-process to handle batch data. 
        if stage == 'train':
            return
        
        q = Queue()
        processes = []
        logits = logits.argmax(dim=1).cpu().unbind(dim=0)
        target = target.cpu().unbind(dim=0)
        
        for x, y in zip(logits, target):
            p = Process(
                target=SklBasedMetrics.update_single, 
                args=(q, x, y, self.radius, self.dil_structure)
            )
            p.start()
            processes.append(p)
    
        for p in processes:
            p.join()
        
        while not q.empty():
            tp_pred, tp_gt, num_pred, num_gt, i, u = q.get()
            self.tp_pred.append(tp_pred)
            self.tp_gt.append(tp_gt)
            self.num_pred.append(num_pred)
            self.num_gt.append(num_gt)
            self.i += i
            self.u += u
        
    def compute(self, *args, **kwargs):
        if sum(self.num_pred) == 0:
            return None
        
        precision = sum(self.tp_pred) / sum(self.num_pred)
        recall = sum(self.tp_gt) / sum(self.num_gt)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        iou = self.i / self.u
        
        return dict(
            iou_skl = iou,
            precision_skl = precision,
            recall_skl = recall,
            f1_score_skl = f1_score,
        )