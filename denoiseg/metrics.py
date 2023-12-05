import torch
from torchmetrics.classification import BinaryJaccardIndex

def prepare_iou(foreground_thr = .5):
    m = BinaryJaccardIndex(threshold=foreground_thr)

    def met(a, b):
        a = a.copy()
        b = b.copy()
        a[a<foreground_thr] = 0
        a[a>=foreground_thr] = 1
        b[b<foreground_thr] =0
        b[b>=foreground_thr] = 1
        return m(torch.Tensor(a),torch.Tensor(b))
    
    return met