import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex


def prepare_iou(foreground_thr=0.5):
    m = BinaryJaccardIndex(threshold=foreground_thr)

    def met(a, b):
        a = np.where(a < foreground_thr, 0, 1)
        b = np.where(b < foreground_thr, 0, 1)
        return m(torch.Tensor(a), torch.Tensor(b))

    return met
