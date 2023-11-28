import logging

import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex

import denoiseg.segmentation as seg

logger = logging.getLogger("denoiseg")


def evaluate_images(
    model, imgs, gts, patch_size=128, foreground_thr=0.5, metric=None, device="cpu"
):
    if metric is None:
        m = BinaryJaccardIndex(threshold=foreground_thr)

        def _met(a, b):
            return m(torch.Tensor(a), torch.Tensor(b))

        metric = _met

    preds = seg.segment_many(model, imgs, gts, patch_size, device)
    metrics = []
    for img, gt, pred in zip(imgs, gts, preds):
        _, fg, _, _ = pred
        metric_res = metric(fg, gt)
        metrics.append(metric_res)
    return preds, np.squeeze(metrics)
