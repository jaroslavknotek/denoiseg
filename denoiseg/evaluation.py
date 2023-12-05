import logging

import numpy as np
import torch
import denoiseg.metrics as m
import denoiseg.segmentation as seg

logger = logging.getLogger("denoiseg")


def evaluate_images(
    model, 
    imgs, 
    gts, 
    patch_size=128, 
    foreground_thr=0.5,
    patch_overlap = 0.5,
    metric=None, 
    device="cpu"
):
    if metric is None:
        logger.info("Using default IOU metric")
        metric = m.prepare_iou(foreground_thr = foreground_thr)

    preds = seg.segment_many(
        model, 
        imgs, 
        gts,
        patch_size,
        patch_overlap = patch_overlap,
        device=device
    )
    
    metrics = []
    for gt, pred in zip(gts, preds):
        _, fg, _, _ = pred
        metric_res = metric(gt,fg)
        metrics.append(metric_res)
    return preds, np.squeeze(metrics)

    