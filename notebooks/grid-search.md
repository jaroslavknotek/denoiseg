---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: torch_cv
    language: python
    name: torch_cv
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

```python
import pathlib

from datetime import datetime

ts = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
training_output_root = pathlib.Path(f'../../training/denoiseg-UHCS/grid_search_{ts}')
```

```python
import imageio
import matplotlib.pyplot as plt
import pathlib
import denoiseg.image_utils as iu

root = pathlib.Path('../data/UHCS/')
gt_root = root/'GT_bin'
img_root = root/'labeled_img'
unlabeled_root = root/'unlabeled_img'

gt_paths = list(sorted(gt_root.glob('*.png')))
img_paths = [ img_root/f"{p.stem}.jpg" for p in gt_paths]
unlabeled_paths = list(unlabeled_root.glob('*.jpg'))

imgs_unlbl = list(map(iu.load_image, unlabeled_paths))
gts_unlbl = [None]*len(imgs_unlbl)

imgs_lbl = list(map(iu.load_image, img_paths))
gts_lbl = list(map(iu.load_image, gt_paths))

imgs_tests = imgs_lbl[:4]
gts_tests = gts_lbl[:4]

gts_lbl = gts_lbl[4:]
imgs_lbl = imgs_lbl[4:]

imgs = imgs_lbl #+ imgs_unlbl
gts = gts_lbl #+ gts_unlbl
```

```python
import numpy as np
import denoiseg.configuration as cfg
import denoiseg.visualization as vis

default_params = cfg.get_default_config()

vis.sample_ds(
    imgs,
    gts,
    default_params,
    n=1
)
```

```python
custom_base = {
    "epochs":500,
    "patience":20,
    "dataset_repeat":100,
    "denoise_loss_weight":0.001,
    "validation_set_percentage":.2,
    "note":"no denoising",
    "denoise_enabled":False
}

overrides = [
#     {"patch_size":64,"model":{"filters":8,"depth":4 }},
#     {"patch_size":64,"model":{"filters":16,"depth":4 }},
    
#     {"patch_size":128,"model":{"filters":8,"depth":4 }},
    {"patch_size":128,"model":{"filters":8,"depth":5 }},
    # {"patch_size":128,"model":{"filters":16,"depth":5 }},
    
    
#     {"patch_size":256,"model":{"filters":8,"depth":4 }},
#     {"patch_size":256,"model":{"filters":8,"depth":5 }},
#     {"patch_size":256,"model":{"filters":8,"depth":6 }},
    
#     {"patch_size":256,"model":{"filters":16,"depth":4 }},
#     {"patch_size":256,"model":{"filters":16,"depth":5 }},
#     {"patch_size":256,"model":{"filters":16,"depth":6 }},
]

customs = [ cfg.merge(custom_base,o) for o in overrides]
```

```python
import denoiseg.visualization as vis
import denoiseg.image_utils as iu
import denoiseg.evaluation as ev

import denoiseg.visualization as vis
import denoiseg.image_utils as iu
import denoiseg.evaluation as ev


jq = ev.prepare_iou(foreground_thr=.5)
def normed_iou(target,prediction):
    norm_pred = iu.norm(prediction)
    return jq(target,norm_pred)
    
```

# Training

```python
import pandas as pd
import denoiseg.segmentation as seg


import denoiseg.training as tr
from tqdm.auto import tqdm
import denoiseg.evaluation as ev
import denoiseg.visualization as vis
import denoiseg.param_search as ps

param_results = ps.search_train(
    training_output_root,
    imgs,
    gts,
    imgs_tests,
    gts_tests,
    customs,
    evaluation_metric_fn = normed_iou,
    device = device,
)
```

```python
param_results
```

```python
import pandas
import json


metrics_paths = [checkpoint.parent/'metrics.csv' for checkpoint,_,_ in param_results]
params_paths = [p.parent/'training_params.json' for p in metrics_paths]
metrics = [pd.read_csv(p) for p in metrics_paths]
mean_metrics = [m['metric'].mean() for m in metrics]
params = [ps.flatten_dict(json.load(open(p))) for p in params_paths]
```

```python
df = pd.DataFrame(params)
df['metric_mean'] = mean_metrics

max_metric = np.argmax(df['metric_mean'])
metrics_paths[max_metric],df.iloc[max_metric]
```

```python

```
