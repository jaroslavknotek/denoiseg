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

training_output_root = pathlib.Path(f'../../training/denoiseg-UHCS/')
```

```python
from tqdm.auto import tqdm
import imageio
import matplotlib.pyplot as plt
import pathlib
import denoiseg.image_utils as iu
import denoiseg.training as tr

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
gts_lbl = list(map(iu.load_mask, gt_paths))

imgs_test = imgs_lbl[:4]
gts_test = gts_lbl[:4]

gts_lbl = gts_lbl[4:]
imgs_lbl = imgs_lbl[4:]

imgs = imgs_lbl + imgs_unlbl
gts = gts_lbl + gts_unlbl

total = len([_ for g in gts if g is not None])
weightmaps =[(tr.unet_weight_map(gt) if gt is not None else None) for gt in tqdm(gts,desc='generating weightmaps',total = total)]

print(f"{len(imgs_lbl)=} {len(imgs_test)=} {len(imgs_unlbl)=} {len(weightmaps)=}")
```

```python
import numpy as np
import denoiseg.configuration as cfg
default_params = cfg.get_default_config()

custom = {
    "patch_size":256,
    "epochs":200,
    "patience":20,
    "dataset_repeat":100,
    #"denoise_loss_weight":0.01,
    "denoise_enabled":False,
    "validation_set_percentage":.2,
    "model":{
        "filters":32,
        "depth":5,
    },
    "note":"added filters",
    "augumentation":{
        "brightness_contrast":True,
        "noise_val":0.001,
        "flip_vertical": True,
        "flip_horizontal": True,
        "rotate_deg": 190,
        "elastic":True,
        "blur_sharp_power": 1,
    }
}

train_params = cfg.merge(default_params,custom)
```

```python
import denoiseg.visualization as vis

vis.sample_ds(
    imgs,
    gts,
    train_params,
    weightmaps= weightmaps
)
```

# Training

```python
import segmentation_models_pytorch as smp

encoder = "resnet18"
decoder_attention_type = None #"scse"
activation = 'sigmoid'

model = smp.Unet(
    encoder_name=encoder,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,                      # model output channels (number of classes in your dataset)
    decoder_attention_type=decoder_attention_type,
    activation=activation
)
```

```python
import denoiseg.segmentation as seg
from tqdm.contrib.logging import logging_redirect_tqdm

with logging_redirect_tqdm():
    checkpoint, out_losses =seg.run_training(
        imgs,
        gts,
        train_params, 
        training_output_root,
        device = device,
        #weightmaps = weightmaps,
        model = model
    )
```

```python
import denoiseg.visualization as vis

vis.plot_loss(out_losses)
plt.show()
```

# Evaluate

```python
import denoiseg.training as tr
from tqdm.auto import tqdm
import denoiseg.evaluation as ev

model = torch.load(checkpoint)
predictions, metrics = ev.evaluate_images(
    model, 
    imgs_test,
    gts_test,
    device = device
)

print(f'Mean IoU {np.mean(metrics):.5f}')
```

```python

for i,(img,gt,pred,met) in enumerate(zip(imgs_test,gts_test,predictions,metrics)):
    
    segms = []
    for im in pred[1:]:
        im = im.copy()
        im = (im - np.min(im))/(np.max(im)-np.min(im))
        im[im<.5] = 0
        im[im>=.5] = 1
        segms.append(im)

    show_imgs = [img,gt,*segms,*pred]
    vis.plot_row(show_imgs,vmin_vmax = (0,1),figsize=(50,10))
    plt.suptitle(f'Metric: {met}',y=0.72)
    plt.tight_layout()
    plt.show()
```

```python

```

```python

```

```python

```

```python

```
