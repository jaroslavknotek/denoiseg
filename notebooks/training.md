---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: denoiseg
    language: python
    name: denoiseg
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import denoiseg.utils as utils
try:
    print(logger)
except NameError:
    logger  = utils.setup_logger(path = 'logs.txt')
```

```python
import sys
sys.path.insert(0,'..')
```

```python
import torch
import numpy as np
import pathlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

```python
%pwd
```

```python
import imageio
import matplotlib.pyplot as plt

root = pathlib.Path('../data/UHCS/')
gt_root = root/'GT_bin'
img_root = root/'labeled_img'
unlabeled_root = root/'unlabeled_img'

gt_paths = list(gt_root.glob('*.png'))
img_paths = [ img_root/f"{p.stem}.jpg" for p in gt_paths]
unlabeled_paths = list(unlabeled_root.glob('*.jpg'))

```

```python
import denoiseg.image_utils as iu

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

for img,gt in zip(imgs,gts):
    _,axs = plt.subplots(1,2)
    axs[0].imshow(img)
    if gt is None:
        gt = np.zeros_like(img)
    axs[1].imshow(gt)
    plt.show()
    break

patch_size = 128
```

```python
import cv2


import itertools
import denoiseg

import denoiseg.dataset as ds

aug_train = ds.setup_augumentation(
    patch_size,
    elastic = True,
    brightness_contrast = True,
    flip_vertical = True,
    flip_horizontal = True,
    blur_sharp_power = 1,
    noise_val = .01,
    rotate_deg = 90
)
dataset = ds.DenoisegDataset(
    imgs,
    gts,
    patch_size,
    aug_train,
    repeat = 1
)
ds.sample_ds(dataset,3)
```

```python
train_params = {
    "patch_size":128,
    "validation_set_percentage":.2,
    "batch_size":32,
    "dataset_repeat":50,
    "model":{
        "filters":8,
        "depth":5,
    },
    #training
    "epochs":100,
    "patience":20,
    "scheduler_patience":10,
    "denoise_loss_weight":1,
    
    "augumentation":{
        "elastic":True,
        "brightness_contrast":True,
        "flip_vertical": True,
        "flip_horizontal": True,
        "blur_sharp_power": 1,
        "noise_val": .01,
        "rotate_deg": 90
    }
}
model_depth = train_params['model']['depth']
patch_size = train_params['patch_size']
if np.log2(patch_size) < model_depth +2:
    logger.warn(
        f"Cannot have crop_size={patch_size} and cnn_depth={model_depth}"
    )
# train_params['dataset_repeat'] = 1
# train_params['epochs'] = 5
# train_params['batch_size'] = 32
# train_params['model']['depth'] = 5
```

```python
train_dataloader,val_dataloader = ds.prepare_dataloaders(imgs,gts,train_params)
```

# Training

```python
import denoiseg.unet
import denoiseg.training

model_config = train_params['model']
model = denoiseg.unet.UNet(
    start_filters=model_config['filters'], 
    depth=model_config['depth'], 
    in_channels=3,
    out_channels=4
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
device = None
checkpoint_path = pathlib.Path('training')/'model-best.pth'
checkpoint_path.parent.mkdir(exist_ok=True,parents=True)

loss_fn = denoiseg.training.get_loss(
    device = device,
    denoise_loss_weight = train_params['denoise_loss_weight']
)
out_losses = denoiseg.training.train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    epochs = train_params['epochs'],
    patience = train_params['patience'],
    scheduler_patience = train_params['scheduler_patience'],
    checkpoint_path = checkpoint_path,
    device = device
)   
```

```python
plt.plot(out_losses['train_loss'],label='train')
plt.plot(out_losses['val_loss'],label='val')
plt.legend()
```

```python
import denoiseg.segmentation as seg
import denoiseg.training as tr
from tqdm.auto import tqdm

import denoiseg.evaluation as ev

predictions, metrics = ev.evaluate_images(model, imgs_test,gts_test)

print('Mean IoU',np.mean(metrics))
```

```python
import denoiseg.visualization as vis

for img,gt,pred,met in zip(imgs_test,gts_test,predictions,metrics):
    show_imgs = [img,gt,*pred]
    vis.plot_row(show_imgs,vmin_vmax = (0,1))
    plt.suptitle(f'Metric: {met}')
    plt.show()
```

```python

```
