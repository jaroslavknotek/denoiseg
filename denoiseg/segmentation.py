import empatches
import numpy as np
import torch
from tqdm.auto import tqdm


import denoiseg.dataset as ds
import denoiseg.unet as unet
import denoiseg.training as training
from datetime import datetime
import denoiseg.utils as utils

import json

def run_training(
    images,
    ground_truths,
    train_params,
    training_output_dir, 
    model = None,
    device = 'cpu'
):
    
    if np.log2(train_params['patch_size']) < train_params['model']['depth'] +2:
        raise Exception(
            f"Cannot have {patch_size=} and {model_depth=}"
        )
    
    checkpoint_path,log_path = _setup_paths_from_root(training_output_dir)  
    logger  = utils.setup_logger(path = log_path)
    
    with open(checkpoint_path.parent/'training_params.json','w') as f:
        json.dump(train_params,f)

    train_dataloader,val_dataloader = ds.prepare_dataloaders(
        images,
        ground_truths,
        train_params
    )
    if model is None:
        model_params = train_params['model']
        logger.info(f"Using default unet model with {model_params=}")
        model = unet.UNet(
            start_filters=model_params['filters'], 
            depth=model_params['depth'], 
            in_channels=3,
            out_channels=4
        )

    loss_fn = training.get_loss(
        train_params['loss_function'],
        device = device,
        denoise_loss_weight = train_params.get('denoise_loss_weight',0),
        denoise_enabled = train_params.get('denoise_enabled',False)
    )
    
    logger.info("Training started")
    losses = training.train(
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
    
    torch.save(model, checkpoint_path.parent/"model-final.pth")
    
    return checkpoint_path,losses

def segment_many(
    model, 
    imgs,
    gts,
    patch_size,
    patch_overlap = .5, 
    device='cpu'
):
    return [
        segment_image(model, img, patch_size, patch_overlap, device=device)
        for img, gt in tqdm(zip(imgs, gts), desc="Segmenting", total=len(imgs))
    ]


def segment_image(model, img, patch_size=128, patch_overlap=0.75, device="cpu"):
    img = _ensure_2d(img)

    emp = empatches.EMPatches()
    img_patches, indices = emp.extract_patches(
        img, patchsize=patch_size, overlap=patch_overlap
    )

    patches_3ch = np.stack([img_patches] * 3, axis=1)
    with torch.no_grad():
        patches_tensor = torch.from_numpy(patches_3ch).to(device)
        patches_pred = model(patches_tensor)
        patches_predictions = np.squeeze(patches_pred.cpu().detach().numpy())

    layer_idxs = patches_predictions.shape[1]
    layers = []
    for i in range(layer_idxs):
        foreground_patches = patches_predictions[:, i]
        layer = emp.merge_patches(foreground_patches, indices, mode="avg")
        layers.append(layer)
    return layers


def _ensure_2d(img, ensure_float=True):
    match img.shape:
        case (_, _):
            img_2d = img
        case (_, _, _):
            img_2d = img[:, :, 0]
        case _:
            raise ValueError("Unexpected img shape")

    if ensure_float:
        max_val = np.max(img_2d)
        if max_val <= 1:
            return np.float32(img_2d)
        elif max_val <= 255:
            return np.float32(img_2d) / 255
        else:
            assumed_type_max = np.ceil(np.log2(max_val))
            return np.float32(img_2d) / assumed_type_max
    else:
        return img_2d

    
def _setup_paths_from_root(training_output_root):
    ts = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    training_output_root_timestamped = training_output_root/f'{ts}'
    training_output_root_timestamped.mkdir(exist_ok=True,parents=True)
    
    log_path = training_output_root_timestamped/'logs.txt'
    checkpoint_path = training_output_root_timestamped/'model-checkpoint-best.pth'
    
    return checkpoint_path,log_path
