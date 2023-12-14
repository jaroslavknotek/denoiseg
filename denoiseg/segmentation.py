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
    weightmaps = None,
    model = None,
    device = 'cpu'
):
    
    model_depth = train_params['model']['depth']
    patch_size = train_params['patch_size']
    if np.log2(patch_size) <  model_depth+2:
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
        train_params,
        weightmaps=weightmaps
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
    device='cpu'
):
    return [
        segment_image(model, img, device=device)
        for img in tqdm(imgs, desc="Segmenting", total=len(imgs))
    ]


def segment_image(model, img, device="cpu",pad_stride = 32):
    img = _ensure_2d(img)

    with torch.no_grad():
        img_3d = np.stack([img]*3)
        tensor = torch.from_numpy(img_3d).to(device)[None]
        padded_tensor,pads = pad_to(tensor, pad_stride)
        res_tensor = model(padded_tensor)
        res_unp = unpad(res_tensor,pads) 
        return np.squeeze(res_unp.cpu().detach().numpy())

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
import torch.nn.functional as F

def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x