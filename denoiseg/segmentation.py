import empatches
import numpy as np
import torch
from tqdm.auto import tqdm


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


def segment_many(model, imgs, gts, patch_size, device):
    return [
        segment_image(model, img, patch_size, device=device)
        for img, gt in tqdm(zip(imgs, gts), desc="Segmenting", total=len(imgs))
    ]


def segment_image(model, img, patch_size=128, patch_overlap=0.5, device="cpu"):
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
