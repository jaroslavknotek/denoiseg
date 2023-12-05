import itertools
import matplotlib.pyplot as plt
import numpy as np
import denoiseg.dataset as ds

def plot_row(imgs, vmin_vmax=None, cell_size=5, figsize = None):
    if figsize is None:
        figsize = (cell_size * len(imgs), cell_size)
        
    _, axs = plt.subplots(
        1, 
        len(imgs), 
        figsize=figsize
    )

    for ax, im in zip(axs, imgs):
        if vmin_vmax is not None:
            ax.imshow(im, vmin=vmin_vmax[0], vmax=vmin_vmax[1],cmap='gray')
        else:
            ax.imshow(im)


def sample_ds(images,ground_truths,training_params,  n = 5):
    
    aug_config = training_params['augumentation']
    sample_aug  = ds.setup_augumentation(
        training_params["patch_size"],
        elastic=aug_config["elastic"],
        brightness_contrast=aug_config["brightness_contrast"],
        flip_vertical=aug_config["flip_vertical"],
        flip_horizontal=aug_config["flip_horizontal"],
        blur_sharp_power=aug_config["blur_sharp_power"],
        noise_val=aug_config["noise_val"],
        rotate_deg=aug_config["rotate_deg"],
    )
    
    dataset = ds.DenoisegDataset(
        images, 
        ground_truths, 
        sample_aug,
        training_params.get('denoise_enabled',True)
    )
    
    tts = (t for t in dataset if np.sum(t["has_label"]) > 0)

    for i, t in enumerate(itertools.islice(tts, 0, 15)):
        if i >= n:
            break
        img = t["x"][0]
        mask = t["y_segmentation"][0]
        wm = np.squeeze(t["mask_denoise"])
        imgs = [img, mask, wm]

        _, axs = plt.subplots(1, len(imgs), figsize=(len(imgs) * 5, 5))

        for ax, im in zip(axs, imgs):
            ax.imshow(im, vmin=0, vmax=np.max(im), cmap="gray")
