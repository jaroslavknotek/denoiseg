import itertools
import warnings

import albumentations as A
import cv2
import numpy as np
import torch

import denoiseg.image_utils as iu
import logging

logger = logging.getLogger('denoiseg')


class DenoisegDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform, denoise = True):
        self.images = images
        self.labels = labels

        self.transform = transform
        self.denoise = denoise

    def __len__(self):
        return len(self.images)

    def _augument(self, image, label):
        transformed = self.transform(image=image, mask=label)
        tr_image = transformed["image"]
        tr_label = transformed["mask"]
        return tr_image, tr_label

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        has_label = label is not None
        if not has_label:
            label = np.zeros_like(image)

        image_aug, label_aug = self._augument(image, label)
        if self.denoise:
            noise_x, noise_y, noise_mask = iu.denoise_xy(image_aug)
        else:
            noise_x = image_aug[None]
            noise_y = image_aug[None]
            noise_mask = np.zeros_like(image_aug)
        
        y = iu.label_to_classes(label_aug)
        x = np.concatenate([noise_x] * 3, axis=0)
        has_label = np.expand_dims(np.array([has_label]), axis=(1, 2, 3))
    
        return {
            "x": x,
            "y_denoise": noise_y,
            "mask_denoise": noise_mask[None, ...],
            "y_segmentation": y,
            "has_label": has_label,
        }


def setup_dataloader(
    images, 
    ground_truths, 
    pick_idc, 
    augumentation_fn, 
    patch_size, 
    batch_size,
    denoise_enabled = True
):
    def index_list_by_list(_list, indices):
        return [_list[i] for i in list(indices)]
    
    picked_imgs = index_list_by_list(images, pick_idc)
    picked_gts = index_list_by_list(ground_truths, pick_idc)
    
    dataset_ = DenoisegDataset(
        picked_imgs, 
        picked_gts, 
        augumentation_fn, 
        denoise = denoise_enabled
    )

    # Shuffle is false because
    # - validation is not shuffled by design
    # - train is shuffled by fair split funciton
    return torch.utils.data.DataLoader(
        dataset_, 
        batch_size=batch_size, 
        shuffle=False
    )


def prepare_dataloaders(images, ground_truths, config):
    
    denoise_enabled = config.get("denoise_enabled",True)
    if not denoise_enabled:
        logger.info("Filtering out denoise images")
        imgs_new = []
        gts_new = []
        ig = [(img,gt) for img,gt in zip(images,ground_truths) if gt is not None]
        for img,gt in ig:
            imgs_new.append(img)
            gts_new.append(gt)
        images =imgs_new
        ground_truths = gts_new
    
    repeat = config["dataset_repeat"]
    before_size = len(ground_truths)
    
    train_idc, val_idc = fair_split_train_val_indices_to_batches(
        list(ground_truths)*repeat, 
        config["batch_size"], 
        config["validation_set_percentage"]
    )
    
    train_idc = np.mod(train_idc,before_size)
    val_idc = np.mod(val_idc,before_size)
    
    aug_config = config["augumentation"]
    aug_train = setup_augumentation(
        config["patch_size"],
        elastic=aug_config["elastic"],
        brightness_contrast=aug_config["brightness_contrast"],
        flip_vertical=aug_config["flip_vertical"],
        flip_horizontal=aug_config["flip_horizontal"],
        blur_sharp_power=aug_config["blur_sharp_power"],
        noise_val=aug_config["noise_val"],
        rotate_deg=aug_config["rotate_deg"],
    )
    
    train_dataloader = setup_dataloader(
        images,
        ground_truths,
        train_idc,
        aug_train,
        config["patch_size"],
        config["batch_size"],
        denoise_enabled
    )

    aug_val = setup_augumentation(config["patch_size"])
    val_dataloader = setup_dataloader(
        images,
        ground_truths,
        val_idc,
        aug_val,
        config["patch_size"],
        config["batch_size"],
        denoise_enabled
    )
    
    logger.info(f"Batches:{len(train_dataloader)=}")
    logger.info(f"Batches:{len(val_dataloader)=}")

    return train_dataloader, val_dataloader


def setup_augumentation(
    patch_size,
    elastic=False,  # True
    brightness_contrast=False,
    flip_vertical=False,
    flip_horizontal=False,
    blur_sharp_power=None,  # 1
    noise_val=None,  # .01
    rotate_deg=None,  # 90
    interpolation=cv2.INTER_CUBIC,
):
    patch_size_padded = int(patch_size * 1.5)
    transform_list = [
        A.PadIfNeeded(patch_size_padded, patch_size_padded),
        A.RandomCrop(patch_size_padded, patch_size_padded),
    ]

    if elastic:
        transform_list += [
            A.ElasticTransform(
                p=0.5,
                alpha=10,
                sigma=120 * 0.1,
                alpha_affine=120 * 0.1,
                interpolation=interpolation,
            )
        ]
    if rotate_deg is not None:
        transform_list += [
            A.Rotate(limit=rotate_deg, interpolation=interpolation),
        ]

    if brightness_contrast:
        transform_list += [
            A.RandomBrightnessContrast(p=0.5),
        ]
    if noise_val is not None:
        transform_list += [
            A.augmentations.transforms.GaussNoise(noise_val, p=1),
        ]

    if blur_sharp_power is not None:
        transform_list += [
            A.OneOf(
                [
                    A.Sharpen(p=1, alpha=(0.2, 0.2 * blur_sharp_power)),
                    A.Blur(blur_limit=3 * blur_sharp_power, p=1),
                ],
                p=0.3,
            ),
        ]

    if flip_horizontal:
        transform_list += [
            A.HorizontalFlip(p=0.5),
        ]
    if flip_vertical:
        transform_list += [
            A.VerticalFlip(p=0.5),
        ]

    transform_list += [A.CenterCrop(patch_size, patch_size)]
    return A.Compose(transform_list)


def batch_fair(is_denoise, batch_size):
    
    training_examples = len(is_denoise)
    assert training_examples > 0, f"Cannot batch. Not enough {training_examples=}"
    if training_examples < batch_size:
        warnings.warn(f"Number of {training_examples=} is less than {batch_size=}")

    denoise_idx = list(np.argwhere(is_denoise).flatten())
    segmantation_idx = list(np.argwhere(~is_denoise).flatten())

    too_much = len(is_denoise)%batch_size
    missing= (batch_size - too_much)%batch_size
        
    total_batchable = len(is_denoise)+missing
    assert total_batchable % batch_size == 0,"It's not batchable"
    
    batch_num = int(total_batchable/batch_size)

    ratio = len(denoise_idx)/len(is_denoise)
    add_den = int(missing*ratio) 
    add_seg = missing - add_den

    denoise_idx.extend(denoise_idx[:add_den])
    segmantation_idx.extend(segmantation_idx[:add_seg])

    np.random.shuffle(denoise_idx)
    np.random.shuffle(segmantation_idx)

    batches = []
    for _ in range(batch_num):
        ratio = len(denoise_idx)/(len(segmantation_idx)+len(denoise_idx))
        take_den = int(ratio*(batch_size))    
        batch = []
        for i in range(take_den):
            if len(denoise_idx)>0:
                item = denoise_idx.pop()
            batch.append(item)

        take_seg = batch_size - len(batch)

        segs = [segmantation_idx.pop() for _ in range(take_seg)]
        batch.extend(segs)

        assert len(batch) == batch_size
        np.random.shuffle(batch)
        batches.append(batch)

    return np.array(batches)

def fair_split_train_val_indices_to_batches(labels, batch_size, val_size):
    
    is_nones = [(label is None) for label in labels]
    is_denoise = np.array(is_nones)
    
    batches = batch_fair(is_denoise, batch_size)

    assert np.unique([len(b) for b in batches]) == [batch_size]
    #assert len(np.unique(np.array(batches).flatten())) == len(labels)
    assert batches.shape[0] * batches.shape[1] >= len(labels)

    val_batches_num = int(np.ceil(len(batches) * val_size))
    val_batches = batches[-val_batches_num:]
    train_batches = batches[:-val_batches_num]

    train_idx = np.concatenate(train_batches)
    val_idx = np.concatenate(val_batches)

    return train_idx, val_idx
