import logging

import albumentations as A
import cv2
import numpy as np
import torch

import denoiseg.image_utils as iu

logger = logging.getLogger("segmentation")


class DenoisegDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform, weightmaps=None):
        self.images = images
        self.labels = labels
        self.weightmaps = weightmaps
        assert len(images) == len(labels), f"{len(images)=}!={len(labels)=}"
        assert weightmaps is None or len(labels) == len(
            weightmaps
        ), f"{len(weightmaps)=}!={len(labels)=}"
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def _transform(self, image, label, weightmap):
        cmb = np.stack([label, weightmap])
        transformed = self.transform(image=image, masks=cmb)
        tr_image = transformed["image"]

        tr_cmp = transformed["masks"]
        tr_label, tr_weightmap = tr_cmp
        return tr_image, tr_label, tr_weightmap

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        weightmap = None
        if self.weightmaps is not None and self.weightmaps[idx] is not None:
            weightmap = np.float32(self.weightmaps[idx])
        if weightmap is None:
            weightmap = np.ones_like(image, dtype=np.float32)

        image_aug, label_aug, weightmap_aug = self._transform(image, label, weightmap)

        y = iu.label_to_classes(label_aug)
        x = np.stack([image_aug] * 3)
    
        return {
            "x": x,
            "y": y,
            "weightmap":weightmap_aug[None]
        }

def setup_dataloader(
    images,
    ground_truths,
    pick_idc,
    augumentation_fn,
    patch_size,
    batch_size,
    denoise_enabled=True,
    weightmaps=None,
    shuffle=False,
):
    def index_list_by_list(_list, indices):
        return [_list[i] for i in list(indices)]

    picked_imgs = index_list_by_list(images, pick_idc)
    picked_gts = index_list_by_list(ground_truths, pick_idc)
    picked_wm = None
    if weightmaps is not None:
        picked_wm = index_list_by_list(weightmaps, pick_idc)

    dataset_ = DenoisegDataset(
        picked_imgs, 
        picked_gts, 
        augumentation_fn, 
        weightmaps = picked_wm
    )

    # Shuffle is false because
    # - validation is not shuffled by design
    # - train is shuffled by fair split funciton
    return torch.utils.data.DataLoader(dataset_, batch_size=batch_size, shuffle=shuffle)


def split(arr, split_per):
    if len(arr) == 0:
        logger.warning("Splitting empty array to train/val")
    n = int(len(arr) * split_per)
    if n == 0:
        logger.warning(f"Validation has 0 size. Increase validatio split {split_per=}")
    return arr[n:], arr[:n]


def prepare_dataloaders(images, ground_truths, config, weightmaps=None):
    denoise_enabled = config.get("denoise_enabled", True)
    if not denoise_enabled:
        logger.info("Filtering out denoise images")
        imgs_new = []
        gts_new = []
        ig = [(img, gt) for img, gt in zip(images, ground_truths) if gt is not None]
        for img, gt in ig:
            imgs_new.append(img)
            gts_new.append(gt)
        images = imgs_new
        ground_truths = gts_new

    is_denoise = np.array([gt is None for gt in ground_truths])
    denoise_idx = np.argwhere(is_denoise).flatten()

    denoise_train_idx, denoise_val_idx = split(
        denoise_idx, config["validation_set_percentage"]
    )

    segmantation_idx = np.argwhere(~is_denoise).flatten()

    seg_train_idx, seg_val_idx = split(
        segmantation_idx, config["validation_set_percentage"]
    )

    train_idc = np.concatenate([denoise_train_idx, seg_train_idx])
    val_idc = np.concatenate([denoise_val_idx, seg_val_idx])

    aug_config = config["augumentation"]
    aug_train = setup_augumentation(
        config["patch_size"],
        elastic=aug_config["elastic"],
        brightness_contrast=aug_config["brightness_contrast"],
        flip_vertical=aug_config["flip_vertical"],
        flip_horizontal=aug_config["flip_horizontal"],
        blur_sharp_power=aug_config["blur_sharp_power"],
        noise_value=aug_config["noise_val"],
        rotate_deg=aug_config["rotate_deg"],
    )

    train_dataloader = setup_dataloader(
        images,
        ground_truths,
        train_idc,
        aug_train,
        config["patch_size"],
        config["batch_size"],
        denoise_enabled=denoise_enabled,
        weightmaps=weightmaps,
        shuffle=True,
    )

    aug_val = setup_augumentation(config["patch_size"])
    val_dataloader = setup_dataloader(
        images,
        ground_truths,
        val_idc,
        aug_val,
        config["patch_size"],
        config["batch_size"],
        denoise_enabled=denoise_enabled,
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
    noise_value=None,  # .01
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
                sigma=12,
                alpha_affine=12,
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
    if noise_value is not None:
        transform_list += [
            A.augmentations.transforms.GaussNoise(noise_value, p=.5),
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
