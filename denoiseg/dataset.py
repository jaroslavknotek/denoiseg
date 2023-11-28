import itertools
import warnings

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import denoiseg.image_utils as iu


class DenoisegDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, crop_size, transform, repeat=1):
        self.images = images
        self.labels = labels

        self.transform = transform
        self.crop_size = crop_size
        self.repeat = repeat

    def __len__(self):
        return len(self.images) * self.repeat

    def _augument(self, image, label):
        transformed = self.transform(image=image, mask=label)
        tr_image = transformed["image"]
        tr_label = transformed["mask"]
        return tr_image, tr_label

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()

        idx = idx % len(self.images)
        image = self.images[idx]
        label = self.labels[idx]

        has_label = label is not None
        if not has_label:
            label = np.zeros_like(image)

        image_aug, label_aug = self._augument(image, label)
        noise_x, noise_y, noise_mask = iu.denoise_xy(image_aug)

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
    images, labels, pick_idc, augumentation_fn, patch_size, dataset_repeat, batch_size
):
    def index_list_by_list(_list, indices):
        return [_list[i] for i in list(indices)]

    picked_imgs = index_list_by_list(images, pick_idc)
    picked_gts = index_list_by_list(labels, pick_idc)

    dataset_ = DenoisegDataset(
        picked_imgs, picked_gts, patch_size, augumentation_fn, repeat=dataset_repeat
    )

    return torch.utils.data.DataLoader(dataset_, batch_size=batch_size, shuffle=False)


def prepare_dataloaders(images, ground_truths, config):
    train_idc, val_idc = fair_split_train_val_indices_to_batches(
        ground_truths, config["batch_size"], config["validation_set_percentage"]
    )

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
        config["dataset_repeat"],
        config["batch_size"],
    )

    aug_val = setup_augumentation(config["patch_size"])
    val_dataloader = setup_dataloader(
        images,
        ground_truths,
        val_idc,
        aug_val,
        config["patch_size"],
        config["dataset_repeat"],
        config["batch_size"],
    )

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
    assert training_examples > 0, f"Cannot batch. Not enouch {training_examples=}"
    if training_examples < batch_size:
        warnings.warn(f"Number of {training_examples=} is less than {batch_size=}")

    denoise_idx = np.argwhere(is_denoise).flatten().copy()
    np.random.shuffle(denoise_idx)

    segmantation_idx = np.argwhere(~is_denoise).flatten().copy()
    np.random.shuffle(segmantation_idx)

    batches_num = int(np.ceil(training_examples / batch_size))

    batched_denoise = batched(denoise_idx, batches_num)
    batched_segmentaiton = batched(segmantation_idx, batches_num)

    batches_of_ids = []
    for den, seg in zip(batched_denoise, batched_segmentaiton):
        rest_num = batch_size - len(den) - len(seg)
        rest = _pick_remaining(rest_num, denoise_idx, segmantation_idx)

        batch = np.concatenate([den, seg, rest])
        batches_of_ids.append(batch)
        assert len(batch) == batch_size, f"{len(batch)=} {batch_size=}"
    return np.array(batches_of_ids).astype(int)


def batched(array, n):
    return [array[i::n] for i in range(n)]


def fair_split_train_val_indices_to_batches(labels, batch_size, val_size):
    is_denoise = np.array([(label is None) for label in labels])

    batches = batch_fair(is_denoise, batch_size)

    assert np.unique([len(b) for b in batches]) == [batch_size]
    assert len(np.unique(np.array(batches).flatten())) == len(labels)
    assert batches.shape[0] * batches.shape[1] >= len(labels)

    val_batches_num = int(np.ceil(len(batches) * val_size))
    val_batches = batches[-val_batches_num:]
    train_batches = batches[:-val_batches_num]

    train_idx = np.concatenate(train_batches)
    val_idx = np.concatenate(val_batches)

    return train_idx, val_idx


def _pick_random(n, array):
    rand_idx = np.random.randint(0, high=len(array), size=(n,))
    return array[rand_idx]


def _pick_remaining(rest_num, denoise_idx, segmantation_idx):
    if rest_num > 0 and len(denoise_idx) > 0:
        return _pick_random(rest_num, denoise_idx)
    elif rest_num > 0 and len(segmantation_idx) > 0:
        return _pick_random(rest_num, segmantation_idx)
    else:
        return []


def sample_ds(dataset, n):
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
