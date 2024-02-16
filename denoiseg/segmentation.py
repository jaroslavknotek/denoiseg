import json
import logging
from datetime import datetime

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import denoiseg.dataset as ds
import denoiseg.training as training
import denoiseg.utils as utils

logger = logging.getLogger("denoiseg")


def run_training(
    images,
    ground_truths,
    train_params,
    training_output_dir,
    weightmaps=None,
    model=None,
    device="cpu",
    train_dataloader=None,
    val_dataloader=None,
    loss_fn=None,
):
    model_depth = train_params["model"]["depth"]
    patch_size = train_params["patch_size"]
    if np.log2(patch_size) < model_depth + 2:
        raise Exception(f"Cannot have {patch_size=} and {model_depth=}")

    checkpoint_path, log_path = _setup_paths_from_root(training_output_dir)
    logger = utils.setup_logger(path=log_path)

    with open(checkpoint_path.parent / "training_params.json", "w") as f:
        json.dump(train_params, f)

    if train_dataloader is None and val_dataloader is None:
        train_dataloader, val_dataloader = ds.prepare_dataloaders(
            images, ground_truths, train_params, weightmaps=weightmaps
        )
    assert train_dataloader is not None, val_dataloader is not None

    if model is None:
        model = _create_default_model(train_params["model"])

    if loss_fn is None:
        loss_fn = training.get_loss(
            train_params["loss_function"],
            device=device,
            denoise_loss_weight=train_params.get("denoise_loss_weight", 0),
            denoise_enabled=train_params.get("denoise_enabled", False),
        )

    logger.info("Training started")
    losses = training.train(
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        epochs=train_params["epochs"],
        patience=train_params["patience"],
        scheduler_patience=train_params["scheduler_patience"],
        checkpoint_path=checkpoint_path,
        device=device,
    )

    torch.save(model, checkpoint_path.parent / "model-final.pth")

    return checkpoint_path, losses


def segment_many(model, imgs, device="cpu"):
    return [
        segment_image(model, img, device=device)
        for img in tqdm(imgs, desc="Segmenting", total=len(imgs))
    ]


def segment_image(model, img, device="cpu", pad_stride=32):
    img = _ensure_2d(img)

    with torch.no_grad():
        img_3d = np.stack([img] * 3)
        tensor = torch.from_numpy(img_3d).to(device)[None]
        padded_tensor, pads = pad_to(tensor, pad_stride)
        res_tensor = model(padded_tensor)
        res_unp = unpad(res_tensor, pads)
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
    ts = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    training_output_root_timestamped = training_output_root / f"{ts}"
    training_output_root_timestamped.mkdir(exist_ok=True, parents=True)

    log_path = training_output_root_timestamped / "logs.txt"
    checkpoint_path = training_output_root_timestamped / "model-checkpoint-best.pth"

    return checkpoint_path, log_path


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
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2] : -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0] : -pad[1]]
    return x


def _create_default_model(
    model_params,
    encoder="resnet18",
    activation="sigmoid",
    decoder_attention_type=None,  # "scse"
):
    logger.info(f"Using default unet model with {model_params=}")
    depth = model_params["depth"]
    channels = np.flip(model_params["filters"] * (2 ** np.arange(depth)))
    return smp.Unet(
        encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        decoder_channels=channels,
        encoder_depth=depth,
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
        decoder_attention_type=decoder_attention_type,
        activation=activation,
    )
