import copy


# TODO: to json
def get_default_config():
    return {
        "patch_size": 128,
        "validation_set_percentage": 0.2,
        "batch_size": 32,
        "dataset_repeat": 50,
        "model": {
            "filters": 8,
            "depth": 5,
        },
        "loss_function": "fl",
        # training
        "epochs": 200,
        "patience": 20,
        "scheduler_patience": 10,
        "denoise_enabled": True,
        "denoise_loss_weight": 0.01,  # relative weight of denoise loss weight. should be betwee 0-1
        "augumentation": {
            "elastic": True,
            "brightness_contrast": True,
            "flip_vertical": True,
            "flip_horizontal": True,
            "blur_sharp_power": 1,
            "noise_val": 0.01,
            "rotate_deg": 90,
        },
    }


def merge(base_dict, update_dict):
    new = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if isinstance(value, dict):
            node = base_dict.get(key, {})
            new[key] = merge(node, value)
        else:
            new[key] = value

    return new
