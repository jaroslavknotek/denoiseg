from tqdm.auto import tqdm
import json
import pandas as pd
import denoiseg.segmentation as seg
import denoiseg.training as tr
import denoiseg.evaluation as ev
import denoiseg.configuration as cfg
import torch
import denoiseg.visualization as vis
import matplotlib.pyplot as plt

import logging
logging.basicConfig()
logger = logging.getLogger('denoiseg')
logger.setLevel(logging.DEBUG)


def pretty_print_dict(obj_dict,indent=  2):
    return json.dumps(obj_dict, indent=indent)

def flatten_dict(dict_):
    queue = list(dict_.items())
    key_vals = []

    while len(queue)>0:
        key,val = queue.pop()
        if isinstance(val,dict):
            for kk,vv in val.items():
                queue.append((f"{key}:{kk}",vv))
        else:
            key_vals.append((key,val))

    return dict(key_vals)

def search_train(
    training_output_root,
    images,
    ground_truths, 
    imgs_test,
    gts_test,
    configurations_overrides, 
    evaluation_metric_fn = None,
    device ='cpu',
    raise_on_exc = False
):
    default_params = cfg.get_default_config()
    results = []
    for custom in tqdm(configurations_overrides,desc='Running training experiments'):
        train_params = cfg.merge(default_params,custom)
        logger.info(f"Training with: f{pretty_print_dict(train_params)}")
        try:
            checkpoint, out_losses =seg.run_training(
                images,
                ground_truths,
                train_params, 
                training_output_root,
                device = device
            )

            model = torch.load(checkpoint)

            predictions, metrics = ev.evaluate_images(
                model, 
                imgs_test,
                gts_test,
                patch_overlap = .75,
                metric=evaluation_metric_fn,
                patch_size=train_params['patch_size'],
                device = device
            )

            # TODO elaborate visual evaluation
            for i,(img,gt,pred,met) in enumerate(zip(imgs_test,gts_test,predictions,metrics)):
                show_imgs = [img,pred[0] ,gt,*pred[1:]]
                vis.plot_row(show_imgs,vmin_vmax = (0,1),figsize=(50,10))
                plt.suptitle(f'Metric: {met}',y = .72)
                plt.savefig(checkpoint.parent/f'fig_test_{i}.png')
                plt.close()

            df = pd.DataFrame(
                list(enumerate(metrics)),
                columns = ['img_id','metric']
            )
            df.to_csv(checkpoint.parent/'metrics.csv')
            
            results.append((train_params,df))
        except Exception as e:
            logger.exception(e)
            if raise_on_exc:
                raise
    return results

