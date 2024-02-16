import pandas as pd
import numpy as np
import warnings
import cv2
from dataclasses import dataclass
import cv2
import scipy.optimize
import numpy as np
import denoiseg.metrics as m
import itertools
import pandas as pd
from tqdm.auto import tqdm


import logging
logger = logging.getLogger('denoiseg')
@dataclass
class SegmentationInstance:
    """Instance Shape"""
    mask:np.array
    image:np.array
    top_left_x: int
    top_left_y: int
    width:int
    height:int

@dataclass
class SegmentationInstanceFeatures:
    """Precipitate features"""
    ellipse_width_px:float
    ellipse_height_px:float
    ellipse_center_x:float
    ellipse_center_y:float
    ellipse_angle_deg:float
    circle_x:float
    circle_y:float
    circle_radius:float
    area_px:float
    shape:str


    
    
def sample_precision_recall(
    ground_truth,
    foreground,
    thresholds,
    component_limit = 500
):

    thr_imgs = [
        _threshold_foreground(foreground,thr) 
        for thr in thresholds
    ]
        
    precs_recs = [
        _calc_prec_rec_from_pred(
            ground_truth,
            img,component_limit = component_limit
        ) 
        for img in thr_imgs
    ]
    precs_recs = np.array(precs_recs)
    vis_thresholds = np.array(thresholds)[None].T
    precs_recs.shape,vis_thresholds.shape
    pr = np.hstack([precs_recs,vis_thresholds])
    precs, recs,thr = pr.T # [np.argsort(pr[:,0])].T
    f1s = np.array([f1(prec,rec)  for prec,rec in zip(precs,recs)])

    return thr_imgs,precs,recs,f1s

def evaluate_match(images, ground_truths,predictions,test_names = None):
    thr_low = .4
    thr_high = .8
    n = 5
    thresholds = np.concatenate([
        [0.0],
        np.linspace(thr_low,thr_high,n),
        [1.0]
    ])

    if test_names is None:
        test_names = [f'test_{i}' for i in range(len(images))] 
        
    evaluations = {}
    
    zipped = zip(test_names,images,ground_truths,predictions)
    for (name,img,gt,pred) in tqdm(zipped,desc = 'Matching prec'):
        
        thr_imgs,precs,recs,f1s = sample_precision_recall(
            gt,
            pred[1],
            thresholds
        )

        imgs_prec_rec = [
            {
                "precision":p,
                "recall":r,
                "f1":f,
                "threshold":t
            }
            for p,r,f,t in zip(precs,recs,f1s,thresholds)
        ]

        evaluations.setdefault(name,{})['samples'] = imgs_prec_rec
        evaluations.setdefault(name,{})['images'] = {
            "image":img,
            "ground_truth":gt,
            "foreground":pred[1]
        }

    return evaluations

def match_precipitates(prediction,label,component_limit = 500):
    p_n, p_grains = cv2.connectedComponents(prediction)
    l_n, l_grains = cv2.connectedComponents(label)    
    
    if p_n > component_limit or l_n > component_limit:
        logger.warning(
            f"Too many components found {component_limit=} #predictions:{p_n} #labels:{l_n}. Cropping"
        )
        p_n = min(p_n,component_limit)
        p_grains[p_grains>component_limit] = 0
        l_n = min(l_n,component_limit)
        l_grains[l_grains>component_limit] = 0
    
    # pairs only #TP
    pred_items,label_items = _pair_using_linear_sum_assignment(
        p_n, 
        p_grains,
        l_n, 
        l_grains
    )
    data = list(zip(pred_items,label_items))
    
    #FP
    p_set = set(pred_items)
    false_positives = [ i for i in range(1,p_n) if i not in p_set]
    for i in false_positives:
        data.append((i,None))
    
    #FN
    l_set = set(label_items)
    label_positives = [i for i in range(1,l_n) if i not in l_set]
    for i in label_positives:
        data.append((None,i))
    df = pd.DataFrame(data,columns = ['pred_id','label_id'])
    return df, p_grains, l_grains

def mean_evaluations(evaluations):
    thrss = []
    precss = []
    recallss = []
    f1ss = []

    for k,v in evaluations.items():
        if "samples" in v: # BACKWARDS 20240209 comabtility. WIll be removed
            imgs_prec_rec = v['samples']        
        else:
            imgs_prec_rec = v
        thresholds = [ vv['threshold'] for vv in imgs_prec_rec]
        precisions = [ vv['precision'] for vv in imgs_prec_rec]
        recalls = [ vv['recall'] for vv in imgs_prec_rec]
        f1s = [ vv['f1'] for vv in imgs_prec_rec]

        thrss.append(thresholds)
        precss.append(precisions)
        recallss.append(recalls)
        f1ss.append(f1s)


    mean_precisions =np.nanmean(precss,axis=0) 
    mean_recalls = np.nanmean(recallss,axis=0)
    mean_f1s = [f1(p,r) for p,r in zip(mean_precisions,mean_recalls)] 
    thresholds = np.mean(thrss,axis=0)
    
    return {
        'mean_precisions':mean_precisions,
        'mean_recalls':mean_recalls,
        'mean_f1s':mean_f1s,
        'thresholds':thresholds
    }

def extract_best_results(mean_evaluations):
    best_id = np.argmax(mean_evaluations['mean_f1s'])

    return {
        "f1":mean_evaluations['mean_f1s'][best_id],
        "threshold" : mean_evaluations['thresholds'][best_id],
        "precision" : mean_evaluations['mean_precisions'][best_id],
        "recall": mean_evaluations['mean_recalls'][best_id],
    }
def extract_instance_properties_df(image,mask):
    segmented_instances = extract_individiual_objects(image,mask)
    features = [extract_features(si) for si in segmented_instances]
    
    rows = [ (ins.__dict__|f.__dict__) for ins,f in zip(segmented_instances,features)]
    return  pd.DataFrame(rows)
    
def extract_features(segm_instance:SegmentationInstance) -> SegmentationInstanceFeatures:
    contour = get_shape_contour(segm_instance.mask)
    if len(contour) == 0:
        plt.imshow(segm_instance.mask)
        plt.show()
    (circle_x,circle_y),circle_radius = cv2.minEnclosingCircle(contour)
    
    # circle radius is from center of the circle to the CENTER of the furthest pixel
    # Since it is center of the pixel you need to add 1 to let the circle
    # reach the pixel borders on the both sides.
    circle_radius += .5
    
    if len(contour) >=5:
        (e_x,e_y),(e_width,e_height),angle = cv2.fitEllipseDirect(contour)
    else:
        e_x = circle_y
        e_y = circle_x
        e_width = circle_radius*2
        e_height = circle_radius*2
        angle = 0
    # ... >0 ensures that you count only ones, not 255 
    pixel_area = np.sum(segm_instance.mask>0)
    
    shape_cls = classify_shape(
        e_height,
        e_width,
        circle_radius,
        pixel_area
    )
    return SegmentationInstanceFeatures(
        ellipse_width_px=e_width,
        ellipse_height_px=e_height,
        ellipse_center_x=e_x + segm_instance.top_left_x,
        ellipse_center_y=e_y + segm_instance.top_left_y,
        ellipse_angle_deg=angle,
        circle_x=circle_x + segm_instance.top_left_x,
        circle_y=circle_y + segm_instance.top_left_y,
        circle_radius=circle_radius,
        area_px=pixel_area,
        shape = shape_cls
    )


def classify_shape(
    ellipse_height_px,
    ellipse_width_px,
    circle_radius,
    precipitate_area_px,
    needle_ratio = .5,
    irregullar_threshold = .6
)->str:
    h = ellipse_height_px
    w = ellipse_width_px
    minor,major = (h,w) if h<w else (w,h)
    axis_ratio = minor/major

    if axis_ratio < needle_ratio:
        return "shape_needle"

    circle_area = np.pi* circle_radius**2
    area_ratio = precipitate_area_px / circle_area
    
    if area_ratio < irregullar_threshold:
        return "shape_irregular"
    else:
        return "shape_circle"
    
    
def get_bounding_box(mask):
    """
    returns bounding box top,bottom,left,right pixel coordinates of the object 
    """
    
    yy,xx = np.nonzero(mask)

    nz_l,nz_r  = np.min(xx),np.max(xx)
    nz_t,nz_b = np.min(yy),np.max(yy)
    
    bb_l = np.maximum(0,nz_l)
    bb_r = np.minimum(mask.shape[1],nz_r )
    
    bb_t = np.maximum(0,nz_t)
    bb_b = np.minimum(mask.shape[0],nz_b)
    return  bb_t,bb_b +1,bb_l,bb_r+1


def extract_individiual_objects(image,mask):
    
    n_found, cmp_mask = cv2.connectedComponents(mask)
    components = (np.uint8(cmp_mask ==i) for i in range(1,n_found))
    bbs = ((get_bounding_box(c),c) for c in components)
    return [ 
        SegmentationInstance(
            top_left_x = l,
            top_left_y = t,
            width = r-l,
            height = b-t,
            mask = c[t:b,l:r],
            image = image[t:b,l:r]
        )
        for (t,b,l,r),c in bbs
    ]

def get_shape_contour(binary_img):    
    padding= 2
    img = np.pad(binary_img,padding)
    contours,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        warnings.warn("Multiple or no contour found, expected only one")

    c  = contours[0]
    return c[:,0,:]  -padding


def _construct_weight_map(weights_dict):
    # Remap arbitrary indices to integers
    p_map= {}

    for i,v in enumerate(weights_dict.keys()):
        p_map[v]=i

    l_keys = itertools.chain(
                *(list(k for k in v.keys()) for v in weights_dict.values())
            )
    l_unique = np.unique(list(l_keys))
    l_map={}
    for i,v in enumerate(l_unique):
        l_map[v]=i
        
    weights = np.zeros((len(p_map),len(l_map)))
    for i,(p,pv) in enumerate(weights_dict.items()):
        for l,lv in pv.items():
            weights[p_map[p],l_map[l]] = lv
    return weights,p_map,l_map

def _pair_using_linear_sum_assignment(p_n, p_grains,l_n, l_grains, cap=500):
    
    if cap is not None:
        p_n = min(cap,p_n)
        p_grains[p_grains >cap] = 0
        
        l_n = min(cap,l_n)
        l_grains[l_grains >cap] = 0
        
    weights_dict = _collect_pairing_weights(p_n, p_grains,l_n, l_grains)
    weights,p_map,l_map = _construct_weight_map(weights_dict)
    p_item_id,l_item_id = scipy.optimize.linear_sum_assignment(weights)
    
    inverse_p_map = { v:k for k,v in p_map.items()}
    p_item = np.array([inverse_p_map[idx] for idx in p_item_id])
    inverse_l_map = { v:k for k,v in l_map.items()}
    l_item = np.array([inverse_l_map[idx] for idx in l_item_id])
    return p_item,l_item
    
def _construct_weight_map(weights_dict):
    # Remap arbitrary indices to integers
    p_map= {}

    for i,v in enumerate(weights_dict.keys()):
        p_map[v]=i

    l_keys = itertools.chain(
                *(list(k for k in v.keys()) for v in weights_dict.values())
            )
    l_unique = np.unique(list(l_keys))
    l_map={}
    for i,v in enumerate(l_unique):
        l_map[v]=i
        
    weights = np.zeros((len(p_map),len(l_map)))
    for i,(p,pv) in enumerate(weights_dict.items()):
        for l,lv in pv.items():
            weights[p_map[p],l_map[l]] = lv
    return weights,p_map,l_map


def _collect_pairing_weights(p_n, p_grains,l_n, l_grains):
    weights_dict = {}
    iou = m.prepare_iou()
    for p_grain_id in range(1,p_n):
        p_grain_mask =  np.uint8(p_grains==p_grain_id)

        intersecting_ids = np.unique(l_grains*p_grain_mask)
        intersecting_ids = intersecting_ids[intersecting_ids>0]
        
        for l_grain_id in intersecting_ids:
            l_grain_mask = np.uint8(l_grains == l_grain_id)
            
            weight = 1 - iou(l_grain_mask,p_grain_mask)
            weights_dict.setdefault(p_grain_id,{}).setdefault(l_grain_id,weight)
            
    return weights_dict


def _threshold_foreground(foreground,thr):
    if thr == 0:
        return np.ones_like(foreground,dtype = np.uint8)
    elif thr == 1:
        return np.zeros_like(foreground,dtype = np.uint8)
    p = np.zeros_like(foreground)
    p[foreground>=thr] = 1
    return np.uint8(p)

def f1(precision, recall):
    if precision + recall ==0:
        return np.nan
    return 2*(precision * recall)/(precision + recall)

def _calc_prec_rec_from_pred(y,p,component_limit=500):    

    if (p == 1).all():
        return (0,1)
    elif (p == 0).all():
        return (1,0)
    
    y = np.uint8(y)
    df,_,_ = match_precipitates(p,y,component_limit = component_limit)
    return _prec_rec(df)


def _prec_rec(df):
    
    grains_pred = len(df[~df['pred_id'].isna()])
    grains_label = len(df[~df['label_id'].isna()])
    
    tp = df[~df['label_id'].isna() & ~df['pred_id'].isna()]
    if grains_pred !=0:
        precision = len(tp) / grains_pred
    else: 
        precision = np.nan
        
    if grains_label != 0:
        recall =  len(tp) / grains_label
    else:
        recall = np.nan
    return precision,recall
