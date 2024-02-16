import itertools
import matplotlib.pyplot as plt
import numpy as np
import denoiseg.dataset as ds
import denoiseg.instance_analysis as ia
import matplotlib.patches
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import collections


def plot_loss(loss_df,figsize = None,ax=None):
    
    loss_df.plot(
        xlabel='Epochs',
        ylabel='Loss [log]',
        logy=True,
        title = "Training Loss on Dataset",
        figsize = figsize,
        ax = ax
    )
    

def plot_precision_recall_curve(mean_evaluations,ax = None):
    if ax is None:
        _,ax = plt.subplots(1,1)
    
    _plot_f1_background(ax)
    
    thresholds =  mean_evaluations['thresholds']
    precs = mean_evaluations['mean_precisions']
    recs = mean_evaluations['mean_recalls']
    f1s = [ia.f1(p,r) for p,r in zip(precs,recs)]

    ax.plot(recs, precs)
    ax.set_title("Precision/Recall Chart")
    
    ax.set_xlabel('Recall')
    ax.set_xlim(0,1)
    
    ax.set_ylabel('Precision')
    ax.set_ylim(0,1)
    
    ax.axis('scaled')

    for prec,rec,f1,thr in zip(precs,recs,f1s,thresholds):
        lbl = f"$f_1$:{f1:.2} $t$:{thr:.2}"
        #ax.plot([rec],[prec],'x',label=)
        ax.text(rec,prec,lbl)

        
def plot_instance_details(df_features,columns=5):
    rows = int( np.ceil(len(df_features)/columns))
    fig,axs = plt.subplots(rows,columns,figsize=(3*columns,3*rows))
    for ax, features in zip(axs.flatten(), df_features.itertuples()):
        img = np.dstack([features.image]*3 + [(np.float32(features.mask) +1)/2])
        ax.imshow(img)
        ax.axis('off')
        radius = features.circle_radius + 2
        
        title = f"x:{int(features.circle_x)} y:{int(features.circle_y)} S(px): {int(features.area_px)}"
        ax.set_title(title)
        ax.imshow(img)

        e = _get_ellipse(features)
        ax.add_patch(e)
        
    fig.suptitle(f"Instances Details")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig

def plot_histograms(df,bins=20):
    if 'area_um' in df.columns: 
        areas = df.area_um.to_numpy()
    else: 
        areas = df.area_px.to_numpy()
    
    fig,axs = plt.subplots(2,1)
    
    _,_, bars = axs[0].hist(areas.flatten(),bins=bins)
    # don't show zeros
    axs[0].bar_label(bars, labels=[int(v) if v > 0 else '' for v in bars.datavalues])
    axs[0].set_title(f"Instance Area Histogram (Total = {len(areas)})")
    axs[0].set_ylabel("Count")
    axs[0].set_xlabel("$\\mu m$")
    
    _plot_shape_bar(axs[1],df)
    return fig


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
    return axs


def sample_ds(images,ground_truths,training_params, weightmaps=None, n = 5):
    
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
        training_params.get('denoise_enabled',True),
        weightmaps = weightmaps
    )
    
    tts = (t for t in dataset if np.sum(t["has_label"]) > 0)

    for i, t in enumerate(itertools.islice(tts, 0, 15)):
        if i >= n:
            break
        img = t["x"][0]
        mask = t["y_segmentation"][0]
        den = np.squeeze(t["mask_denoise"])
        wm = np.squeeze(t["weightmap"])
        imgs = [img, mask, den,wm]

        _, axs = plt.subplots(1, len(imgs), figsize=(len(imgs) * 5, 5))

        for ax, im in zip(axs, imgs):
            ax.imshow(im, vmin=0, vmax=np.max(im), cmap="gray")





def _get_shape_color(shape_class):
    if shape_class == "shape_irregular":
        return "magenta"
    elif shape_class == "shape_circle":
        return "#00FF00"
    else: 
        return 'red'

def _get_ellipse(features):
    deg = features.ellipse_angle_deg    
    x = features.ellipse_center_x - features.top_left_x
    y =features.ellipse_center_y - features.top_left_y
    width = features.ellipse_width_px 
    height = features.ellipse_height_px

    c = _get_shape_color(features.shape)
    return matplotlib.patches.Ellipse(
        (x,y),
        width,
        height,
        angle = deg,
        fill=False,
        edgecolor=c,
        lw=2,
        alpha = .5
    )

def add_contours_morph(img,mask,contour_width = 1, color_rgb = (255,0,0)):

    if len(img.shape) == 2:
        img_rgb = np.dstack([img]*3)
    else:
        img_rgb = img
        
    contours = mask - cv2.erode(mask,np.ones((3,3))) 
    contours = cv2.dilate(contours,np.ones((contour_width,contour_width)))
    img_rgb[contours==255] = color_rgb
    return img_rgb

def _plot_shape_bar(ax,df):
    labels, values = zip(*collections.Counter(df['shape']).items())
    indexes = np.arange(len(labels))
    width=.8
    bar_colors = list(map(_get_shape_color,labels))
    texts = list(map(_get_shape_text,labels))
    
    ax.set_title("Shape Distribution")
    ax.bar(indexes, values, width,color = bar_colors)
    ax.set_xticks(indexes, texts)
    ax.set_ylabel("Number of instances")
    ax.set_xlabel("Shapes")
    

def _get_shape_text(shape_class):
    if shape_class == "shape_irregular":
        return "Irregular"
    elif shape_class == "shape_circle":
        return "Circle"
    else: 
        return 'Needle-like'


def _norm(img):
    m = np.min(img)
    return (img - m)/(np.max(img)-m)

def _save_imgs(eval_path, image_dict):
    for k,v in image_dict.items():
        path = eval_path/f"{k}.png" 
        v = np.uint8(_norm(v)*255)
        imageio.imwrite(path,v)

def _plot_f1_background(ax,nn=100):
    x = np.linspace(0, 1, nn)
    y = np.linspace(0, 1, nn)
    xv, yv = np.meshgrid(x, y)

    f1_nn = np.array([ ia.f1(yy,xx) for yy in y for xx in x ])
    f1_grid = (f1_nn.reshape((nn,nn)) % .1) > .05
    ax.imshow(f1_grid,alpha = .1,cmap='gray', extent=[0,1,1,0])
