import matplotlib.pyplot as plt


def plot_row(imgs, vmin_vmax=None, cell_size=5):
    _, axs = plt.subplots(1, len(imgs), figsize=(cell_size * len(imgs), cell_size))

    for ax, im in zip(axs, imgs):
        if vmin_vmax is not None:
            ax.imshow(im, vmin=vmin_vmax[0], vmax=vmin_vmax[1],cmap='gray')
        else:
            ax.imshow(im)
