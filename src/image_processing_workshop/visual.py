import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import torch


def plot_image(img, ax=None, title=None, normalize=True, is_grayscale=True, reshape=[28, 28]):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
    if isinstance(img, torch.Tensor):
        img = img.detach().numpy()
    if (len(img.shape) == 3 and is_grayscale) or (len(img.shape) == 4 and not is_grayscale):
        img = img[0]
    if reshape:
        img = img.reshape(*reshape)

    if is_grayscale:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax

def plot_classify(img, model):
    if img.shape[0] != 1:
        img = img.unsqueeze(dim=0)
    model.eval()
    ps = model(img).detach().numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='gray')
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['T-shirt/top',
                        'Trouser',
                        'Pullover',
                        'Dress',
                        'Coat',
                        'Sandal',
                        'Shirt',
                        'Sneaker',
                        'Bag',
                        'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def plot_recon(img, recon):
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')



def plot_df_examples(df, image_shape=(28, 28)):
    examples_count = min(25, len(df))
    cols = 5
    rows = np.ceil(examples_count / cols)

    fig = plt.figure(figsize=(20, 25))
    for img_id in range(examples_count):
        ax = plt.subplot(rows, cols, img_id + 1)

        img = df.image.iloc[img_id]
        if img is None:
            continue
        img = img.reshape(image_shape)
        prediction_name = df.predicted_class_name_top1.iloc[img_id]
        prediction_score = df.predicted_class_score_top1.iloc[img_id]

        ax.imshow(img, cmap='gray')
        ax.set_title("{0}: {1}".format(prediction_name, round(prediction_score, 2)))
        ax.axes.set_axis_off()


def plot_coocurance_matrix(df, use_log=False):

    coocurance_cols = ['label_class_name', 'predicted_class_name_top1']
    coocurance_df = pd.pivot_table(df[coocurance_cols], index=coocurance_cols[0],
                                   columns=coocurance_cols[1], aggfunc=len, fill_value=0)
    if use_log:
        coocurance_df = np.log(coocurance_df)
        coocurance_df = coocurance_df.replace([-np.inf], 0)

    coocurance_df = coocurance_df.div(coocurance_df.sum(axis=1), axis=0) * 100
    coocurance_df = coocurance_df.round(2)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    sns.heatmap(coocurance_df, ax=ax, annot=True, linewidths=.5,
                cbar_kws={"orientation": "horizontal"}, cmap="YlGnBu")
