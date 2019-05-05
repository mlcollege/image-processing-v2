import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def plot_image(input_tensor, image_shape=None, figsize=(10, 10)):
    # Type check.
    if isinstance(input_tensor, torch.Tensor):
        img = input_tensor.detach().numpy()
    else:
        img = input_tensor

    # Reshaping
    if image_shape:
        img = img.reshape(*image_shape)
    elif len(img.shape) == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
        # Get rid of 1 shape in grayscale because of imshow
        if img.shape[2] == 1:
            img = img.reshape(img.shape[0], img.shape[1])

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    if img.shape[-1] != 3:
        ax.imshow(img, cmap='gray')
    else:
        if img.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax


def plot_classify(input_tensor, model, image_shape=None, figsize=(10, 10), topn=10, category_names=None):
    # Transform input_tensor to batch and make predicton.
    input_tensor = input_tensor.unsqueeze(dim=0)
    model.eval()
    predictions = model(input_tensor)
    predictions = predictions.detach().numpy().squeeze()

    # Preselct topn predictions.
    if category_names is None:
        category_names = ['T-shirt/top', 'Trouser', 'Pullover',
                          'Dress', 'Coat', 'Sandal', 'Shirt',
                          'Sneaker', 'Bag', 'Ankle Boot']
    topn = min(topn, len(category_names))
    category_names = np.array(category_names)
    topn_pos = predictions.argsort()[::-1][:topn]
    topn_names = category_names[topn_pos].tolist()
    topn_preds = predictions[topn_pos]

    img = input_tensor.detach().numpy().squeeze()
    # Reshaping
    if image_shape:
        img = img.reshape(*image_shape)
    elif len(img.shape) == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
        # Get rid of 1 shape in grayscale because of imshow
        if img.shape[2] == 1:
            img = img.reshape(img.shape[0], img.shape[1])

    fig, (ax1, ax2) = plt.subplots(figsize=figsize, ncols=2)
    if img.shape[-1] != 3:
        ax1.imshow(img, cmap='gray')
    else:
        if img.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        ax1.imshow(img)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    ax2.barh(np.arange(topn), topn_preds)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(topn))
    ax2.set_yticklabels(topn_names, size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def plot_df_examples(df, image_shape=None):
    examples_count = min(25, len(df))
    cols = 5
    rows = np.ceil(examples_count / cols)

    fig = plt.figure(figsize=(20, 25))
    for img_id in range(examples_count):
        ax = plt.subplot(rows, cols, img_id + 1)

        img = df.image.iloc[img_id]
        if img is None:
            continue

        # Reshaping
        if image_shape:
            img = img.reshape(*image_shape)
        elif len(img.shape) == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
            # Get rid of 1 shape in grayscale because of imshow
            if img.shape[2] == 1:
                img = img.reshape(img.shape[0], img.shape[1])

        prediction_name = df.predicted_class_name_top1.iloc[img_id]
        prediction_score = df.predicted_class_score_top1.iloc[img_id]
        if img.shape[-1] != 3:
            ax.imshow(img, cmap='gray')
        else:
            if img.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
            ax.imshow(img)
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
