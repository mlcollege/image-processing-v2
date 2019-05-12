import pandas as pd
import numpy as np
import torch


def get_results_df(model, valid_loader):
    model.eval()

    label_class_ids = []
    label_class_scores = []
    predicted_class_ids = []
    predicted_class_scores = []
    all_images = []

    for images, labels in valid_loader:
        b_label_class_ids = labels.detach().numpy()
        label_class_ids += b_label_class_ids.tolist()
        b_predictions = model(images)
        b_predictions = b_predictions.detach().numpy()
        b_predicted_class_ids = np.argmax(b_predictions, axis=1)
        predicted_class_ids += b_predicted_class_ids.tolist()
        b_predicted_class_scores = np.max(b_predictions, axis=1)
        predicted_class_scores += b_predicted_class_scores.tolist()
        b_label_class_scores = b_predictions[np.arange(b_label_class_ids.shape[0]), b_label_class_ids]
        label_class_scores += b_label_class_scores.tolist()
        b_images = images.detach().numpy()
        all_images += b_images.tolist()

    label_class_names = [valid_loader.dataset.classes[c_id] for c_id in label_class_ids]
    predicted_class_names = [valid_loader.dataset.classes[c_id] for c_id in predicted_class_ids]

    return pd.DataFrame(
        {'label_class_name': label_class_names,
         'label_class_score': label_class_scores,
         'predicted_class_name_top1': predicted_class_names,
         'predicted_class_score_top1': predicted_class_scores,
         'image': [np.array(img) for img in all_images]}
    )


def get_recall(df, class_name):
    true_positives = len(df[(df.label_class_name == class_name) & (df.predicted_class_name_top1 == class_name)])
    trues = len(df[(df.label_class_name == class_name)])
    if trues == 0:
        trues = 1
    return round(true_positives / trues * 100, 2)


def get_precision(df, class_name):
    true_positives = len(df[(df.label_class_name == class_name) & (df.predicted_class_name_top1 == class_name)])
    positives = len(df[(df.predicted_class_name_top1 == class_name)])
    if positives == 0:
        positives = 1
    return round(true_positives / positives * 100, 2)


def get_accuracy(df):
    return round(float(np.mean((df.label_class_name == df.predicted_class_name_top1).astype(int))) * 100, 2)

def get_rec_prec(df, class_names=None):
    if class_names is None:
        class_names = ['T-shirt/top',
                       'Trouser',
                       'Pullover',
                       'Dress',
                       'Coat',
                       'Sandal',
                       'Shirt',
                       'Sneaker',
                       'Bag',
                       'Ankle boot']
    return pd.DataFrame(
        {
            "class_name": [class_name for class_name in class_names],
            "recall": [get_recall(df, class_name) for class_name in class_names],
            "precision": [get_precision(df, class_name) for class_name in class_names]
        })


def get_false_positives(df, label_class_name, predicted_class_name=None):
    if predicted_class_name is None:
        condition = (df['label_class_name'] == label_class_name) & (df['predicted_class_name_top1'] != label_class_name)
    else:
        condition = (df['label_class_name'] == label_class_name) & (df['predicted_class_name_top1'] == predicted_class_name)
    return df[condition].sort_values(by='predicted_class_score_top1', ascending=False)
