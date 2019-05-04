import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO
import requests
import json


def get_image_from_url(url, resize=None, to_grayscale=True):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
    else:
        raise AttributeError("Wrong url")

    if to_grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    if resize:
        img = img.resize((resize[1], resize[0]))
    return np.array(img).astype("uint8")


def get_imagenet_category_names(use_cache=True):
    cache_path = './imagenet_category_names.json'
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'r') as fr:
            return json.load(fr)

    for attempt in range(3):
        response = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
        if response.status_code == 200:
            break
    if response.status_code != 200:
        return None

    id2category_name = json.loads(response.text)
    category_names = [id2category_name[str(k)][1] for k in range(1000)]
    with open(cache_path, 'w') as fw:
        json.dump(category_names, fw)
    return category_names
