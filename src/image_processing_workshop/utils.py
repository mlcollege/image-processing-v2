from io import BytesIO
import json
import numpy as np
import os
from PIL import Image
import requests
import tqdm
import urllib.request


def get_image_from_url(url, resize=None, to_grayscale=False):
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


def image_is_ok(file):
    try:
        image = Image.open(file)
        if image.mode != 'RGB':
            return False

        if image.format != 'JPEG':
            return False

        if image.height < 224 or image.width < 224:
            return False

        image.verify()
        return True
    except Exception as err:
        return False


def scrape_urls(url_file, category_name, root_folder='./dataset'):
    successed = 0
    failed  = 0

    train_valid_id = 0
    train_valid = ['train', 'valid']

    os.makedirs(os.path.join(root_folder, train_valid[0], category_name), exist_ok=True)
    os.makedirs(os.path.join(root_folder, train_valid[1], category_name), exist_ok=True)

    for url in tqdm.tqdm(open(url_file, 'r').readlines()):
        url = url.strip()
        file_name = url.split('/')[-1].split('.', 2)[0]+'.jpg'

        try:
            path = os.path.join(root_folder, train_valid[train_valid_id], category_name, file_name)
            urllib.request.urlretrieve(url, path)
            if not image_is_ok(path):
                os.remove(path)
                raise OSError(f'Not valid file: {path}')
        except Exception as err:
            print(err)
            failed += 1
        else:
            successed += 1
            train_valid_id = (train_valid_id + 1) % 2

    print(f'Failed {failed}\nSucc {successed}')
    return (os.path.join(root_folder, train_valid[0], category_name),
            os.path.join(root_folder, train_valid[1], category_name))
