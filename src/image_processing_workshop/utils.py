from io import BytesIO
import json
import numpy as np
import os
from PIL import Image
import requests
import tqdm
import urllib.request
import socket


def get_patch(patch_id=None, use_cache=True, resize=None):
    url = "https://pbs.twimg.com/media/DSU7iNMU8AAciHy.png"
    mapping = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 0],
        3: [0, 1]
    }

    cache_path = './patch.npy'
    if use_cache and os.path.exists(cache_path):
        patches = np.load(cache_path)
    else:
        patches = get_image_from_url(url, resize=[224, 224, 3])

    row_id, col_id = mapping.get(patch_id, [None, None])
    if row_id is None:
        return patches
    else:
        patch = patches[row_id*112: (row_id+1)*112, col_id*112: (col_id+1)*112, :]
        if resize:
            patch = Image.fromarray(patch)
            if len(resize) >= 2:
                patch = patch.resize((resize[1], resize[0]))
            else:
                patch = patch.resize([resize[0], int(patch.size[1] / patch.size[0] * resize[0])])
        return np.array(patch).astype("uint8")



def apply_patch(img, patch, pos_w=0, pos_h=0):
    img_shape = img.shape
    patch_shape = patch.shape
    if img_shape[0] < patch_shape[0] or img_shape[1] < patch_shape[1]:
        return img

    if pos_w + patch.shape[1] >= img.shape[1]:
        pos_w = img.shape[1] - patch.shape[1]
    if pos_h + patch.shape[0] >= img.shape[0]:
        pos_h = img.shape[0] - patch.shape[0]

    patch_ids = np.where(patch > 0)
    img_ids = (patch_ids[0]+pos_h, patch_ids[1]+pos_w, patch_ids[2])

    corrupted_img = img.copy()
    corrupted_img[img_ids] = patch[patch_ids]
    return corrupted_img


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
        if len(resize) >= 2:
            img = img.resize((resize[1], resize[0]))
        else:
            img = img.resize([resize[0], int(img.size[1] / img.size[0] * resize[0])])
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


def scrape_urls(url_file, category_name, root_folder='./dataset', valid_rate=3):
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(15)

    successed = 0
    failed = 0

    scraped_image_count = 0
    subfolder = 'train'

    os.makedirs(os.path.join(root_folder, 'train', category_name), exist_ok=True)
    os.makedirs(os.path.join(root_folder, 'valid', category_name), exist_ok=True)

    for url in tqdm.tqdm(open(url_file, 'r').readlines()):
        url = url.strip()
        file_name = url.split('/')[-1].split('.', 2)[0] + '.jpg'

        try:
            path = os.path.join(root_folder, subfolder, category_name, file_name)
            urllib.request.urlretrieve(url, path)
            if not image_is_ok(path):
                os.remove(path)
                raise OSError(f'Not valid file: {path}')
        except Exception as err:
            print(err)
            failed += 1
        else:
            successed += 1
            scraped_image_count = (scraped_image_count + 1) % valid_rate
            if scraped_image_count == 0:
                subfolder = 'valid'
            else:
                subfolder = 'train'

    socket.setdefaulttimeout(old_timeout)
    print(f'Failed {failed}\nSucc {successed}')
    return (os.path.join(root_folder, 'train', category_name),
            os.path.join(root_folder, 'valid', category_name))
