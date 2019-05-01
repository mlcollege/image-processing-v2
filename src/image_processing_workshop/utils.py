import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO


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
