import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib.tensorboard.plugins import projector
from PIL import Image
from tensorflow.keras.models import Model


def get_sprite_img(images, image_shape=[28, 28]):
    image_cout = len(images)
    h, w = image_shape[:2]

    rows = int(np.ceil(np.sqrt(image_cout)))
    cols = rows

    if len(image_shape) == 3:
        sprite_img = np.zeros([rows * h, cols * w, image_shape[2]])
    else:
        sprite_img = np.zeros([rows * h, cols * w])

    image_id = 0
    for row_id in range(rows):
        for col_id in range(cols):
            if image_id >= image_cout:
                if len(image_shape) == 3:
                    sprite_img = Image.fromarray(np.uint8(sprite_img))
                else:
                    sprite_img = Image.fromarray(np.uint8(sprite_img * 0xFF))
                return sprite_img

            row_pos = row_id * h
            col_pos = col_id * w
            sprite_img[row_pos:row_pos + h, col_pos:col_pos + w] = images[image_id].reshape(image_shape)
            image_id += 1
    return sprite_img

def save_sprite_image(sprite_img, path):
    plt.imsave(path, sprite_img, cmap='gray')


def get_label_class_names(labels, class_names):
    return [class_names[c_id] for c_id in np.argmax(labels, axis=1).tolist()]


def save_label_class_names(label_class_names, path):
    with open(path, 'w') as fw:
        for name in label_class_names:
            fw.write(name + '\n')

def get_embedding(model, images, embedding_layer_name='embedding'):
    embedding_model = Model(inputs=model.input,
                            outputs=model.get_layer(embedding_layer_name).output)
    return embedding_model.predict(images)


def setup_embedding_projection(model, images, labels, log_dir, class_names=None, embedding_layer_name='embedding'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if class_names is None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Extract class names for test data.
    path_for_metadata = os.path.join(log_dir, 'metadata.tsv')
    label_class_names = get_label_class_names(labels, class_names)
    save_label_class_names(label_class_names, path_for_metadata)

    # Extract sprite images for test data.
    path_for_sprites = os.path.join(log_dir, 'digits.png')
    sprite_image = get_sprite_img(images)
    save_sprite_image(sprite_image, path_for_sprites)

    # Get test data embeddings.
    raw_embedding = get_embedding(model, images, embedding_layer_name)
    embedding_var = tf.Variable(raw_embedding, name='embedding')
    summary_writer = tf.summary.FileWriter(log_dir)

    # Setup projector.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = 'metadata.tsv'
    embedding.sprite.image_path = 'digits.png'
    embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_dir, "model.ckpt"), 1)
