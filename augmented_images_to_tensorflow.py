import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from show_images import load_image

# Shuffle false because labels will be loaded in same format so have to be in same structure
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120))) # Make image smaller, neural network more efficient
train_images = train_images.map(lambda x: x / 255.0) # Lets us apply sigmoid activation to final layer of neural network

test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = train_images.map(load_image)
test_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images = train_images.map(lambda x: x / 255.0)

val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = train_images.map(load_image)
val_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
val_images = train_images.map(lambda x: x / 255.0)

print(train_images.as_numpy_iterator().next())