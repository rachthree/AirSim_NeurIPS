import glob
from pathlib import Path
import json
import numpy as np

import tensorflow as tf
import tensorflow.train as tf_train


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf_train.Feature(bytes_list=tf_train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, np.ndarray):
        value = [value]
    return tf_train.Feature(float_list=tf_train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, np.ndarray):
        value = [value]
    return tf_train.Feature(int64_list=tf_train.Int64List(value=value))


def _tensor_feature(array):
    return tf.io.serialize_tensor(array)


FEATURE_DESCRIPTION = {'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/channels': tf.io.FixedLenFeature([], tf.int64),
    'image/decoded': tf.io.FixedLenFeature([], tf.string),
    'image/mask': tf.io.FixedLenFeature([], tf.string),
    'image/depth':  tf.io.FixedLenFeature([], tf.string)
}


def get_tf_example(img_path, mask_path, depth_path):
    with open(img_path, 'rb') as f:
        img_str = f.read()
    img = tf.image.decode_image(img_str, channels=3)
    img = img.numpy()
    img_shape = img.shape

    mask_mat = np.load(mask_path)
    depth_mat = np.load(depth_path)

    feature = {'image/height': _int64_feature(img_shape[0]),
        'image/width': _int64_feature(img_shape[1]),
        'image/channels': _int64_feature(img_shape[2]),
        'image/decoded': _bytes_feature(_tensor_feature(img)),
        'image/mask': _bytes_feature(_tensor_feature(mask_mat)),
        'image/depth':  _bytes_feature(_tensor_feature(depth_mat))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_tf_records(file_dir):
    img_list = glob.glob(str(Path(file_dir, f'*.jpg')))
    for img_path in img_list:
        img_path = Path(img_path)
        mask_path = str(img_path).replace('.jpg', '_mask.npy')
        depth_path = str(img_path).replace('.jpg', '_depth.npy')

        filename = Path(file_dir, img_path.name.replace('.jpg', '.tfrecord'))
        example = get_tf_example(img_path, mask_path, depth_path)
        with tf.io.TFRecordWriter(str(filename)) as f:
            f.write(example.SerializeToString())
