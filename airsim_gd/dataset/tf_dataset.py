from pathlib import Path
import json

import tensorflow as tf

from airsim_gd.dataset.utils.data import FEATURE_DESCRIPTION


class TFCVDataset(object):
    def __init__(self, tfr_dir, prefetch=16, batch_size=32, seed=None, num_parallel_calls=8,
                 max_depth=100.0, mode='mask', seg_id_list_path=Path("airsim_gd/dataset/sim_seg_id.json")):
        self.tfr_dir = tfr_dir
        self.prefetch = prefetch
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.seed = seed
        self.max_depth = max_depth
        self.mode = mode

        with open(Path(seg_id_list_path)) as f:
            seg_id_list = json.load(f)
        keys_tensor = tf.constant(seg_id_list)
        vals_tensor = tf.constant(list(range(seg_id_list)))
        self.sim2trainlabel = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), 0)

    def preprocess(self, tfr_example):
        features = tf.io.parse_single_example(tfr_example, FEATURE_DESCRIPTION)
        h = tf.cast(features['height'], dtype=tf.int32)
        w = tf.cast(features['width'], dtype=tf.int32)
        c = tf.cast(features['channels'], dtype=tf.int32)

        img = tf.cast(features['img_b'], tf.float32) / 255.0
        img = tf.reshape(img, (h, w, c))

        mask = tf.reshape(features['mask_b'], (h, w))
        mask = self.sim2trainlabel.lookup(mask)
        mask = tf.one_hot(mask, depth=len(self.sim2trainlabel))

        depth = features['depth_b']
        if self.max_depth is not None:
            depth[depth > self.max_depth] = self.max_depth
        depth = tf.reshape(depth, (h, w))

        if self.mode == 'mask':
            return img, mask

        elif self.mode == 'depth':
            return img, depth

        else:
            return img, [mask, depth]

    def generate_dataset(self):
        files = tf.io.matching_files(Path(self.tfr_dir, '*.tfrecord'))
        dataset_size = tf.cast(tf.shape(files)[0], tf.int64)
        if self.prefetch == 'batch_size':
            self.prefetch = self.batch_size

        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(self.preprocess, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=self.seed).repeat()
        return dataset.batch(self.batch_size).prefetch(self.prefetch)
