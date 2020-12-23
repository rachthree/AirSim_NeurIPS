from pathlib import Path
import glob
import json

import tensorflow as tf

from airsim_gd.dataset.utils.data import FEATURE_DESCRIPTION

AUTO = tf.data.experimental.AUTOTUNE

class TFCVDataset(object):
    def __init__(self, *, tfr_dir, prefetch=1, batch_size=32, seed=None, num_parallel_calls=4, img_norm=True,
                 max_depth=100.0, mode='mask', seg_id_list, output_names=None, autotune=False):
        self.tfr_dir = tfr_dir
        self.prefetch = prefetch
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.seed = seed
        self.max_depth = max_depth
        self.mode = mode
        self.img_norm = img_norm
        self.n_classes = len(seg_id_list)
        self.output_names = output_names

        if autotune:
            # Override
            self.prefetch = AUTO
            self.num_parallel_calls = AUTO

        keys_tensor = tf.constant(seg_id_list)
        vals_tensor = tf.constant(list(range(self.n_classes)))
        self.sim2trainlabel = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), 0)
        self.img_shape = self.get_image_shape()

    def get_image_shape(self):
        file_list = glob.glob(str(Path(self.tfr_dir, f'*.tfrecord')))
        tf_example = tf.data.TFRecordDataset(file_list[0])
        tf_example = tf_example.map(lambda x: tf.io.parse_single_example(x, FEATURE_DESCRIPTION))
        tf_example = [data for data in tf_example][0]  # only need one
        img_shape = [tf_example['image/height'].numpy(), tf_example['image/width'].numpy(), tf_example['image/channels'].numpy()]
        return img_shape

    def preprocess(self, tfr_serialized):
        # tf_record_data = tf.data.TFRecordDataset(tf_record)
        # parsed_features = tf_record_data.map(lambda x: tf.io.parse_single_example(tf_record, FEATURE_DESCRIPTION))
        features = tf.io.parse_single_example(tfr_serialized, FEATURE_DESCRIPTION)
        # features = [data for data in parsed_features][0]
        img = tf.cast(tf.io.parse_tensor(features['image/decoded'], tf.uint8), tf.float32)
        img.set_shape(self.img_shape)
        if self.img_norm:
            img /= 255.0
        # img = tf.expand_dims(img, axis=0)

        mask = tf.io.parse_tensor(features['image/mask'], tf.int32)
        mask.set_shape([self.img_shape[0], self.img_shape[1]])
        mask = self.sim2trainlabel.lookup(mask)
        mask = tf.one_hot(mask, depth=self.n_classes)
        # mask = tf.expand_dims(mask, axis=0)

        depth = tf.cast(tf.io.parse_tensor(features['image/depth'], tf.float64), tf.float32)
        depth.set_shape([self.img_shape[0], self.img_shape[1]])
        if self.max_depth is not None:
            depth = tf.clip_by_value(depth, clip_value_min=0.0, clip_value_max=self.max_depth)
            # depth /= self.max_depth
        # depth = tf.expand_dims(depth, axis=0)

        if self.mode == 'mask':
            output = mask

        elif self.mode == 'depth':
            output = depth

        else:
            raise ValueError('Mode for TFCVDataset not recognized.')

        if self.output_names is not None:
            out_dict = {}
            for layer in self.output_names:
                out_dict[layer] = output
            output = out_dict

        return img, output

    def generate_dataset(self):
        files = tf.io.matching_files(str(Path(self.tfr_dir, '*.tfrecord')))
        dataset_size = tf.cast(tf.shape(files)[0], tf.int64).numpy()

        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=self.seed)
        # dataset = tf.data.Dataset.list_files(str(Path(self.tfr_dir, '*.tfrecord')), shuffle=True, seed=self.seed)
        # dataset = dataset.repeat()  # model.fit will figure out how many in epoch on first epoch
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=1)
        dataset = dataset.map(self.preprocess, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch)
        return dataset
