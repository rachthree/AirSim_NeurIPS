import json
from pathlib import Path
import time
import glob
from argparse import ArgumentParser
import math

import tensorflow as tf
from tensorflow import keras

from airsim_gd.dataset.tf_dataset import TFCVDataset
from airsim_gd.vision.nn.utils import train_utils
from airsim_gd.vision.nn import fast_scnn
from airsim_gd.dataset.utils.data import FEATURE_DESCRIPTION


GEN_MODELS = {'fast_scnn': fast_scnn.generate_model}


class CVTrainer(object):
    # TODO: Load params from config yaml file
    def __init__(self, *, train_dir, val_dir, save_dir,
                 arch, sess_name, input_names, output_names,
                 epochs, seed=None, end_learning_rate, batch_size=16, n_aux=0,
                 max_depth, sim_seg_id_list_path, seg_id_map_path):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.save_dir = Path(save_dir, sess_name)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.arch = arch
        self.input_layer_names = input_names
        self.output_layer_names = output_names

        self.epochs = epochs
        self.seed = seed
        self.end_learning_rate = end_learning_rate
        self.batch_size = batch_size

        with open(Path(sim_seg_id_list_path)) as f:
            self.sim_seg_id_list = json.load(f)

        with open(Path(seg_id_map_path)) as f:
            self.id2rgb_dict = json.load(f)

        self.sess_name = time.ctime(time.time()).replace(':', '.') if sess_name is None else sess_name

        print('\nPrepping training dataset...\n')
        self.train_ds = TFCVDataset(tfr_dir=self.train_dir, seed=self.seed, batch_size=self.batch_size,
                                    max_depth=max_depth, output_names=output_names, seg_id_list=self.sim_seg_id_list).generate_dataset()
        print('\nPrepping validation dataset...\n')
        self.val_ds = TFCVDataset(tfr_dir=self.val_dir, seed=self.seed, batch_size=self.batch_size//2,
                                  max_depth=max_depth, output_names=output_names, seg_id_list=self.sim_seg_id_list).generate_dataset()

        file_list = glob.glob(str(Path(self.train_dir, f'*.tfrecord')))
        tf_example = tf.data.TFRecordDataset(file_list[0])
        tf_example = tf_example.map(lambda x: tf.io.parse_single_example(x, FEATURE_DESCRIPTION))
        tf_example = [data for data in tf_example][0]  # only need one
        self.img_shape = (tf_example['image/height'].numpy(), tf_example['image/width'].numpy(), tf_example['image/channels'].numpy())
        self.n_train_data = len(file_list)

    def get_model(self):
        n_classes = len(self.sim_seg_id_list)

        return GEN_MODELS[self.arch](self.img_shape, n_classes)

    def get_callbacks(self):
        ckpt_path = Path(self.save_dir, 'checkpoints')
        cp_cb = keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor='val_loss', verbose=1, save_best_only=False)

        es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50,
                                              restore_best_weights=True)
        tb_cb = keras.callbacks.TensorBoard(log_dir=str(self.save_dir), histogram_freq=1, write_graph=True, write_images=False,
                                            update_freq='epoch')
        # img_cb = train_utils.TFImageCallback(model_save_dir=self.save_dir, seg_id_list=self.sim_seg_id_list,
        #                                      input_layer_names=self.input_layer_names, output_layer_names=self.output_layer_names,
        #                                      id22rgb_dict=self.id2rgb_dict)
        return [cp_cb, es_cb, tb_cb]

    def load_model(self):
        return keras.models.load_model(self.save_dir)

    def __call__(self, mode):
        callback_list = self.get_callbacks()
        # steps_per_epoch =

        if mode == 'train':
            print('\nCreating model...\n')
            model, loss_dict, loss_weights = self.get_model()
            decay_steps = self.epochs * self.n_train_data
            learning_rate = keras.optimizers.schedules.PolynomialDecay(0.045, decay_steps, self.end_learning_rate,
                                                                       power=0.5)
            optimizer = keras.optimizers.SGD(momentum=0.9, learning_rate=learning_rate)
            model.compile(loss=loss_dict, loss_weights=loss_weights, optimizer=optimizer, metrics=['accuracy'])

            print('\nModel created. Training now...\n')
            history = model.fit(self.train_ds, epochs=self.epochs, validation_data=self.val_ds, callbacks=callback_list)

        elif mode == 'resume':
            print('\nLoading model...\n')
            model = self.load_model()

            print('\nResuming training...\n')
            history = model.fit(self.train_ds, epochs=self.epochs, validation_data=self.val_ds, callbacks=callback_list)

        else:
            raise ValueError(f"Mode {mode} not recognized.")

        print('\nTraining completed.\n')
        model.save(str(self.save_dir), overwrite=False)
        print('\nModel saved.\n')
        return history


def main(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    trainer = CVTrainer(train_dir=args.train_dir, val_dir=args.val_dir, save_dir=args.save_dir,
                        arch=args.arch, sess_name=args.sess_name,
                        epochs=args.epochs, seed=args.seed, end_learning_rate=args.end_learning_rate, batch_size=args.batch_size, n_aux=args.n_aux,
                        max_depth=args.max_depth, sim_seg_id_list_path=args.sim_seg_id_list_path, seg_id_map_path=args.seg_id_map_path,
                        input_names=args.input_names, output_names=args.output_names)

    trainer(args.mode)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=Path('../data/nominal/train'))
    parser.add_argument('--val_dir', type=str, default=Path('../data/nominal/val'))
    parser.add_argument('--save_dir', type=str, default=Path.cwd().parent.joinpath("model_training"))
    parser.add_argument('--mode', type=str, choices=['train', 'resume'])
    parser.add_argument('--arch', type=str, default='fast_scnn')
    parser.add_argument('--arch_args', type=str, default=None)
    parser.add_argument('--sess_name', type=str, default=time.ctime(time.time()).replace(':', '.'))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sim_seg_id_list_path', type=str, default=Path('sim_seg_id_list.json'))
    parser.add_argument('--seg_id_map_path', type=str, default=Path("airsim_gd/dataset/segmentation_id_maps.json"))
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--end_learning_rate', type=float, default=0.00001)
    parser.add_argument('--n_aux', type=int, default=0)
    parser.add_argument('--max_depth', type=float, default=100.0)
    parser.add_argument('--input_names', type=str, nargs='+', default=['input_layer'])
    parser.add_argument('--output_names', type=str, nargs='+', default=None)
    args = parser.parse_args()
    main(args)