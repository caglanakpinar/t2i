import os
import keras
import tensorflow as tf

from mlp import Paths, Params, BaseData, log


class BSRData(BaseData):
    def __init__(self, params: Params):
        self.params = params
        self.dataset_url = self.params.get('dataset_url')
        print("AAAASADSADSA :::::SADSA ::::", self.dataset_url)
        self.data_dir = keras.utils.get_file(origin=self.dataset_url, fname="BSR", untar=True)
        self.root_dir = os.path.join(self.data_dir, "BSR/BSDS500/data")
        self.crop_size = self.params.get('crop_size')
        self.upscale_factor = self.params.get('upscale_factor')
        self.input_size = self.crop_size // self.upscale_factor
        self.batch_size = self.params.get('batch_size')
        self.validation_split = self.params.get('validation_split')

    @staticmethod
    def process_input(input, input_size, upscale_factor):
        input = tf.image.rgb_to_yuv(input)
        last_dimension_axis = len(input.shape) - 1
        y, u, v = tf.split(input, 3, axis=last_dimension_axis)
        return tf.image.resize(y, [input_size, input_size], method="area")

    @staticmethod
    def process_target(input):
        input = tf.image.rgb_to_yuv(input)
        last_dimension_axis = len(input.shape) - 1
        y, u, v = tf.split(input, 3, axis=last_dimension_axis)
        return y

    def create_tfds(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.root_dir,
            batch_size=self.batch_size,
            image_size=(self.crop_size, self.crop_size),
            validation_split=self.validation_split,
            subset="training",
            seed=1337,
            label_mode=None,
        )
        train_ds = train_ds.map(
            lambda x: (
                self.process_input(x, self.input_size, self.upscale_factor),
                self.process_target(x)
            )
        )

        valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.root_dir,
            batch_size=self.batch_size,
            image_size=(self.crop_size, self.crop_size),
            validation_split=self.validation_split,
            subset="validation",
            seed=1337,
            label_mode=None,
        )
        valid_ds = valid_ds.map(
            lambda x: (
                self.process_input(x, self.input_size, self.upscale_factor),
                self.process_target(x)
            )
        )
        self.data = (
            train_ds.prefetch(buffer_size=32),
            valid_ds.prefetch(buffer_size=32)
        )


    @classmethod
    def read(
            cls,
            params: Params,
            **kwargs
    ):
        _cls = BSRData(params)
        _cls.create_tfds()
        return _cls

