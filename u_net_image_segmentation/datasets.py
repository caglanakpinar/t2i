import os
import random
import tensorflow as tf
from tensorflow import io as tf_io
from tensorflow import image as tf_image
from tensorflow import data as tf_data

from mlp import Paths, Params, BaseData, log


class AnnotationsData(BaseData):
    def __init__(self, params: Params):
        self.params = params
        self.input_dir = self.params.get('input_dir')
        self.target_dir = self.params.get('target_dir')
        self.img_size = self.params.get('img_size')
        self.num_classes = self.params.get('num_classes')
        self.batch_size = self.params.get('batch_size')
        self.paths = {}

    def get_paths(self):
        input_img_paths, target_img_paths = (
            sorted(
                [
                    os.path.join(self.input_dir, fname)
                    for fname in os.listdir(self.input_dir)
                    if fname.endswith(".jpg")
                ]
            ),
            sorted(
                [
                    os.path.join(self.target_dir, fname)
                    for fname in os.listdir(self.target_dir)
                    if fname.endswith(".png") and not fname.startswith(".")
                ]
            )

        )
        # Split our img paths into a training and a validation set
        val_samples = 1000
        random.Random(1337).shuffle(input_img_paths)
        random.Random(1337).shuffle(target_img_paths)
        self.paths = {
            "train_input_img_paths": input_img_paths[:-val_samples],
            "train_target_img_paths": target_img_paths[:-val_samples],
            "val_input_img_paths": input_img_paths[-val_samples:],
            "val_target_img_paths": target_img_paths[-val_samples:],
        }

    def get_dataset(
            self,
            batch_size,
            img_size,
            input_img_paths,
            target_img_paths,
            max_dataset_len=None,
    ):
        """Returns a TF Dataset."""

        def load_img_masks(input_img_path, target_img_path):
            input_img = tf_io.read_file(input_img_path)
            input_img = tf_io.decode_png(input_img, channels=3)
            input_img = tf_image.resize(input_img, img_size)
            input_img = tf_image.convert_image_dtype(input_img, "float32")

            target_img = tf_io.read_file(target_img_path)
            target_img = tf_io.decode_png(target_img, channels=1)
            target_img = tf_image.resize(target_img, img_size, method="nearest")
            target_img = tf_image.convert_image_dtype(target_img, "uint8")

            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            target_img -= 1
            return input_img, target_img

        # For faster debugging, limit the size of data
        if max_dataset_len:
            input_img_paths = input_img_paths[:max_dataset_len]
            target_img_paths = target_img_paths[:max_dataset_len]
        dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
        dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
        return dataset.batch(batch_size)

    def create_tfds(self):
        train_ds = self.get_dataset(
            self.batch_size,
            self.img_size,
            self.paths.get("train_input_img_paths"),
            self.paths.get("train_target_img_paths"),
            max_dataset_len=1000,
        )
        valid_ds = self.get_dataset(
            self.batch_size,
            self.img_size,
            self.paths.get("val_input_img_paths"),
            self.paths.get("val_target_img_paths")
        )
        self.data = (
            train_ds.prefetch(buffer_size=self.batch_size),
            valid_ds.prefetch(buffer_size=self.batch_size)
        )

    @classmethod
    def read(
            cls,
            params: Params,
            **kwargs
    ):
        print( Paths.parent_dir / "images.tar.gz")
        assert (
                Paths.parent_dir / "images.tar.gz"
        ).exists(),  "pls download https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz"
        assert (
                Paths.parent_dir / "annotations.tar.gz"
        ).exists(),  "pls download https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz"
        if not (Paths.parent_dir / "images.tar.gz").exists():
            os.system("tar -xf images.tar.gz")
        if not (Paths.parent_dir / "annotations.tar.gz").exists():
            os.system("tar -xf annotations.tar.gz")
        _cls = AnnotationsData(params)
        _cls.get_paths()
        _cls.create_tfds()
        return _cls

