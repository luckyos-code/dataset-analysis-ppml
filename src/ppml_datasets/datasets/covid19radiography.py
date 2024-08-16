import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import gdown
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import get_img


class Covid19RadiographyDataset(AbstractDataset):
    def __init__(
        self,
        model_img_shape: Tuple[int, int, int],
        builds_ds_info: bool = False,
        preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
        batch_size: int = 32,
        augment_train: bool = True,
        dataset_path: str = "data",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.05, 0.15),
    ):
        """Initialize the Covid19 dataset from AbstractDataset class."""
        super().__init__(
            tfds_name = None,
            dataset_name="covid",#19-radiography",
            dataset_path=dataset_path,
            dataset_img_shape=(299, 299, 3),
            num_classes=2,
            random_seed=42,
            model_img_shape=model_img_shape,
            batch_size=batch_size,
            convert_to_rgb=False,
            augment_train=augment_train,
            preprocessing_function=preprocessing_func,
            shuffle=True,
            is_tfds_ds=False,
            builds_ds_info=builds_ds_info,
        )

        self.train_val_test_split = train_val_test_split

        variants: List[Dict[str, Optional[str]]] = [
            {"activation": "relu", "pretraining": None},
            {"activation": "relu", "pretraining": "imagenet"},
            {"activation": "relu", "pretraining": "pneumonia"},
            {"activation": "tanh", "pretraining": None},
            {"activation": "tanh", "pretraining": "imagenet"},
            {"activation": "tanh", "pretraining": "pneumonia"},
        ]
        self.variants = variants

        if self.dataset_path:
            dataset_path = self.dataset_path

        self.base_imgpath: str = os.path.join(
            dataset_path, "COVID-19_Radiography_Dataset"
        )
        self.normal_imgpath: str = os.path.join(self.base_imgpath, "Normal/images")
        self.covid_imgpath: str = os.path.join(self.base_imgpath, "COVID/images")
        self.imbalance_ratio: float = 1.5

        self.train_attack_data: tf.data.Dataset | None = None
        self.test_attack_data: tf.data.Dataset | None = None

    def _load_dataset(self):
        """Retrieve covid19 dataset.

        Overwrites the _load_dataset functionality of the base class, since we cannot load the dataset from tfds here.
        """
        AUTOTUNE = tf.data.AUTOTUNE
        np.random.seed(self.random_seed)

        if not os.path.exists(self.base_imgpath):
            url = "https://drive.google.com/uc?id=1ZMgUQkwNqvMrZ8QaQmSbiDqXOWAewwou&confirm=t"
            output = os.path.join(
                self.base_imgpath, "COVID-19_Radiography_Database.zip"
            )
            gdown.cached_download(
                url,
                output,
                quiet=False,
                use_cookies=False,
                postprocess=gdown.extractall,
            )
            os.remove(output)

        # excluding duplicates normal images from pneumonia pre-training
        excl_imgs = ["Normal-" + str(i) + ".png" for i in range(8852, 10192 + 1)]

        # collect from image paths
        normal_images = [
            os.path.join(self.normal_imgpath, name)
            for name in os.listdir(self.normal_imgpath)
            if os.path.isfile(os.path.join(self.normal_imgpath, name))
            and name not in excl_imgs
        ]
        covid_images = [
            os.path.join(self.covid_imgpath, name)
            for name in os.listdir(self.covid_imgpath)
            if os.path.isfile(os.path.join(self.covid_imgpath, name))
        ]

        # create train-test split
        label_encoding = ["normal", "COVID-19"]  # normal = 0, COVID-19 = 1
        files, labels = [], []

        np.random.shuffle(covid_images)
        files.extend(covid_images)
        labels.extend(np.full(len(covid_images), label_encoding.index("COVID-19")))

        np.random.shuffle(normal_images)
        if self.imbalance_ratio:
            normal_images = normal_images[
                : int(self.imbalance_ratio * len(covid_images))
            ]
        files.extend(normal_images)
        labels.extend(np.full(len(normal_images), label_encoding.index("normal")))

        files, labels = np.array(files), np.array(labels)

        val_split = self.train_val_test_split[1]
        test_split = self.train_val_test_split[2]
        x_train, x_rest, y_train, y_rest = train_test_split(
            files,
            labels,
            test_size=val_split + test_split,
            random_state=self.random_seed,
        )
        x_test, x_val, y_test, y_val = train_test_split(
            x_rest,
            y_rest,
            test_size=val_split / (val_split + test_split),
            random_state=self.random_seed,
        )

        # build tensorflow dataset
        train_files = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_files = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_files = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # image retrieval
        self.ds_train = train_files.map(get_img, num_parallel_calls=AUTOTUNE)
        self.ds_val = val_files.map(get_img, num_parallel_calls=AUTOTUNE)
        self.ds_test = test_files.map(get_img, num_parallel_calls=AUTOTUNE)

def build_covid(
    model_input_shape: Tuple[int, int, int],
    batch_size: int,
    mods: Dict[str, List[Any]],
    augment_train: bool = False,
    builds_ds_info: bool = False,
) -> AbstractDataset:
    ds = Covid19RadiographyDataset(
        model_img_shape=model_input_shape,
        builds_ds_info=False,
        batch_size=batch_size,
        augment_train=False,
    )
    ds.load_dataset()

    if mods:
        print(
            "Cannot use mods for Covid19RadiographyDataset!"
        )
        sys.exit(1)

    return ds