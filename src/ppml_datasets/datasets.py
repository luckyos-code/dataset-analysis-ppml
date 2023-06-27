import gdown
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Callable, Tuple, List, Dict, Optional

from ppml_datasets.abstract_dataset_handler import AbstractDataset, RgbToGrayscale, AbstractDatasetClassSize, AbstractDatasetClassImbalance
from ppml_datasets.utils import get_img


class MnistDataset(AbstractDataset):
    def __init__(self,
                 model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        super().__init__(tfds_name="mnist",
                         dataset_name="mnist",
                         dataset_path=dataset_path,
                         dataset_img_shape=(28, 28, 1),
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=True,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class MnistDatasetClassSize(AbstractDatasetClassSize):
    def __init__(self,
                 ds: MnistDataset,
                 class_size: int):
        self.class_size = class_size
        self.ds = ds
        super().__init__(tfds_name=self.ds.tfds_name,
                         num_classes=self.ds.num_classes,
                         dataset_name=f"{self.ds.dataset_name}_c{class_size}",
                         dataset_path=self.ds.dataset_path,
                         model_img_shape=self.ds.model_img_shape,
                         batch_size=self.ds.batch_size,
                         convert_to_rgb=self.ds.convert_to_rgb,
                         augment_train=self.ds.augment_train,
                         shuffle=self.ds.shuffle,
                         is_tfds_ds=self.ds.is_tfds_ds,
                         builds_ds_info=self.ds.builds_ds_info)


class MnistDatasetClassImbalance(AbstractDatasetClassImbalance):
    def __init__(self,
                 ds: MnistDataset,
                 imbalance_mode: str,
                 imbalance_ratio: float):
        self.imbalance_mode = imbalance_mode
        self.imbalance_ratio = imbalance_ratio
        self.ds = ds
        super().__init__(tfds_name=self.ds.tfds_name,
                         num_classes=self.ds.num_classes,
                         dataset_name=f"{self.ds.dataset_name}_i{self.imbalance_mode}{self.imbalance_ratio}",
                         dataset_path=self.ds.dataset_path,
                         model_img_shape=self.ds.model_img_shape,
                         batch_size=self.ds.batch_size,
                         convert_to_rgb=self.ds.convert_to_rgb,
                         augment_train=self.ds.augment_train,
                         shuffle=self.ds.shuffle,
                         is_tfds_ds=self.ds.is_tfds_ds,
                         builds_ds_info=self.ds.builds_ds_info)


class FashionMnistDataset(AbstractDataset):
    def __init__(self,
                 model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        super().__init__(tfds_name="fashion_mnist",
                         dataset_name="fmnist",
                         dataset_path=dataset_path,
                         dataset_img_shape=(28, 28, 1),
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=True,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class FashionMnistDatasetClassSize(AbstractDatasetClassSize):
    def __init__(self,
                 ds: FashionMnistDataset,
                 class_size: int):
        self.class_size = class_size
        self.ds = ds
        super().__init__(tfds_name=self.ds.tfds_name,
                         num_classes=self.ds.num_classes,
                         dataset_name=f"{self.ds.dataset_name}_c{class_size}",
                         dataset_path=self.ds.dataset_path,
                         model_img_shape=self.ds.model_img_shape,
                         batch_size=self.ds.batch_size,
                         convert_to_rgb=self.ds.convert_to_rgb,
                         augment_train=self.ds.augment_train,
                         shuffle=self.ds.shuffle,
                         is_tfds_ds=self.ds.is_tfds_ds,
                         builds_ds_info=self.ds.builds_ds_info)


class FashionMnistDatasetClassImbalance(AbstractDatasetClassImbalance):
    def __init__(self,
                 ds: FashionMnistDataset,
                 imbalance_mode: str,
                 imbalance_ratio: float):
        self.imbalance_mode = imbalance_mode
        self.imbalance_ratio = imbalance_ratio
        self.ds = ds
        super().__init__(tfds_name=self.ds.tfds_name,
                         num_classes=self.ds.num_classes,
                         dataset_name=f"{self.ds.dataset_name}_i{self.imbalance_mode}{self.imbalance_ratio}",
                         dataset_path=self.ds.dataset_path,
                         model_img_shape=self.ds.model_img_shape,
                         batch_size=self.ds.batch_size,
                         convert_to_rgb=self.ds.convert_to_rgb,
                         augment_train=self.ds.augment_train,
                         shuffle=self.ds.shuffle,
                         is_tfds_ds=self.ds.is_tfds_ds,
                         builds_ds_info=self.ds.builds_ds_info)


class Cifar10Dataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        """Initialize the CIFAR10 dataset from AbstractDataset class."""
        super().__init__(tfds_name="cifar10",
                         dataset_name="cifar10",
                         dataset_path=dataset_path,
                         dataset_img_shape=(32, 32, 3),
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class Cifar10Dataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 class_size: int,
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        """Initialize the CIFAR10 dataset from AbstractDataset class."""
        super().__init__(tfds_name="cifar10",
                         dataset_name="cifar10",
                         dataset_path=dataset_path,
                         dataset_img_shape=(32, 32, 3),
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)
        self.class_size = class_size

    def _load_dataset(self):
        print(f"Creating Cifar10 dataset with custom class size of {self.class_size}")
        # load default cifar10 from tfds
        self._load_from_tfds()
        # shuffle ds before reducing class size
        self.ds_train = self.ds_train.shuffle(
            buffer_size=self.ds_train.cardinality().numpy(), seed=self.random_seed)
        self.reduce_samples_per_class_train_ds(self.class_size)


class Cifar10DatasetGray(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        """Initialize the CIFAR10 dataset from AbstractDataset class."""
        super().__init__(dataset_name="cifar10_gray",
                         tfds_name="cifar10",
                         dataset_path=dataset_path,
                         dataset_img_shape=(32, 32, 1),
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=True,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)

    def _load_dataset(self):

        # load default cifar10 from tfds
        self._load_from_tfds()

        to_grayscale = tf.keras.Sequential([
            RgbToGrayscale()
        ])

        print("Creating cifar10gray")
        self.ds_train = self.ds_train.map(
            lambda x, y: (to_grayscale(x, training=True), y))
        self.ds_test = self.ds_test.map(
            lambda x, y: (to_grayscale(x, training=True), y))


class Cifar100Dataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        """Initialize the CIFAR100 dataset from AbstractDataset class."""
        super().__init__(dataset_name="cifar100",
                         tfds_name="cifar100",
                         dataset_path=dataset_path,
                         dataset_img_shape=(32, 32, 3),
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True, is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class ImagenetteDataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None):
        """Initialize the full size v2 imagenette dataset from AbstractDataset class."""
        super().__init__(dataset_name="imagenette/full-size-v2",
                         tfds_name="imagenette/full-size-v2",
                         dataset_path="data",
                         dataset_img_shape=(None, None, 3),
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         train_val_test_split=(1, 1, 0),
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=True,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class Covid19RadiographyDataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 dataset_path: str,
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None):
        """Initialize the Covid19 dataset from AbstractDataset class."""
        super().__init__(dataset_name="covid19-radiography",
                         dataset_path=dataset_path,
                         dataset_img_shape=(299, 299, 3),
                         num_classes=2,
                         model_img_shape=model_img_shape,
                         train_val_test_split=(0.8, 0.05, 0.15),
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=True,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=False,
                         builds_ds_info=builds_ds_info)

        variants: List[Dict[str, Optional[str]]] = [
            {'activation': 'relu', 'pretraining': None},
            {'activation': 'relu', 'pretraining': 'imagenet'},
            {'activation': 'relu', 'pretraining': 'pneumonia'},
            {'activation': 'tanh', 'pretraining': None},
            {'activation': 'tanh', 'pretraining': 'imagenet'},
            {'activation': 'tanh', 'pretraining': 'pneumonia'},
        ]
        self.variants = variants

        if self.dataset_path:
            dataset_path = self.dataset_path

        self.base_imgpath: str = os.path.join(dataset_path, "COVID-19_Radiography_Dataset")
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

        if not os.path.exists(self.base_imgpath):
            url = 'https://drive.google.com/uc?id=1ZMgUQkwNqvMrZ8QaQmSbiDqXOWAewwou&confirm=t'
            output = os.path.join(self.base_imgpath, 'COVID-19_Radiography_Database.zip')
            gdown.cached_download(url, output, quiet=False, use_cookies=False,
                                  postprocess=gdown.extractall)
            os.remove(output)

        # excluding duplicates normal images from pneumonia pre-training
        excl_imgs = ['Normal-' + str(i) + '.png' for i in range(8852, 10192 + 1)]

        # collect from image paths
        normal_images = [os.path.join(self.normal_imgpath, name) for name in os.listdir(
            self.normal_imgpath) if os.path.isfile(os.path.join(self.normal_imgpath, name)) and name not in excl_imgs]
        covid_images = [os.path.join(self.covid_imgpath, name) for name in os.listdir(
            self.covid_imgpath) if os.path.isfile(os.path.join(self.covid_imgpath, name))]

        # create train-test split
        label_encoding = ['normal', 'COVID-19']  # normal = 0, COVID-19 = 1
        files, labels = [], []

        np.random.shuffle(covid_images)
        files.extend(covid_images)
        labels.extend(np.full(len(covid_images), label_encoding.index('COVID-19')))

        np.random.shuffle(normal_images)
        if self.imbalance_ratio:
            normal_images = normal_images[:int(self.imbalance_ratio * len(covid_images))]
        files.extend(normal_images)
        labels.extend(np.full(len(normal_images), label_encoding.index('normal')))

        files, labels = np.array(files), np.array(labels)

        val_split = self.train_val_test_split[1]
        test_split = self.train_val_test_split[2]
        x_train, x_rest, y_train, y_rest = train_test_split(
            files, labels, test_size=val_split + test_split, random_state=self.random_seed)
        x_test, x_val, y_test, y_val = train_test_split(
            x_rest, y_rest, test_size=val_split / (val_split + test_split), random_state=self.random_seed)

        # build tensorflow dataset
        train_files = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_files = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_files = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # image retrieval
        self.ds_train = train_files.map(get_img, num_parallel_calls=AUTOTUNE)
        self.ds_val = val_files.map(get_img, num_parallel_calls=AUTOTUNE)
        self.ds_test = test_files.map(get_img, num_parallel_calls=AUTOTUNE)
