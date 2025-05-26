import sys

sys.path.append(".")
sys.path.append("..")

from torch.utils.data import Dataset
import numpy as np
import albumentations as albu
import cv2
from PIL import Image
import rasterio
import random
from glob import glob
from configs.config import Config
from constants import DEPTH_MASK_SCALE_VALUE, DEPTH, IMAGE
import torch
import matplotlib.pyplot as plt
from typing import List

CONF = Config.from_yaml("configs/config.yaml")
DATA_CONF = CONF.data_config


MAIN_FOLDER = DATA_CONF.data_path
DATA_FOLDER_V = MAIN_FOLDER.format("Vaihingen/og_data/RGB/area{}.tif")
LABEL_FOLDER_V = MAIN_FOLDER.format("Vaihingen/og_data/NDSM/area{}.jpg")
DATA_FOLDER_DFC = MAIN_FOLDER.format("DFC2018/og_data/RGB/UH_NAD83_{}.tif")
DFC_DSM_FOLDER = MAIN_FOLDER.format("DFC2018/og_data/DSM/UH_NAD83_{}.tif")
DFC_DEM_FOLDER = MAIN_FOLDER.format("DFC2018/og_data/DEM/UH_NAD83_{}.tif")
DATA_FOLDER_P = MAIN_FOLDER.format("Potsdam/og_data/RGB/{}_RGB.tif")
LABEL_FOLDER_P = MAIN_FOLDER.format("Potsdam/og_data/DSM/{}_DSM.tif")
DEM_FOLDER_P = MAIN_FOLDER.format("Potsdam/og_data/DEM/{}_DEM.tif")



# Scale factor for Vaihingen dataset depth masks
VAIHIGEN_SCALE = 20.0

# Scale factor for depth masks
SCALE_DEPTH_VALUE = 128.0

DFC_SCALE = 1


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    image = np.transpose(image, (1, 2, 0))
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()
    plt.close()


def read_image(path: str) -> np.ndarray:
    """
    Read an image from the given path.

    Args:
        path (str): The path to the image.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    return np.array(Image.open(path))


def read_depth_image(path: str) -> np.ndarray:
    """
    Read a depth image from the given path.

    Args:
        path (str): The path to the depth image.

    Returns:
        np.ndarray: The depth image as a numpy array.
    """
    arr = read_image(path)
    mask = arr == 0
    f_img = (arr - 1)
    f_img[mask] = 0
    return np.float32(f_img)


def get_random_pos(img, window_shape):
    """Extract of 2D random patch of shape window_shape in the image"""
    diff_x = img.shape[0] - window_shape[0] - 1
    diff_y = img.shape[1] - window_shape[1] - 1
    x1 = random.randint(0, diff_x)
    y1 = random.randint(0, diff_y)
    return x1, x1 + window_shape[0], y1, y1 + window_shape[1]


def get_crop(dataset_name, img, depth):
    if dataset_name == 'DFC2018':
        window_size = (int(DATA_CONF.crop_size[0] * DFC_SCALE),
                       int(DATA_CONF.crop_size[1] * DFC_SCALE))
        x1, x2, y1, y2 = get_random_pos(img, window_size)
    elif dataset_name == "Vaihingen":
        x1, x2, y1, y2 = get_random_pos(img, DATA_CONF.crop_size)

    return {
        IMAGE: img[x1:x2, y1:y2, :].astype(np.float32),
        DEPTH: depth[x1:x2, y1:y2].astype(np.float32),
    }


def get_ids(dataset_name: str, dataset_mode: str):
    if dataset_name == "DFC2018":
        all_files = sorted(glob(DATA_FOLDER_DFC.replace("{}", "*")))
        all_ids = [
            ("_".join(f.split("_")[4:6])).split(".")[0] for f in all_files  # noqa: WPS221
        ]
    elif dataset_name == "Vaihingen":
        all_files = sorted(glob(DATA_FOLDER_V.replace("{}", "*")))
        all_ids = [
            f.split("area")[1].split(".tif")[0] for f in all_files
        ]  # noqa: WPS221

    # Random tile numbers for train/test split
    len_train_sample = int(CONF.data_config.train_size * len(all_ids))
    train_ids = all_ids[:len_train_sample]
    test_ids = all_ids[len_train_sample:]

    if dataset_mode == "train":
        return train_ids

    return test_ids

def split(all_files: List, dataset_mode: str):
    len_train_sample = int(CONF.data_config.train_size * len(all_files))
    train_files = random.sample(all_files, len_train_sample)
    test_files = list(set(all_files) - set(train_files))

    if dataset_mode == "train":
        return train_files

    return test_files


class Vaihingen(Dataset):
    def __init__(
        self,
        dataset_mode: str,
        transform_func: callable,
    ):
        """
        Initialize the Vaighingen.

        Args:
            dataset_mode (str): The mode of the dataset, either 'train' or 'test'.
            transform_func (callable): The transform function to be applied to the data.
            debug (bool, optional): Whether to run the dataset in debug mode. Defaults to False.
            max_files (int, optional): The maximum number of files to include in the dataset. Defaults to None.
        """
        self.root_data_folder = "../data"
        self.dataset_mode = dataset_mode
        self.dataset_name = 'Vaihingen'
        self.transform_func = transform_func
        self.ids = get_ids(self.dataset_name, dataset_mode)
        self._data_files = [DATA_FOLDER_V.format(idx) for idx in self.ids]
        self._label_files = [LABEL_FOLDER_V.format(idx) for idx in self.ids]

        if self.dataset_mode == "train":
            self.both_aug_transform = albu.Compose(
                [
                    albu.HorizontalFlip(),
                    albu.VerticalFlip(),
                ],
                p=1,
                is_check_shapes=False,
            )
            
            self.img_aug_transform = albu.Compose(
                [
                    albu.GaussNoise(var_limit=(0.0, 0.05), p=0.3),
                ],
                p=0.2,
            )
        else:
            self.both_aug_transform = None
            self.img_aug_transform = None

    def __transform_depth_mask__(self, arr):
        mask = arr == 0
        f_img = (arr - 1)
        f_img[mask] = 0

        return np.float32(f_img)

    def __aug_img__(self, croped_image_dict):
        if self.both_aug_transform is not None:
            both_augs = self.both_aug_transform(
                image=croped_image_dict[IMAGE],
                mask=croped_image_dict[DEPTH],
            )
            
            auged_img = self.img_aug_transform(image=both_augs[IMAGE])[IMAGE]
            auged_depth = both_augs['mask']
            
            return self.transform_func(
                {
                    IMAGE: auged_img,
                    DEPTH: auged_depth,
                    },
                )
        return self.transform_func(
            {
                IMAGE: croped_image_dict[IMAGE],
                DEPTH: croped_image_dict[DEPTH],
                },
        )

    def __getitem__(self, index: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            tuple: The image and depth.
        """
        random_idx = random.randint(0, len(self._data_files) - 1)
        img = Image.open(self._data_files[random_idx]).convert("RGB")
        img = np.array(img) / DEPTH_MASK_SCALE_VALUE
        arr = Image.open(self._label_files[random_idx])
        arr = np.array(arr) / 10

        croped_image_dict = get_crop(self.dataset_name, img, self.__transform_depth_mask__(arr))

        t_image = self.__aug_img__(croped_image_dict)

        t_image[IMAGE] = torch.from_numpy(t_image[IMAGE])
        t_image[DEPTH] = torch.from_numpy(t_image[DEPTH])

        return t_image[IMAGE], t_image[DEPTH]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        if self.dataset_mode == 'train':
            return 2000
        else:
            return 200

class DFC2018(Dataset):
    def __init__(
        self,
        dataset_mode: str,
        transform_func: callable,
    ):
        self.root_data_folder = "../data"
        self.dataset_mode = dataset_mode
        self.dataset_name = 'DFC2018'
        self.transform_func = transform_func
        self.ids = get_ids(self.dataset_name, dataset_mode)
        self._data_files = [DATA_FOLDER_DFC.format(idx) for idx in self.ids]
        self._dsm_files = [DFC_DSM_FOLDER.format(idx) for idx in self.ids]
        self._dem_files = [DFC_DEM_FOLDER.format(idx) for idx in self.ids]

        if self.dataset_mode == "train":
            self.both_aug_transform = albu.Compose(
                [
                    albu.HorizontalFlip(),
                    albu.VerticalFlip(),
                    albu.RandomRotate90(),
                ],
                p=1,
                is_check_shapes=False,
            )
            
            self.img_aug_transform = albu.Compose(
                [
                    albu.GaussNoise(var_limit=(0.0, 0.01), p=0.1),
                ],
                p=0.1,
            )
        else:
            self.both_aug_transform = None
            self.img_aug_transform = None

    def __read_image__(self, random_idx):
        rgb_img = Image.open(self._data_files[random_idx]).convert("RGB")
        rgb_img = np.array(rgb_img) / 255.
        
        with rasterio.open(self._dsm_files[random_idx]) as dsm_src:
            dsm_img = dsm_src.read(1)
        with rasterio.open(self._dem_files[random_idx]) as dem_src:
            dem_img = dem_src.read(1)

        return {
            IMAGE: rgb_img,
            DEPTH: cv2.resize(
                dsm_img - dem_img,
                (rgb_img.shape[1], rgb_img.shape[0]),
            ),
        }

    def __aug_img__(self, croped_image_dict):
        if self.both_aug_transform is not None:
            both_augs = self.both_aug_transform(
                image=croped_image_dict[IMAGE],
                mask=croped_image_dict[DEPTH],
            )
            
            auged_img = self.img_aug_transform(image=both_augs[IMAGE])[IMAGE]
            auged_depth = both_augs['mask']
            
            return self.transform_func(
                {
                    IMAGE: auged_img,
                    DEPTH: auged_depth,
                    },
                )
        return self.transform_func(
            {
                IMAGE: croped_image_dict[IMAGE],
                DEPTH: croped_image_dict[DEPTH],
                },
        )

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self._data_files) - 1)
        image_dct = self.__read_image__(random_idx)

        croped_image_dict = get_crop(self.dataset_name, image_dct[IMAGE], image_dct[DEPTH])

        t_image = self.__aug_img__(croped_image_dict)
        
        t_image[IMAGE] = t_image[IMAGE].reshape(
            1,
            t_image[IMAGE].shape[0],
            t_image[IMAGE].shape[1],
            t_image[IMAGE].shape[2],
        )

        t_image[DEPTH] = t_image[DEPTH].reshape(
            1,
            1,
            t_image[DEPTH].shape[0],
            t_image[DEPTH].shape[1],
        )

        t_image[IMAGE] = torch.nn.functional.interpolate(
            torch.Tensor(t_image[IMAGE]),
            size=tuple(DATA_CONF.crop_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        t_image[DEPTH] = torch.nn.functional.interpolate(
            torch.Tensor(t_image[DEPTH]),
            size=tuple(DATA_CONF.crop_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        t_image[DEPTH] = t_image[DEPTH] * DFC_SCALE
        
        return t_image[IMAGE], t_image[DEPTH]

    def __len__(self):
        if self.dataset_mode == 'train':
            return 1000
        else:
            return 100
    
class Potsdam(Dataset):
    def __init__(
        self,
        dataset_mode: str,
        transform_func: callable,
    ):
        """
        Initialize the Potsdam dataset.

        Args:
            dataset_mode (str): The mode of the dataset, either 'train' or 'test'.
            transform_func (callable): The transform function to be applied to the data.
        """
        self.root_data_folder = "../data"
        self.dataset_mode = dataset_mode
        self.dataset_name = 'Potsdam'
        self.transform_func = transform_func
        self.ids = self._get_potsdam_ids(dataset_mode)
        self._data_files = [DATA_FOLDER_P.format(idx) for idx in self.ids]
        self._dsm_files = [LABEL_FOLDER_P.format(idx) for idx in self.ids]
        self._dem_files = [DEM_FOLDER_P.format(idx) for idx in self.ids]

        if self.dataset_mode == "train":
            self.both_aug_transform = albu.Compose(
                [
                    albu.HorizontalFlip(),
                    albu.VerticalFlip(),
                    albu.RandomRotate90(),
                ],
                p=1,
                is_check_shapes=False,
            )
            
            self.img_aug_transform = albu.Compose(
                [
                    albu.GaussNoise(var_limit=(0.0, 0.02), p=0.2),
                    albu.RandomBrightnessContrast(
                        brightness_limit=0.1, 
                        contrast_limit=0.1, 
                        p=0.3
                    ),
                ],
                p=0.3,
            )
        else:
            self.both_aug_transform = None
            self.img_aug_transform = None

    def _get_potsdam_ids(self, dataset_mode: str):
        """Get Potsdam dataset IDs for train/test split."""
        all_files = sorted(glob(DATA_FOLDER_P.replace("{}", "*")))
        all_ids = [
            f.split("/")[-1].split("_RGB.tif")[0] for f in all_files
        ]
        
        # Random tile numbers for train/test split
        len_train_sample = int(CONF.data_config.train_size * len(all_ids))
        random.shuffle(all_ids)
        train_ids = all_ids[:len_train_sample]
        test_ids = all_ids[len_train_sample:]

        if dataset_mode == "train":
            return train_ids
        return test_ids

    def __read_image__(self, random_idx):
        """Read RGB image and corresponding height data."""
        rgb_img = Image.open(self._data_files[random_idx]).convert("RGB")
        rgb_img = np.array(rgb_img) / 255.0
        
        with rasterio.open(self._dsm_files[random_idx]) as dsm_src:
            dsm_img = dsm_src.read(1).astype(np.float32)
        
        try:
            with rasterio.open(self._dem_files[random_idx]) as dem_src:
                dem_img = dem_src.read(1).astype(np.float32)
        except FileNotFoundError:
            dem_img = np.zeros_like(dsm_img)
        
        ndsm = dsm_img - dem_img
        
        if ndsm.shape != rgb_img.shape[:2]:
            ndsm = cv2.resize(
                ndsm,
                (rgb_img.shape[1], rgb_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return {
            IMAGE: rgb_img,
            DEPTH: ndsm,
        }

    def __process_depth_mask__(self, depth_arr):
        """Process depth mask to handle invalid values."""
        mask_invalid = ~np.isfinite(depth_arr)
        depth_arr[mask_invalid] = 0.0
        
        depth_arr = np.clip(depth_arr, 0, 100)
        
        return depth_arr.astype(np.float32)

    def __get_crop__(self, img, depth):
        """Get random crop from image and depth."""
        x1, x2, y1, y2 = get_random_pos(img, DATA_CONF.crop_size)
        
        return {
            IMAGE: img[x1:x2, y1:y2, :].astype(np.float32),
            DEPTH: depth[x1:x2, y1:y2].astype(np.float32),
        }

    def __aug_img__(self, cropped_image_dict):
        """Apply augmentations to image and depth."""
        if self.both_aug_transform is not None:
            both_augs = self.both_aug_transform(
                image=cropped_image_dict[IMAGE],
                mask=cropped_image_dict[DEPTH],
            )
            
            auged_img = self.img_aug_transform(image=both_augs[IMAGE])[IMAGE]
            auged_depth = both_augs['mask']
            
            return self.transform_func(
                {
                    IMAGE: auged_img,
                    DEPTH: auged_depth,
                }
            )
        
        return self.transform_func(
            {
                IMAGE: cropped_image_dict[IMAGE],
                DEPTH: cropped_image_dict[DEPTH],
            }
        )

    def __getitem__(self, index: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            tuple: The image and depth tensors.
        """
        random_idx = random.randint(0, len(self._data_files) - 1)
        
        image_dict = self.__read_image__(random_idx)
        
        processed_depth = self.__process_depth_mask__(image_dict[DEPTH])
        
        cropped_image_dict = self.__get_crop__(image_dict[IMAGE], processed_depth)
        
        transformed_image = self.__aug_img__(cropped_image_dict)
    
        transformed_image[IMAGE] = torch.from_numpy(transformed_image[IMAGE])
        transformed_image[DEPTH] = torch.from_numpy(transformed_image[DEPTH])
        
        return transformed_image[IMAGE], transformed_image[DEPTH]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        if self.dataset_mode == 'train':
            return 1500
        else:
            return 300 
