# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os, glob
from typing import List, Optional, Tuple
import pickle
from .cameras import PerspectiveCameras
from .volumes import Volumes
import numpy as np
import torch
from torch.utils.data import Dataset
import socket
# print(socket.gethostname())
import random

# DEFAULT_DATA_ROOT = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/roi' \
# if not socket.gethostname()=='visl-25u' else '/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256'


DEFAULT_DATA_ROOT = '/home/roironen/Data' \
if not socket.gethostname()=='visl-25u' else '/media/roironen/8AAE21F5AE21DB09/Data'


    # os.path.join(
    # os.path.dirname(os.path.realpath(__file__)), "../../..", "data/data"
# )

# DEFAULT_URL_ROOT = "https://dl.fbaipublicfiles.com/pytorch3d_nerf_data"

ALL_DATASETS = ("BOMEX_10cams_polarization", "BOMEX_CASS_10cams", "CASS_10cams", "CASS_10cams_50m", "BOMEX_10cams",
                "BOMEX_10cams_50m", "BOMEX_32cams", "BOMEX_32cams_50m", "BOMEX_10cams_varying", "BOMEX_10cams_varyingV2",
                "BOMEX_10cams_varyingV3", "BOMEX_10cams_varyingV4",
                "subset_of_seven_clouds")


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch


class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(self, entries: List) -> None:
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(
        self,
    ) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]


def get_cloud_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset, Dataset, int]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    dataset_name = cfg.data.dataset_name

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == 'CASS_10cams':
        data_root = os.path.join(data_root, 'CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields')
        image_size = [236, 236]
    elif dataset_name == 'BOMEX_CASS_10cams':
        data_root_cass = os.path.join(data_root, 'CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields')
        data_root_bomex = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras')
        # image_size = [236, 236]
    elif dataset_name == 'CASS_10cams_50m':
        data_root = data_root.replace('home', 'wdata')
        data_root = os.path.join(data_root, 'CASS_50m_256x256x139_600CCN/10cameras_50m')
        image_size = [96, 96]
    elif dataset_name == 'BOMEX_10cams':
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras')
        image_size = [116, 116]
    elif dataset_name == 'BOMEX_10cams_varying':
        data_root = data_root.replace('home', 'wdata')
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', 'varying_positions')
        image_size = [116, 116]
    elif dataset_name == 'BOMEX_10cams_varyingV2':
        data_root = data_root.replace('home', 'wdata')
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', 'varying_positionsV2')
        image_size = [116, 116]
    elif dataset_name == 'BOMEX_10cams_varyingV3':
        data_root = data_root.replace('home', 'wdata')
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', 'varying_positionsV3')
        image_size = [116, 116]
    elif dataset_name == 'BOMEX_10cams_varyingV4':
        data_root = data_root.replace('home', 'wdata')
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', 'varying_positionsV4')
        image_size = [116, 116]
    elif dataset_name == 'BOMEX_10cams_50m':
        data_root = data_root.replace('home', 'wdata')
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_50m')
        image_size = [48, 48]
    elif dataset_name == 'BOMEX_32cams_50m':
        data_root = data_root.replace('home', 'wdata')
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '32cameras_50m')
        image_size = [48, 48]
    elif dataset_name == 'BOMEX_32cams':
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '32cameras')
        image_size = [116, 116]
    elif dataset_name == 'subset_of_seven_clouds':
        data_root = data_root.replace('home', 'wdata')
        data_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras_50m')
        image_size = [48, 48]
    else:
        data_root = os.path.join(data_root,'BOMEX_256x256x100_5000CCN_50m_micro_256/roi',dataset_name)
        image_size = [236, 236]
    # image_size = cfg.data.image_size

    if not dataset_name == 'BOMEX_CASS_10cams':
        print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
        data_train_paths = [f for f in glob.glob(os.path.join(data_root, "train/cloud*.pkl"))]
    else:
        print(f"Loading dataset {dataset_name}...")
        data_train_paths = [f for f in glob.glob(os.path.join(data_root_cass, "train/cloud*.pkl"))]
        data_train_paths += [f for f in glob.glob(os.path.join(data_root_bomex, "train/cloud*.pkl"))]
        random.shuffle(data_train_paths)
    train_len = cfg.data.n_training if cfg.data.n_training>0 else len(data_train_paths)
    # for cloud_path in data_train_paths:
    #     try:
    #         with open(cloud_path, 'rb') as f:
    #             data = pickle.load(f)
    #     except:
    #         print(cloud_path)
    # print('DONE')
    # for file in data_train_paths:
    #     with open(file, 'rb') as f:
    #         data = pickle.load(f)
    #     grid = data['grid']
    #     grid[0] -= grid[0][0]
    #     grid[1] -= grid[1][0]
    #     grid[2] -= grid[2][0]
    #     data['grid'] = grid.copy()
    #     grid_net = grid
    #     grid_net[0] += 0.025
    #     grid_net[1] += 0.025
    #     grid_net[2] += 0.02
    #     data['net_grid'] = grid_net
    #     with open(file, 'wb') as f:
    #         pickle.dump(data, f)
    data_train_paths = data_train_paths[:train_len]

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    rand_cam = cfg.data.rand_cam
    train_dataset = CloudDataset(
            data_train_paths,
        n_cam=n_cam,
        rand_cam = rand_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
    dataset_name = dataset_name,

    )
    if not (dataset_name == 'BOMEX_CASS_10cams' or dataset_name == 'subset_of_seven_clouds'):
        val_paths = [f for f in glob.glob(os.path.join(data_root, "test/cloud*.pkl"))]
    elif dataset_name == 'subset_of_seven_clouds':
        val_paths = [f for f in glob.glob(os.path.join('/wdata/roironen/Data/subset_of_seven_clouds', "cloud*.pkl"))]
        print(val_paths)
    else:
        val_paths = [f for f in glob.glob(os.path.join(data_root_cass, "test/cloud*.pkl"))]
        val_paths += [f for f in glob.glob(os.path.join(data_root_bomex, "test/cloud*.pkl"))]
        random.shuffle(val_paths)

    # if dataset_name == 'CASS_10cams':
    #     val_paths = [f for f in glob.glob(os.path.join(data_root, "test/cloud*.pkl"))]
    # elif dataset_name == 'BOMEX_10cams' or dataset_name == 'BOMEX_32cams' or dataset_name == 'BOMEX_10cams_varying' \
    #         or 'BOMEX_10cams_50m':
    #     val_paths = [f for f in glob.glob(os.path.join(data_root, "test/cloud*.pkl"))]
    # elif dataset_name == 'BOMEX_10cams_varyingV2':
    #     val_paths = [f for f in glob.glob(os.path.join(data_root, "test/cloud*.pkl"))]
    # else:
    #     val_paths = [f for f in glob.glob(os.path.join(data_root, "val/cloud*.pkl"))]
    # for file in val_paths:
    #     with open(file, 'rb') as f:
    #         data = pickle.load(f)
    #     grid = data['grid']
    #     grid[0] -= grid[0][0]
    #     grid[1] -= grid[1][0]
    #     grid[2] -= grid[2][0]
    #     data['grid'] = grid.copy()
    #     grid_net = grid
    #     grid_net[0] += 0.025
    #     grid_net[1] += 0.025
    #     grid_net[2] += 0.02
    #     with open(file, 'wb') as f:
    #         pickle.dump(data, f)
    val_len = cfg.data.n_val if cfg.data.n_val>0 else len(val_paths)
    val_paths = val_paths[:val_len]
    val_dataset = CloudDataset(val_paths, n_cam=n_cam,
        rand_cam = rand_cam, mask_type=cfg.ct_net.val_mask_type, mean=mean, std=std,   dataset_name = dataset_name
)

    return train_dataset, val_dataset, n_cam


class CloudDataset(Dataset):
    def __init__(self, cloud_dir, n_cam, rand_cam=False, transform=None, target_transform=None, mask_type=None, mean=0, std=1, dataset_name=''):
        self.cloud_dir = cloud_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.rand_cam = rand_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        cloud_path = self.cloud_dir[idx]

        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        images = data['images']
        mask = None
        if self.mask_type == 'space_carving':
            mask = data['mask']
        elif self.mask_type == 'space_carving_morph':
            mask = data['mask_morph']
        elif self.mask_type == 'space_carving_0.9':
            mask = data['mask0.9']
        if 'varying' in self.dataset_name:
            index = torch.randperm(10)[0]
            cam_i = torch.arange(index,100,10)
            mask = mask[index] if mask is not None else None
        else:
            cam_i = torch.arange(self.n_cam)
        images = images[cam_i]
        images -= self.mean
        images /= self.std

        # grid = data['net_grid']
        if hasattr(data, 'image_sizes'):
            image_sizes = data['image_sizes'][cam_i]
        else:
            image_sizes = [image.shape for image in images]
        extinction = data['ext']
        grid = data['grid']
        camera_center = data['cameras_pos'][cam_i]
        projection_matrix = data['cameras_P'][cam_i]

        # if cloud_path == '/wdata/roironen/Data/subset_of_seven_clouds/cloud_results_BOMEX_13x25x36_28440.pkl':
        #     import scipy.io as sio
        #     images = sio.loadmat('/wdata/roironen/Data/subset_of_seven_clouds/satellites_images/satellites_images_28440.mat')['satellites_images']
        #     images -= self.mean
        #     images /= self.std
        #     extinction = sio.loadmat('/wdata/roironen/Data/subset_of_seven_clouds/lwcs/cloud28440.mat')['GT']
        #     mask = sio.loadmat('/wdata/roironen/Data/subset_of_seven_clouds/masks/mask_28440.mat')['CARVER']

        # train_ext.append(train_data['ext'])

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return images, extinction, grid, image_sizes, projection_matrix, camera_center, mask



def get_cloud_microphysics_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset, Dataset, int]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    dataset_name = cfg.data.dataset_name

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == 'BOMEX_10cams_polarization':
        # data_root = '/wdata/yaelsc/Roi_ICCP22/CloudCT10sat_polarized_LWC_LOW_SC'
        data_root = '/wdata/yaelsc/Roi_ICCP22/'
        image_size = [123, 123]

    if dataset_name == 'BOMEX_10cams_polarization':
        print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
        data_train_paths = [f for f in glob.glob(os.path.join(data_root, "CloudCT10sat_polarized_LWC_NOISE/*.pkl"))]

    train_len = cfg.data.n_training if cfg.data.n_training>0 else len(data_train_paths)

    data_train_paths = data_train_paths[:train_len]

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    rand_cam = cfg.data.rand_cam
    train_dataset = MicrophysicsCloudDataset(
            data_train_paths,
        n_cam=n_cam,
        rand_cam = rand_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
    dataset_name = dataset_name,

    )
    if dataset_name == 'BOMEX_10cams_polarization':
        val_paths =  [f for f in glob.glob(os.path.join(data_root, "CloudCT10sat_polarized_LWC_NOISE_TEST/*.pkl"))]

    val_len = cfg.data.n_val if cfg.data.n_val>0 else len(val_paths)
    val_paths = val_paths[:val_len]
    val_dataset = MicrophysicsCloudDataset(val_paths, n_cam=n_cam,
        rand_cam = rand_cam, mask_type=cfg.ct_net.val_mask_type, mean=mean, std=std,   dataset_name = dataset_name
)

    return train_dataset, val_dataset, n_cam


class MicrophysicsCloudDataset(Dataset):
    def __init__(self, cloud_dir, n_cam, rand_cam=False, transform=None, target_transform=None, mask_type=None, mean=0, std=1, dataset_name=''):
        self.cloud_dir = cloud_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.rand_cam = rand_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        cloud_path = self.cloud_dir[idx]
        if 'TEST' in cloud_path:
            data_root = os.path.join(DEFAULT_DATA_ROOT, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras', 'test')
        else:
            data_root = os.path.join(DEFAULT_DATA_ROOT, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras', 'train')
        image_index = cloud_path.split('satellites_images_')[-1].split('.pkl')[0]
        projection_path = os.path.join(data_root, f"cloud_results_{image_index}.pkl")
        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        images = np.concatenate((data['images'], data['DoLPs'][:, None], data['AoLPs'][:, None]), 1)
        mask = None
        if self.mask_type == 'space_carving':
            mask = data['mask']
        elif self.mask_type == 'space_carving_morph':
            mask = data['mask_morph']
        if 'varying' in self.dataset_name:
            index = torch.randperm(10)[0]
            cam_i = torch.arange(index,100,10)
            mask = mask[index] if mask is not None else None
        else:
            cam_i = torch.arange(self.n_cam)
        images = images[cam_i]
        images -= np.array(self.mean).reshape((1,5,1,1))
        images /= np.array(self.std).reshape((1,5,1,1))

        microphysics = np.array([data['lwc'],data['reff'],data['veff']])
        microphysics = microphysics[:,1:-1,1:-1,:-1]
        mask = mask[1:-1,1:-1,:-1]
        # grid = data['net_grid']
        if hasattr(data, 'image_sizes'):
            image_sizes = data['image_sizes'][cam_i]
        else:
            image_sizes = [image.shape[1:] for image in images]
        with open(projection_path, 'rb') as f:
            data = pickle.load(f)

        grid = data['grid']
        camera_center = data['cameras_pos'][cam_i]
        projection_matrix = data['cameras_P'][cam_i]



        # train_ext.append(train_data['ext'])

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return images, microphysics, grid, image_sizes, projection_matrix, camera_center, mask

ALL_DATASETS_AIRMSPI = ("BOMEX_9cams")

def get_airmspi_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset, Dataset, int]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    dataset_name = cfg.data.dataset_name

    if dataset_name not in ALL_DATASETS_AIRMSPI:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == 'BOMEX_9cams':
        data_root = os.path.join(data_root, '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train')
        image_root = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/AirMSPI/LOW_SC/AIRMSPI_IMAGES_LWC_LOW_SC'
        mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        image_size = [350, 350]
    else:
        NotImplementedError()
    images_mapping_list = []
    for _, map in mapping.items():
        voxels_list = []
        v = map.values()
        voxels = np.array(list(v),dtype=object)
        ctr = 0
        for i, voxel in enumerate(voxels):
            if len(voxel)>0:
                pixels = np.unravel_index(voxel, np.array(image_size))
                mean_px = np.mean(pixels,1)
                voxels_list.append(mean_px)
            else:
                ctr +=1
                voxels_list.append([-100000,-100000])
        images_mapping_list.append(voxels_list)
    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
    image_train_paths = [f for f in glob.glob(os.path.join(image_root, "*.pkl"))]
    cloud_train_path = data_root
    train_len = cfg.data.n_training if cfg.data.n_training>0 else len(image_train_paths)

    image_train_paths = image_train_paths[:train_len]
    # cloud_train_paths = cloud_train_paths[:train_len]
    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    # rand_cam = cfg.data.rand_cam
    train_dataset = AirMSPIDataset(
            cloud_train_path,
        image_train_paths,
        mapping=images_mapping_list,
        n_cam=n_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
    dataset_name = dataset_name,
        drop_index = cfg.data.drop_index
    )



    return train_dataset, train_dataset,  n_cam


class AirMSPIDataset(Dataset):
    def __init__(self, cloud_dir,image_dir, n_cam,mapping,  mask_type=None, mean=0, std=1, dataset_name='', drop_index=-1):
        self.cloud_dir = cloud_dir
        self.mapping = mapping
        self.image_dir = image_dir
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.drop_index = drop_index
        if self.n_cam != 9 and self.drop_index>-1:
            self.mapping.pop(drop_index)

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        image_dir = self.image_dir[idx]
        image_index = image_dir.split('satellites_images_')[-1].split('.pkl')[0]
        cloud_path = os.path.join(self.cloud_dir, f"cloud_results_{image_index}.pkl")


        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        with open(image_dir, 'rb') as f:
            images = pickle.load(f)['images']
        if self.n_cam!=9:
            images = np.delete(images,self.drop_index,0)
        mask = None
        if self.mask_type == 'space_carving':
            mask = data['mask']
        elif self.mask_type == 'space_carving_morph':
            mask = data['mask_morph']
        images -= self.mean
        images /= self.std
        grid = data['grid']
        # grid = data['net_grid']
        # if hasattr(data, 'image_sizes'):
        #     image_sizes = data['image_sizes']
        # else:
        #     image_sizes = [image.shape for image in images]
        extinction = data['ext'] / 10

        images_mapping_list = [ np.array(map)[mask.ravel()] for map in self.mapping]

        return images, extinction, grid, images_mapping_list, mask
