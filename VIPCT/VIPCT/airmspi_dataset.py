# This file contains the code for real-world AirMSPI cloud dataset loaders for VIP-CT.
# It is based on PyTorch3D source code ('https://github.com/facebookresearch/pytorch3d') by FAIR
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Roi Ronen, Vadim Holodovsky and Yoav. Y. Schechner, "Variable Imaging Projection Cloud Scattering Tomography",
# Proc. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.


import os, glob
from typing import Tuple
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import socket
import random
import scipy.io as sio

DEFAULT_DATA_ROOT = '/home/roironen/Data' \
if not socket.gethostname() == 'visl-25u' else '/media/roironen/8AAE21F5AE21DB09/Data'

def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch

ALL_DATASETS_AIRMSPI = ("BOMEX_aux_9cams", "32N123W_experiment_cloud1", "32N123W_experiment_cloud2", "18S8E_experiment" )
#
# def get_airmspi_datasets(
#     cfg,
#     data_root: str = DEFAULT_DATA_ROOT,
# ) -> Tuple[Dataset, Dataset, int]:
#     """
#     Obtains the training and validation dataset object for a dataset specified
#     with the `dataset_name` argument.
#
#     Args:
#         dataset_name: The name of the dataset to load.
#         image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
#         data_root: The root folder at which the data is stored.
#
#     Returns:
#         train_dataset: The training dataset object.
#         val_dataset: The validation dataset object.
#         test_dataset: The testing dataset object.
#     """
#     dataset_name = cfg.data.dataset_name
#
#     if dataset_name not in ALL_DATASETS_AIRMSPI:
#         raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")
#
#     if dataset_name == 'BOMEX_9cams':
#         data_root = os.path.join(data_root, '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train')
#         image_root = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/AIRMSPI_IMAGES_LWC_LOW_SC_NOISY_PROJ/'
#         mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
#         with open(mapping_path, 'rb') as f:
#             mapping = pickle.load(f)
#         image_size = [350, 350]
#     elif dataset_name == 'BOMEX_9cams_varying':
#         data_root = os.path.join(data_root, '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train')
#         image_root = '/wdata/roironen/Data/AirMSPINEW/'
#         mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
#         with open(mapping_path, 'rb') as f:
#             mapping = pickle.load(f)
#         image_size = [350, 350]
#     else:
#         NotImplementedError()
#     images_mapping_list = []
#     for _, map in mapping.items():
#         voxels_list = []
#         v = map.values()
#         voxels = np.array(list(v),dtype=object)
#         ctr = 0
#         for i, voxel in enumerate(voxels):
#             if len(voxel)>0:
#                 pixels = np.unravel_index(voxel, np.array(image_size))
#                 mean_px = np.mean(pixels,1)
#                 voxels_list.append(mean_px)
#             else:
#                 ctr +=1
#                 voxels_list.append([-100000,-100000])
#         images_mapping_list.append(voxels_list)
#
#     #pixel_centers = sio.loadmat('/wdata/roironen/Data/pixel_centers.mat')['xpc']
#     print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
#     image_train_paths = [f for f in glob.glob(os.path.join(image_root, "*.pkl"))]
#     cloud_train_path = data_root
#     train_len = cfg.data.n_training if cfg.data.n_training>0 else len(image_train_paths)
#
#     image_train_paths = image_train_paths[:train_len]
#     # cloud_train_paths = cloud_train_paths[:train_len]
#     n_cam = cfg.data.n_cam
#     mean = cfg.data.mean
#     std = cfg.data.std
#     # rand_cam = cfg.data.rand_cam
#     train_dataset = AirMSPIDataset(
#             cloud_train_path,
#         image_train_paths,
#         mapping=images_mapping_list,
#         n_cam=n_cam,
#         mask_type=cfg.ct_net.mask_type,
#         mean=mean,
#         std=std,
#     dataset_name = dataset_name,
#         drop_index = cfg.data.drop_index,
#         #pixel_centers=pixel_centers
#     )
#
#
#
#     return train_dataset, train_dataset,  n_cam
#
#
# class AirMSPIDataset(Dataset):
#     def __init__(self, cloud_dir,image_dir, n_cam,mapping, mask_type=None, mean=0, std=1, dataset_name='', drop_index=-1):
#         self.cloud_dir = cloud_dir
#         self.mapping = mapping
#         self.image_dir = image_dir
#         self.mask_type = mask_type
#         self.n_cam = n_cam
#         self.mean = mean
#         self.std = std
#         self.dataset_name = dataset_name
#         self.drop_index = drop_index
#         if self.n_cam != 9 and self.drop_index>-1:
#             self.mapping.pop(drop_index)
#             #self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,0)
#
#     def __len__(self):
#         return len(self.cloud_dir)
#
#     def __getitem__(self, idx):
#         image_dir = self.image_dir[idx]
#         image_index = image_dir.split('satellites_images_')[-1].split('.pkl')[0]
#         cloud_path = os.path.join(self.cloud_dir, f"cloud_results_{image_index}.pkl")
#
#
#         with open(cloud_path, 'rb') as f:
#             data = pickle.load(f)
#         with open(image_dir, 'rb') as f:
#             images = pickle.load(f)['images']
#         if self.n_cam!=9:
#             images = np.delete(images,self.drop_index,0)
#         mask = None
#         if self.mask_type == 'space_carving':
#             mask = data['mask']
#         elif self.mask_type == 'space_carving_morph':
#             mask = data['mask_morph']
#         images -= self.mean
#         images /= self.std
#         grid = data['grid']
#         # grid = data['net_grid']
#         # if hasattr(data, 'image_sizes'):
#         #     image_sizes = data['image_sizes']
#         # else:
#         #     image_sizes = [image.shape for image in images]
#         extinction = data['ext'] / 10
#
#         images_mapping_list = [ np.array(map)[mask.ravel()] for map in self.mapping]
#
#         return images, extinction, grid, images_mapping_list, mask
#
#
#
# def get_airmspi_datasetsV2(
#     cfg,
#     data_root: str = DEFAULT_DATA_ROOT,
# ) -> Tuple[Dataset, Dataset, int]:
#     """
#     Obtains the training and validation dataset object for a dataset specified
#     with the `dataset_name` argument.
#
#     Args:
#         dataset_name: The name of the dataset to load.
#         image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
#         data_root: The root folder at which the data is stored.
#
#     Returns:
#         train_dataset: The training dataset object.
#         val_dataset: The validation dataset object.
#         test_dataset: The testing dataset object.
#     """
#     dataset_name = cfg.data.dataset_name
#
#     if dataset_name not in ALL_DATASETS_AIRMSPI:
#         raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")
#
#     if dataset_name == 'BOMEX_9cams':
#         data_root = os.path.join(data_root, '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train')
#         image_root = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/AirMSPI/LOW_SC/AIRMSPI_IMAGES_LWC_LOW_SC'
#         mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
#         with open(mapping_path, 'rb') as f:
#             mapping = pickle.load(f)
#         image_size = [350, 350]
#     else:
#         NotImplementedError()
#     images_mapping_list = []
#     pixel_centers_list = []
#     pixel_centers = sio.loadmat('/wdata/roironen/Data/pixel_centers.mat')['xpc']
#     camera_ind = 0
#     for _, map in mapping.items():
#         voxels_list = []
#         pixel_list = []
#         v = map.values()
#         voxels = np.array(list(v),dtype=object)
#         for i, voxel in enumerate(voxels):
#             if len(voxel)>0:
#                 pixels = np.unravel_index(voxel, np.array(image_size))
#                 mean_px = np.mean(pixels,1)
#                 voxels_list.append(mean_px)
#                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
#             else:
#                 voxels_list.append([-100000,-100000])
#                 pixel_list.append([-10000, -10000, -10000])
#
#         camera_ind += 1
#         images_mapping_list.append(voxels_list)
#         pixel_centers_list.append(pixel_list)
#
#     print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
#     image_train_paths = [f for f in glob.glob(os.path.join(image_root, "*.pkl"))]
#     cloud_train_path = data_root
#     train_len = cfg.data.n_training if cfg.data.n_training>0 else len(image_train_paths)
#
#     image_train_paths = image_train_paths[:train_len]
#     # cloud_train_paths = cloud_train_paths[:train_len]
#     n_cam = cfg.data.n_cam
#     mean = cfg.data.mean
#     std = cfg.data.std
#     # rand_cam = cfg.data.rand_cam
#     train_dataset = AirMSPIDatasetV2(
#             cloud_train_path,
#         image_train_paths,
#         mapping=images_mapping_list,
#         n_cam=n_cam,
#         mask_type=cfg.ct_net.mask_type,
#         mean=mean,
#         std=std,
#     dataset_name = dataset_name,
#         drop_index = cfg.data.drop_index,
#         pixel_centers=pixel_centers_list
#     )
#
#
#
#     return train_dataset, train_dataset,  n_cam
#
#
# class AirMSPIDatasetV2(Dataset):
#     def __init__(self, cloud_dir,image_dir, n_cam,mapping, pixel_centers, mask_type=None, mean=0, std=1, dataset_name='', drop_index=-1):
#         self.cloud_dir = cloud_dir
#         self.mapping = mapping
#         self.image_dir = image_dir
#         self.mask_type = mask_type
#         self.n_cam = n_cam
#         self.mean = mean
#         self.std = std
#         self.dataset_name = dataset_name
#         self.pixel_centers = pixel_centers
#         self.drop_index = drop_index
#         if self.n_cam != 9 and self.drop_index>-1:
#             self.mapping.pop(drop_index)
#             self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,0)
#
#     def __len__(self):
#         return len(self.cloud_dir)
#
#     def __getitem__(self, idx):
#         image_dir = self.image_dir[idx]
#         image_index = image_dir.split('satellites_images_')[-1].split('.pkl')[0]
#         cloud_path = os.path.join(self.cloud_dir, f"cloud_results_{image_index}.pkl")
#
#
#         with open(cloud_path, 'rb') as f:
#             data = pickle.load(f)
#         with open(image_dir, 'rb') as f:
#             images = pickle.load(f)['images']
#         if self.n_cam!=9:
#             images = np.delete(images,self.drop_index,0)
#         mask = None
#         if self.mask_type == 'space_carving':
#             mask = data['mask']
#         elif self.mask_type == 'space_carving_morph':
#             mask = data['mask_morph']
#         images -= self.mean
#         images /= self.std
#         grid = data['grid']
#         # grid = data['net_grid']
#         # if hasattr(data, 'image_sizes'):
#         #     image_sizes = data['image_sizes']
#         # else:
#         #     image_sizes = [image.shape for image in images]
#         extinction = data['ext'] / 10
#
#         images_mapping_list = [ np.array(map)[mask.ravel()] for map in self.mapping]
#         pixel_centers = [ np.array(centers)[mask.ravel()] for centers in self.pixel_centers]
#
#
#         return images, extinction, grid, images_mapping_list, pixel_centers, mask
#

def get_airmspi_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset, Dataset]:
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

    if dataset_name == 'BOMEX_aux_9cams':
        data_root = os.path.join(data_root, '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train')
        image_root = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/'
        # mapping_paths = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebat_voxel_pixel_list32x32x32_BOMEX_img350x350_20160826_104727Z_SouthAtlanticOcean-14S19W.pkl'
        mapping_paths = [f for f in glob.glob('/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebat_voxel_pixel_list32x32x32_BOMEX_img350x350*.pkl')]
        pixel_center_paths = [f for f in glob.glob('/wdata/roironen/Data/AirMSPI-Varying/training/*.mat')]
        image_size = [350, 350]
    else:
        NotImplementedError()
    ## building map if necessary
    # images_mapping_lists = []
    # pixel_centers_lists = []
    # for mapping_path, pixel_center_path in zip(mapping_paths, pixel_center_paths):
    #     with open(mapping_path, 'rb') as f:
    #         mapping = pickle.load(f)
    #     images_mapping_list = []
    #     pixel_centers_list = []
    #     pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    #     camera_ind = 0
    #     for _, map in mapping.items():
    #         voxels_list = []
    #         pixel_list = []
    #         v = map.values()
    #         voxels = np.array(list(v),dtype=object)
    #         for i, voxel in enumerate(voxels):
    #             if len(voxel)>0:
    #                 pixels = np.unravel_index(voxel, np.array(image_size))
    #                 mean_px = np.mean(pixels,1)
    #                 voxels_list.append(mean_px)
    #                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #             else:
    #                 voxels_list.append([-100000,-100000])
    #                 pixel_list.append([-10000, -10000, -10000])
    #
    #         camera_ind += 1
    #         images_mapping_list.append(voxels_list)
    #         pixel_centers_list.append(pixel_list)
    #     images_mapping_lists.append((images_mapping_list))
    #     pixel_centers_lists.append(pixel_centers_list)
    # print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
    # with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_images_mapping_lists32x32x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(images_mapping_lists, f, pickle.HIGHEST_PROTOCOL)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_pixel_centers_lists32x32x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(pixel_centers_lists, f, pickle.HIGHEST_PROTOCOL)
    with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_images_mapping_lists32x32x32_BOMEX_img350x350.pkl', 'rb') as f:
        images_mapping_lists = pickle.load(f)
    with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_pixel_centers_lists32x32x32_BOMEX_img350x350.pkl', 'rb') as f:
        pixel_centers_lists = pickle.load(f)
    image_train_paths = [f for f in glob.glob(os.path.join(image_root, "SIMULATED_AIRMSPI_TRAIN*"))]
    image_train_paths = [glob.glob(os.path.join(f, "*.pkl")) for f in image_train_paths]

    cloud_train_path = data_root
    assert cfg.data.n_training <= 0

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    train_dataset = AirMSPIDataset(
            cloud_train_path,
        image_train_paths,
        mapping=images_mapping_lists,
        n_cam=n_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
        drop_index = cfg.data.drop_index,
        pixel_centers=pixel_centers_lists
    )

    return train_dataset, train_dataset

def get_real_world_airmspi_datasets(
    cfg,
) -> Tuple[Dataset]:
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

    if dataset_name == '32N123W_experiment_cloud1':
        image_path = '/wdata/yaelsc/AirMSPI_raw_data/raw_data/croped_airmspi_9images_for_Roi.mat'
        mapping_path = '/wdata/roironen/Data/AirMSPI-Varying/test/rebat_images_mapping_lists72x72x32_BOMEX_img350x350.pkl'
        pixel_centers_path = '/wdata/roironen/Data/AirMSPI-Varying/test/rebat_pixel_centers_lists72x72x32_BOMEX_img350x350.pkl'
        mask_path = '/wdata/roironen/Data/mask_72x72x32_vox50x50x40mROI.mat'
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 72
        ny = 72
        nz = 32
    elif dataset_name == '32N123W_experiment_cloud2':
        image_path = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebatel_iccp22/32N123W_experiment/croped_airmspi_9images_for_Roi.mat'
        mapping_path = '/wdata/roironen/Data/AirMSPI-Varying/test/32N123W_rebat_images_mapping_lists60x60x32_18S8E_img350x350.pkl'
        pixel_centers_path = '/wdata/roironen/Data/AirMSPI-Varying/test/32N123W_rebat_pixel_centers_lists60x60x32_18S8E_img350x350.pkl'
        mask_path = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebatel_iccp22/32N123W_experiment/mask_60x60x32_vox50x50x40m.mat'
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 60
        ny = 60
        nz = 32
    elif dataset_name == '18S8E_experiment':
        image_path = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebatel_iccp22/18S8E_experiment/croped_airmspi_9images_for_Roi.mat'
        mapping_path = '/wdata/roironen/Data/AirMSPI-Varying/test/rebat_images_mapping_lists32x32x32_18S8E_img350x350.pkl'
        pixel_centers_path = '/wdata/roironen/Data/AirMSPI-Varying/test/rebat_pixel_centers_lists32x32x32_18S8E_img350x350.pkl'
        mask_path = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/rebatel_iccp22/18S8E_experiment/mask_52x52x32_vox50x50x40m.mat'
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 52
        ny = 52
        nz = 32

    else:
        NotImplementedError()
    ## building map if necessary
    # images_mapping_lists = []
    # pixel_centers_lists = []
    # for mapping_path, pixel_center_path in zip(mapping_paths, pixel_center_paths):
    #     with open(mapping_path, 'rb') as f:
    #         mapping = pickle.load(f)
    #     images_mapping_list = []
    #     pixel_centers_list = []
    #     pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    #     camera_ind = 0
    #     for _, map in mapping.items():
    #         voxels_list = []
    #         pixel_list = []
    #         v = map.values()
    #         voxels = np.array(list(v),dtype=object)
    #         for i, voxel in enumerate(voxels):
    #             if len(voxel)>0:
    #                 pixels = np.unravel_index(voxel, np.array(image_size))
    #                 mean_px = np.mean(pixels,1)
    #                 voxels_list.append(mean_px)
    #                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #             else:
    #                 voxels_list.append([-100000,-100000])
    #                 pixel_list.append([-10000, -10000, -10000])
    #
    #         camera_ind += 1
    #         images_mapping_list.append(voxels_list)
    #         pixel_centers_list.append(pixel_list)
    #     images_mapping_lists.append((images_mapping_list))
    #     pixel_centers_lists.append(pixel_centers_list)
    # print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
    # with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_images_mapping_lists32x32x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(images_mapping_lists, f, pickle.HIGHEST_PROTOCOL)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_pixel_centers_lists32x32x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(pixel_centers_lists, f, pickle.HIGHEST_PROTOCOL)
    with open(mapping_path, 'rb') as f:
        images_mapping_list = pickle.load(f)
    with open(pixel_centers_path, 'rb') as f:
        pixel_centers_list = pickle.load(f)

    # images_mapping_list = [[np.array(map) for map in images_mapping_list]]
    # pixel_centers_list = [[np.array(centers) for centers in pixel_centers_list]]

    gx = np.linspace(0, dx * nx, nx, dtype=np.float32)
    gy = np.linspace(0, dy * ny, ny, dtype=np.float32)
    gz = np.linspace(0, dz * nz, nz, dtype=np.float32)
    grid = [np.array([gx, gy, gz])]
    mask = sio.loadmat(mask_path)['mask']

    assert cfg.data.n_training <= 0

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    dataset = AirMSPIDataset_test(
        image_dir = image_path,
        mapping=images_mapping_list,
        n_cam=n_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
        drop_index = cfg.data.drop_index,
        pixel_centers=pixel_centers_list,
        mask = mask,
        grid = grid

    )

    return dataset

class AirMSPIDataset(Dataset):
    def __init__(self, cloud_dir,image_dir, n_cam,mapping, pixel_centers, mask_type=None, mean=0, std=1, dataset_name='', drop_index=-1):
        self.cloud_dir = cloud_dir
        self.mapping = mapping
        self.image_dir = image_dir
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.pixel_centers = pixel_centers
        self.drop_index = drop_index
        if self.n_cam != 9 and self.drop_index>-1:
            for map in self.mapping:
                map.pop(drop_index)
            self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,1)

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        geometry_ind = np.random.randint(len(self.image_dir))
        image_dir = self.image_dir[geometry_ind][idx]
        image_index = image_dir.split('satellites_images_')[-1].split('.pkl')[0]
        cloud_path = os.path.join(self.cloud_dir, f"cloud_results_{image_index}.pkl")


        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        with open(image_dir, 'rb') as f:
            images = pickle.load(f)['images']
        if self.n_cam != 9:
            images = np.delete(images, self.drop_index,0)
        mask = None
        if self.mask_type == 'space_carving':
            mask = data['mask']
        elif self.mask_type == 'space_carving_morph':
            mask = data['mask_morph']
        images -= self.mean
        images /= self.std
        grid = data['grid']
        extinction = data['ext'] / 10 # images are of BOMEX_aux dataset while ext of BOMEX

        images_mapping_list = [np.array(map)[mask.ravel()] for map in self.mapping[geometry_ind]]
        pixel_centers = [np.array(centers)[mask.ravel()] for centers in self.pixel_centers[geometry_ind]]

        return images, extinction, grid, images_mapping_list, pixel_centers, mask

class AirMSPIDataset_test(Dataset):
    def __init__(self, image_dir, n_cam, mapping, pixel_centers, mask_type=None, mean=0, std=1, dataset_name='',
                 drop_index=-1, mask = None, grid = None):
        self.mapping = mapping
        self.image_dir = image_dir
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.pixel_centers = pixel_centers
        self.drop_index = drop_index
        self.mask = mask
        self.grid =grid
        if self.n_cam != 9 and self.drop_index>-1:
            self.mapping.pop(self.drop_index)
            self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,0)


    def __getitem__(self, idx):
        images = sio.loadmat(self.image_dir)['croped_airmspi_images']
        if self.n_cam != 9:
            images = np.delete(images, self.drop_index, 0)
        mask = None
        if self.mask_type == 'space_carving':
            mask = self.mask
        else:
            mask = np.ones_like(self.grid)
        images -= self.mean
        images /= self.std
        images_mapping_list = [[np.array(map) for map in self.mapping]]
        pixel_centers_list = [[np.array(centers) for centers in self.pixel_centers]]

        return images, self.grid, images_mapping_list, pixel_centers_list, mask
