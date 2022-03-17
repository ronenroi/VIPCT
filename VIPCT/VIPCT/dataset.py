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

DEFAULT_DATA_ROOT = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/roi' \
if not socket.gethostname()=='visl-25u' else '/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256'

    # os.path.join(
    # os.path.dirname(os.path.realpath(__file__)), "../../..", "data/data"
# )

# DEFAULT_URL_ROOT = "https://dl.fbaipublicfiles.com/pytorch3d_nerf_data"

ALL_DATASETS = ("satellites_images","dom_satellites_images")


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
    data_root = os.path.join(data_root,dataset_name)
    image_size = cfg.data.image_size
    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
    data_train_paths = [f for f in glob.glob(os.path.join(data_root, "train/cloud*.pkl"))]
    train_len = cfg.data.n_training if cfg.data.n_training>0 else len(data_train_paths)
    data_train_paths = data_train_paths[:train_len]

    n_cam = cfg.data.n_cam
    rand_cam = cfg.data.rand_cam
    train_dataset = CloudDataset(
            data_train_paths,
        n_cam=n_cam,
        rand_cam = rand_cam,
        mask_type=cfg.ct_net.mask_type
    )


    val_paths = [f for f in glob.glob(os.path.join(data_root, "val/cloud*.pkl"))]
    val_len = cfg.data.n_val if cfg.data.n_val>0 else len(val_paths)
    val_paths = val_paths[:val_len]
    val_dataset = CloudDataset(val_paths, n_cam=n_cam,
        rand_cam = rand_cam, mask_type=cfg.ct_net.val_mask_type)


    return train_dataset, val_dataset, n_cam


class CloudDataset(Dataset):
    def __init__(self, cloud_dir, n_cam, rand_cam=False, transform=None, target_transform=None, mask_type=None):
        self.cloud_dir = cloud_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.rand_cam = rand_cam

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        cloud_path = self.cloud_dir[idx]
        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        if self.rand_cam:
            cam_i = torch.randperm(data['images'].shape[0])[:self.n_cam]
        else:
            cam_i = torch.arange(self.n_cam)
        images = data['images'][cam_i]
        grid = data['net_grid']
        image_sizes = data['image_sizes'][cam_i]
        extinction = data['ext']
        camera_center = data['cameras_pos'][cam_i]
        projection_matrix = data['cameras_P'][cam_i]
        mask = None
        if self.mask_type == 'space_carving':
            mask = data['mask']
        # train_ext.append(train_data['ext'])

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return images, extinction, grid, image_sizes, projection_matrix, camera_center, mask