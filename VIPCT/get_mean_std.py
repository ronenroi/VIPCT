import os, glob

import pickle
import matplotlib.pyplot as plt
from VIPCT.volumes import Volumes

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from VIPCT.util.types import Device
from VIPCT.util.renderer_utils import TensorProperties, convert_to_tensors_and_broadcast
from VIPCT.cameras import PerspectiveCameras
from VIPCT.encoder import Backbone
import matplotlib.patches as patches
import matplotlib.backends.backend_pdf
import matplotlib.cm as cm
from scipy.interpolate import griddata
from scipy import interpolate

if __name__ == "__main__":
    def sample_features(latents, uv):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param latent (B, C, H, W) images features
        :param uv (B, N, 2) image points (x,y)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :return (B, C, N) L is latent size
        """
        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = torch.empty(0, device=uv.device)
        for latent in latents:
            samples = torch.cat((samples, torch.squeeze(F.grid_sample(
                latent,
                uv,
                align_corners=True,
                mode='bilinear',
                padding_mode='zeros',
            ))), dim=1)
        return samples  # (Cams,cum_channels, N)


    images = []
    data_train_paths = [f for f in glob.glob('/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/train/cloud_*.pkl')]
    for path in data_train_paths:
        with open(path, 'rb') as outfile:
            x = pickle.load(outfile)

        images.append(x['images'])


    print()