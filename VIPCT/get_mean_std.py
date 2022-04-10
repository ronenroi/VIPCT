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


    images = []
    # data_train_paths = [f for f in glob.glob('/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/train/cloud_*.pkl')]
    data_train_paths = [f for f in glob.glob('/wdata/yaelsc/Roi_ICCP22/CloudCT10sat_polarized_LWC_NOISE/*.pkl')]
    for path in data_train_paths:
        with open(path, 'rb') as outfile:
            x = pickle.load(outfile)
        im = np.concatenate((x['images'],x['DoLPs'][:,None],x['AoLPs'][:,None]),1)

        images.append(im)


    print()