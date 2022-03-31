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


    # with open('/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/32cameras/train/cloud_results_0.pkl', 'rb') as outfile:
    with open('/media/roironen/8AAE21F5AE21DB09/Data/CUBE.pkl', 'rb') as outfile:
        x = pickle.load(outfile)
    image_sizes = np.array([image.shape for image in x['images']])
    cameras = PerspectiveCameras(image_size=image_sizes[None], P=torch.tensor(x['cameras_P'],device='cuda').float(),
                                  device='cuda')
    grid = torch.tensor(x['net_grid'],device='cuda')
    grid = torch.tensor(x['grid'],device='cuda')

    extinction = x['ext'][None,None]
    mask = x['mask'][None]
    # mask = [torch.tensor(m) if mask is not None else m for m in mask]
    mask = [torch.ones(extinction.shape, device='cuda', dtype=bool)]
    layers = 4
    # images = [torch.arange(int(128/(i+1))**2).reshape(1,1,int(128/(i+1)),-1).double().repeat(image_sizes.shape[0],1,1,1) for i in range(layers)]
    images = x['images']
    volume = Volumes(torch.tensor(extinction, device='cuda').float(), grid)

    backbone = Backbone(backbone='resnet50_fpn',
            pretrained=False,
            num_layers=4,
            index_interp="bilinear",
            index_padding="border",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
            sampling_output_size=10,
            sampling_support = 10,
            out_channels = 1,
            n_sampling_nets=1,
            to_flatten = False,
            modify_first_layer=True).to('cuda')
    volume, query_points, _ = volume.get_query_points(1000, 'topk', masks=mask)
    uv = cameras.project_points(query_points, screen=True)
    uv_swap= torch.zeros(uv[0].shape,device='cuda')
    uv_swap[..., 0] = uv[0][..., 1]
    uv_swap[..., 1] = uv[0][..., 0]
    uv_swap = [uv_swap]

    boxes = [backbone.make_boxes(box_center).reshape(*box_center.shape[:2],-1) for box_center in uv]
    samples = backbone.sample_roi_debug([torch.tensor(images,device='cuda').unsqueeze(1).unsqueeze(1)],uv)
    # indices = torch.topk(torch.tensor(x['ext']).reshape(-1), 10).indices
    # print(torch.tensor(x['ext']).reshape(-1)[indices])
    # grid = x['grid']
    # volume = Volumes(torch.tensor(x['ext'])[None, None].double(), grid)
    samples[0] = samples[0].reshape(32,1000,10,10)
    i=0
    N=2
    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
    ind = np.random.permutation(1000)[:N]
    ind[0] = 0
    for im, box, center in zip(images, boxes[0], uv[0]):
        fig, ax = plt.subplots(1)
        colors = cm.YlOrBr(np.linspace(0, 1, N))

        plt.imshow(im / np.max(im))
        plt.scatter(center[ind, 0].cpu().numpy(), center[ind, 1].cpu().numpy(), s=1, c='red',
                    marker='x')
        # x = np.linspace(0, 115, 116)
        # y = np.linspace(0, 115, 116)
        # X, Y = np.meshgrid(x, y)
        # f = interpolate.interp2d(X, Y, im, kind='linear')
        for b, v, c in zip(box[ind], volume[0][ind],colors) :
            b = b.cpu().numpy()
            centerx = b[0]
            centery = b[1]
            h = (b[2] - b[0])
            w = (b[3] - b[1])

            # Ti = f(centery,centerx)
            # print(Ti)
            coord = np.round([centery,centerx]).astype(int)
            Ti = im[coord[0], coord[1]]
            Ti = round(Ti,3)
            # print(im[coord[1], coord[0]])
            # print(im[coord[0],coord[1]])
            # Ti = griddata((X, Y), im, np.array([centerx, centery]).reshape((1,2)), method='cubic')
            rect = patches.Rectangle((centerx, centery), w, h, linewidth=1,
                                     edgecolor=c, facecolor="none")
            plt.text(centerx+1, centery, f'{int(v)},{Ti:.3f}', fontsize=6,c =c)
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.title(i)
        i += 1
        # plt.show()
        pdf.savefig(fig)
    pdf.close()
    samples = samples[0][:,ind]
    pdf = matplotlib.backends.backend_pdf.PdfPages("output1.pdf")

    fig, axs = plt.subplots(32,N)
    for im, sample, axx in zip(images, samples, axs):
        for i, ax in enumerate(axx):
            ax.imshow(sample.cpu().numpy()[i] / np.max(im))
            ax.axis('off')
    plt.show()
    pdf.savefig(fig)
    pdf.close()
    print()