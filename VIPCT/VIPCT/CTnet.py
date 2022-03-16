# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from .volumes import Volumes

from typing import List, Optional, Tuple

import torch
from .cameras import PerspectiveCameras

from .mlp_function import MLPWithInputSkips
from .encoder import Backbone
from .positinal_encoding import PositionalEncoding
from .self_attention import SelfAttention
from .mask import MaskGenerator

def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class CTnet(torch.nn.Module):
    """
    """

    def __init__(
        self,
        cfg,
        n_cam
    ):
        """
        Args:

        """

        super().__init__()

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        # self._renderer = torch.nn.ModuleDict()


        # Parse out image dimensions.
        image_size = cfg.data.image_size
        # chunk_size_test = cfg.raysampler.chunk_size_test
        # n_harmonic_functions_xyz = cfg.ct_net.n_harmonic_functions_xyz
        # n_harmonic_functions_dir = cfg.ct_net.n_harmonic_functions_dir
        n_hidden_neurons_xyz = cfg.ct_net.n_hidden_neurons_xyz
        n_hidden_neurons_dir = cfg.ct_net.n_hidden_neurons_dir
        n_layers_xyz = cfg.ct_net.n_layers_xyz
        n_layers_dir = cfg.ct_net.n_layers_dir
        visualization = cfg.visualization.visdom
        append_xyz = cfg.ct_net.append_xyz
        append_dir = cfg.ct_net.append_dir




        # self._spatial_encoder = Backbone.from_conf(cfg)
        self._image_encoder = Backbone.from_cfg(cfg)
        # self.harmonic_embedding = PositionalEncoding(cfg.ct_net.n_harmonic_functions_xyz,
        #                                              cfg.ct_net.n_harmonic_functions_dir)
        # self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self.visualization = visualization
        self.stop_encoder_grad = cfg.ct_net.stop_encoder_grad
        self.dir_at_camera_coordinates = cfg.ct_net.dir_at_camera_coordinates
        self.norm_dir = cfg.ct_net.norm_dir
        self.n_query = cfg.ct_net.n_query
        self.val_n_query = cfg.ct_net.val_n_query
        self.n_cam = n_cam
        self.query_point_method = cfg.ct_net.query_point_method
        self.query_point_val_method = cfg.ct_net.query_point_val_method if hasattr(cfg.ct_net,'query_point_val_method') else 'all'

        self.mask_type = None if cfg.ct_net.mask_type == 'None' else cfg.ct_net.mask_type
        self.val_mask_type = None if cfg.ct_net.val_mask_type == 'None' else cfg.ct_net.val_mask_type

        # self.mask_net = MaskGenerator()
        if n_layers_xyz:
            self.mlp_xyz = MLPWithInputSkips(
                n_layers_xyz,
                self.harmonic_embedding.embedding_dim_xyz,
                n_hidden_neurons_xyz,
                self.harmonic_embedding.embedding_dim_xyz,
                n_hidden_neurons_xyz,
                input_skips=append_xyz,
            )
        else:
            self.mlp_xyz = None

        # self.mlp_dir = MLPWithInputSkips(
        #     n_layers_dir,
        #     self.harmonic_embedding.embedding_dim_dir,
        #     n_hidden_neurons_dir,
        #     self.harmonic_embedding.embedding_dim_dir,
        #     n_hidden_neurons_dir,
        #     input_skips=append_dir,
        # )
        # if self.mlp_xyz:
        #     embedding_dim_xyz = n_hidden_neurons_xyz
        # else:
        #     embedding_dim_xyz = self.harmonic_embedding.embedding_dim_xyz
        #
        # if self.mlp_dir:
        #     embedding_dim_dir = n_hidden_neurons_dir
        # else:
        #     embedding_dim_dir = self.harmonic_embedding.embedding_dim_dir

        # self.attention = SelfAttention(self._image_encoder.latent_size, 8,bias=False)
        # linear1 = torch.nn.Linear(self.n_cam * (self._image_encoder.latent_size
        #                                         + embedding_dim_dir) + embedding_dim_xyz, 512)
        linear1 = torch.nn.Linear(n_cam * self._image_encoder.latent_size, 512)
        linear2 = torch.nn.Linear(512, 64)
        linear3 = torch.nn.Linear(64, 1)
        _xavier_init(linear1)
        _xavier_init(linear2)
        _xavier_init(linear3)

        self.decoder = torch.nn.Sequential(
            linear1,
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(True),
            linear2,
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(True),
            linear3
        )



    def forward(
        self,
        cameras: PerspectiveCameras,
        image: torch.Tensor,
        volume: Volumes,
        masks: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """
        Args:
            camera_hashes: A unique identifier of a pre-cached camera.
            If `None`, the cache is not searched and the sampled rays are
            calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
            ('batch_size', 'num_cameras', ...).
            volume: A batch of corresponding ground truth 3D volumes of shape
            ('batch_size', Nx, Ny, Nz).

        """

        if len(image.shape)==4:
            image = image[:,:,None,...]
        Vbatch = len(volume)

        image = image.view(-1, *image.shape[2:])
        image_features = self._image_encoder(image)
        image_features = [features.view(Vbatch,self.n_cam,*features.shape[1:]) for features in image_features]
        del image
        # mask = self.mask_net(image_features)

        if self.training:
            volume, query_points, _ = volume.get_query_points(self.n_query, self.query_point_method, masks = masks)
        else:
            volume, query_points, query_indices = volume.get_query_points(self.val_n_query, self.query_point_val_method, masks = masks)
        n_query = [points.shape[0] for points in query_points]
        uv = cameras.project_points(query_points,screen=True)
        del cameras

        if self.training:
            latent = self._image_encoder.sample_roi(image_features, uv)#.transpose(1, 2)
            if self.stop_encoder_grad:
                latent = [lat.detach() for lat in latent]

            # latent = latent.reshape(Vbatch * n_query, -1)
            latent = torch.vstack(latent)
            output = self.decoder(latent)#.reshape(Vbatch, n_query)
            output = torch.split(output, n_query)
            out = {"output": output, "volume": volume}
        else:

            n_chunk = int(torch.ceil(torch.tensor(n_query).sum() / self.val_n_query))
            # if n_chunk == 0 or torch.tensor(n_query).sum() < self.val_n_query:
            #     print()
            uv = [torch.chunk(p, n_chunk, dim=1) for p in uv]
            output = [torch.empty(0,device=image_features[0].device)] * len(n_query)
            for chunk in range(n_chunk):
                uv_chunk = [p[chunk] for p in uv]
                n_split = [points.shape[1] for points in uv_chunk]
                latent_chunk = self._image_encoder.sample_roi(image_features, uv_chunk)#.transpose(1, 2)
                latent_chunk = torch.vstack(latent_chunk)
                output_chunk = self.decoder(latent_chunk)
                output_chunk = torch.split(output_chunk, n_split)
                output = [torch.cat((out_i, out_chunk_i)) for out_i, out_chunk_i in zip(output, output_chunk)]
            out = {"output": output, "volume": volume, 'query_indices': query_indices}
        del uv
        del image_features
        # latent = latent  # .reshape(len(cameras),Vbatch * query_points.shape[1], -1)

        # volume = torch.vstack(volume)


        return out


# class CTnet_global(torch.nn.Module):
#     """
#     """
#
#     def __init__(
#         self,
#         cfg,
#         n_cam
#     ):
#         """
#         Args:
#
#         """
#
#         super().__init__()
#
#         # The renderers and implicit functions are stored under the fine/coarse
#         # keys in ModuleDict PyTorch modules.
#         # self._renderer = torch.nn.ModuleDict()
#
#
#         # Parse out image dimensions.
#         image_size = cfg.data.image_size
#         # chunk_size_test = cfg.raysampler.chunk_size_test
#         # n_harmonic_functions_xyz = cfg.ct_net.n_harmonic_functions_xyz
#         # n_harmonic_functions_dir = cfg.ct_net.n_harmonic_functions_dir
#         n_hidden_neurons_xyz = cfg.ct_net.n_hidden_neurons_xyz
#         n_hidden_neurons_dir = cfg.ct_net.n_hidden_neurons_dir
#         n_layers_xyz = cfg.ct_net.n_layers_xyz
#         n_layers_dir = cfg.ct_net.n_layers_dir
#         visualization = cfg.visualization.visdom
#         append_xyz = cfg.ct_net.append_xyz
#         append_dir = cfg.ct_net.append_dir
#
#
#
#
#         # self._spatial_encoder = Backbone.from_conf(cfg)
#         self._image_encoder = Backbone.from_cfg(cfg)
#         self.harmonic_embedding = PositionalEncoding(cfg.ct_net.n_harmonic_functions_xyz,
#                                                      cfg.ct_net.n_harmonic_functions_dir)
#         # self._chunk_size_test = chunk_size_test
#         self._image_size = image_size
#         self.visualization = visualization
#         self.stop_encoder_grad = cfg.ct_net.stop_encoder_grad
#         self.dir_at_camera_coordinates = cfg.ct_net.dir_at_camera_coordinates
#         self.norm_dir = cfg.ct_net.norm_dir
#         self.n_query = cfg.ct_net.n_query
#         self.n_cam = n_cam
#
#
#         # self.mlp_xyz = MLPWithInputSkips(
#         #     n_layers_xyz,
#         #     self.harmonic_embedding.embedding_dim_xyz + self._image_encoder.latent_size,
#         #     n_hidden_neurons_xyz,
#         #     self.harmonic_embedding.embedding_dim_xyz + self._image_encoder.latent_size,
#         #     n_hidden_neurons_xyz,
#         #     input_skips=append_xyz,
#         # )
#
#         # self.mlp_dir = MLPWithInputSkips(
#         #     n_layers_dir,
#         #     self.harmonic_embedding.embedding_dim_dir,
#         #     n_hidden_neurons_dir,
#         #     self.harmonic_embedding.embedding_dim_dir,
#         #     n_hidden_neurons_dir,
#         #     input_skips=append_dir,
#         # )
#         # if self.mlp_xyz:
#         #     embedding_dim_xyz = n_hidden_neurons_xyz
#         # else:
#         #     embedding_dim_xyz = self.harmonic_embedding.embedding_dim_xyz
#         #
#         # if self.mlp_dir:
#         #     embedding_dim_dir = n_hidden_neurons_dir
#         # else:
#         #     embedding_dim_dir = self.harmonic_embedding.embedding_dim_dir
#         linear1 = torch.nn.Linear(n_hidden_neurons_xyz * self.n_query, self.n_query)
#         linear2 = torch.nn.Linear(self.n_query, self.n_query)
#         _xavier_init(linear1)
#         _xavier_init(linear2)
#
#         self.decoder = torch.nn.Sequential(
#             linear1,
#             torch.nn.ReLU(True),
#             linear2,
#             # torch.nn.ReLU(True)
#         )
#
#
#     def forward(
#         self,
#         cameras: PerspectiveCameras,
#         image: torch.Tensor,
#         volume: torch.Tensor,
#     ) -> Tuple[dict, dict]:
#         """
#         Args:
#             camera_hashes: A unique identifier of a pre-cached camera.
#             If `None`, the cache is not searched and the sampled rays are
#             calculated from scratch.
#             camera: A batch of cameras from which the scene is rendered.
#             image: A batch of corresponding ground truth images of shape
#             ('batch_size', ·, ·, 3).
#             volume: A batch of corresponding ground truth 3D volumes of shape
#             ('batch_size', Nx, Ny, Nz).
#
#         """
#
#         if len(image.shape)==3:
#             image = image[None,...]
#         volume = Volumes(
#             densities=volume[None,None],
#             features=torch.ones(1, 3, *volume.shape, device=volume.device)/10,
#             voxel_size=3.0 / volume.shape[0],
#         ).to(volume.device)
#
#         query_points = volume.get_coord_grid()
#         Vbatch, Vx, Vy, Vz, Ncoord = query_points.shape
#         query_points = query_points.reshape(Vbatch, -1, Ncoord)
#         n_query = min(self.n_query, query_points.shape[1])
#         # indices = torch.randperm(query_points.shape[1])[:n_query]
#         indices = torch.topk(volume.densities().reshape(-1), n_query).indices[torch.randperm(n_query)]
#         volume = volume.densities().reshape(1,-1)[:, indices]
#         query_points = query_points[:,indices,:]
#         image_features = self._image_encoder(image)
#         for camera, im in zip(cameras, image):
#             camera.image_size = im.shape[1:3]
#         del image
#
#         viewdirs = []
#         latent = []
#         for camera, features in zip(cameras, image_features):
#
#             # * Encode the view directions
#             if self.norm_dir:
#                 dir = torch.nn.functional.normalize(query_points - camera.get_camera_center(), dim=-1)
#             else:
#                 dir = query_points - camera.get_camera_center()
#             # if self.dir_at_camera_coordinates:
#             #     R_ = Rotate(camera.R, device=features.device)
#             #     dir = R_.transform_points(dir)
#             viewdirs.append(dir)
#             uv = camera.transform_points_screen(query_points)[..., :2]
#             latent.append(self._image_encoder.sample_images(features[None],
#                                                             uv,
#                                                             torch.tensor(
#                                                                 [camera.image_size[0], camera.image_size[1]],
#                                                                 device=uv.device)
#                                                             ))
#             del camera
#             del uv
#         del image_features
#         # project the only the first point in each ray
#         # rays_points_world = torch.stack(rays_points_world)
#         # xyz = torch.stack(xyz).transpose(0, 1)
#         viewdirs = torch.stack(viewdirs).transpose(0, 1)
#         latent = torch.mean(torch.stack(latent).permute(1, 3, 2, 0),-1) ## TODO learn weighting from norm for dir (distance to camera)?
#
#         if self.stop_encoder_grad:
#             latent = latent.detach()
#         # latent = latent.transpose(0, 1)#.reshape(
#         #     -1, self.latent_size
#         # )  # (SB * NS * B, latent)
#         query_points, viewdirs = self.harmonic_embedding(query_points, viewdirs)
#         # if self.mlp_dir:
#         #     viewdirs = self.mlp_dir(viewdirs,viewdirs)
#         # viewdirs = viewdirs.transpose(1, 2).reshape(*query_points.shape[:2], -1)
#
#         latent = torch.cat((latent, query_points),-1)
#         del query_points
#         del viewdirs
#
#         if self.mlp_xyz:
#             latent = self.mlp_xyz(latent,latent)
#
#         # latent = torch.cat((latent, query_points), dim=-1)
#         # latent = torch.cat((latent , viewdirs), dim=-1)
#         latent = latent.reshape(Vbatch, -1)
#         output = self.decoder(latent)
#
#         out = {"output": output, "volume": volume}
#
#
#         return out
