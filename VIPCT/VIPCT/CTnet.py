# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from .volumes import Volumes

from typing import List, Optional, Tuple

import torch
from .cameras import PerspectiveCameras, AirMSPICameras

from .mlp_function import MLPWithInputSkips
from .encoder import Backbone
from .decoder import Decoder
from .feature_encoding import FeatureEncoding
from .positinal_encoding import PositionalEncoding
from .self_attention import SelfAttention
from .mask import MaskGenerator




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
        # image_size = cfg.data.image_size
        # chunk_size_test = cfg.raysampler.chunk_size_test
        # n_harmonic_functions_xyz = cfg.ct_net.n_harmonic_functions_xyz
        # n_harmonic_functions_dir = cfg.ct_net.n_harmonic_functions_dir
        # n_hidden_neurons_xyz = cfg.ct_net.n_hidden_neurons_xyz
        # n_hidden_neurons_dir = cfg.ct_net.n_hidden_neurons_dir
        n_layers_xyz = cfg.ct_net.n_layers_xyz
        # n_layers_dir = cfg.ct_net.n_layers_dir
        # visualization = cfg.visualization.visdom
        append_xyz = cfg.ct_net.append_xyz
        # append_dir = cfg.ct_net.append_dir
        feature_encoding = cfg.ct_net.feature_encoding
        self.use_neighbours = cfg.ct_net.use_neighbours if hasattr(cfg.ct_net,'use_neighbours') else False


        # self._spatial_encoder = Backbone.from_conf(cfg)
        self._image_encoder = Backbone.from_cfg(cfg)
        # self.harmonic_embedding = PositionalEncoding(cfg.ct_net.n_harmonic_functions_xyz,
        #                                              cfg.ct_net.n_harmonic_functions_dir)
        # self._chunk_size_test = chunk_size_test
        # self._image_size = image_size
        # self.visualization = visualization
        self.stop_encoder_grad = cfg.ct_net.stop_encoder_grad
        # self.dir_at_camera_coordinates = cfg.ct_net.dir_at_camera_coordinates
        # self.norm_dir = cfg.ct_net.norm_dir
        self.n_query = cfg.ct_net.n_query
        self.val_n_query = cfg.ct_net.val_n_query
        self.n_cam = n_cam
        self.query_point_method = cfg.ct_net.query_point_method
        self.query_point_val_method = cfg.ct_net.query_point_val_method if hasattr(cfg.ct_net,'query_point_val_method') else 'all'

        self.mask_type = None if cfg.ct_net.mask_type == 'None' else cfg.ct_net.mask_type
        self.val_mask_type = None if cfg.ct_net.val_mask_type == 'None' else cfg.ct_net.val_mask_type

        self.decoder_input_size = self._image_encoder.latent_size
        # self.mask_net = MaskGenerator()

        if n_layers_xyz:
            self.mlp_xyz = MLPWithInputSkips(
                n_layers_xyz,
                3, #self.harmonic_embedding.embedding_dim_xyz,
                3,#self.harmonic_embedding.embedding_dim_xyz,
                64,
                input_skips=append_xyz,
            )
            self.mlp_cam_center = MLPWithInputSkips(
                n_layers_xyz,
                3,  # self.harmonic_embedding.embedding_dim_xyz,
                3,  # self.harmonic_embedding.embedding_dim_xyz,
                64,
                input_skips=append_xyz,
            )
            self.decoder_input_size += 64*2

        else:
            self.mlp_xyz = None
            self.mlp_cam_center = None
        self.decoder_input_size *= n_cam
        if feature_encoding:
            self._feature_encoder = FeatureEncoding(n_harmonic_functions=cfg.feature_encoder.n_harm)
            self.decoder_input_size *= (1 + 2 * cfg.feature_encoder.n_harm)
        else:
            self._feature_encoder = None

        self.decoder = Decoder.from_cfg(cfg, self.decoder_input_size, self.use_neighbours)

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
            if self.use_neighbours:
                volume, query_points, _ = volume.get_query_points_and_neighbours(self.n_query, self.query_point_method, masks=masks)
            else:
                volume, query_points, _ = volume.get_query_points(self.n_query, self.query_point_method, masks = masks)
        else:
            volume, query_points, query_indices = volume.get_query_points(self.val_n_query, self.query_point_val_method, masks = masks)
        n_query = [points.shape[0] for points in query_points]
        uv = cameras.project_points(query_points,screen=True)

        if self.mlp_cam_center:
            cam_centers = cameras.get_camera_center()
            embed_camera_center = self.mlp_cam_center(cam_centers.view(-1,3),cam_centers.view(-1,3)).view(*cam_centers.shape[:-1],-1)

        else:
            embed_camera_center = None
        del cameras
        if self.mlp_xyz:
            query_points = torch.vstack(query_points).view(-1,3)
            query_points = self.mlp_xyz(query_points, query_points)
        else:
            query_points = None
        if self.training:

            latent = self._image_encoder.sample_roi(image_features, uv)#.transpose(1, 2)

            if self.stop_encoder_grad:
                latent = [lat.detach() for lat in latent]

            # latent = latent.reshape(Vbatch * n_query, -1)
            latent = torch.vstack(latent).transpose(0, 1)
            if query_points is not None:
                query_points = query_points.unsqueeze(1).expand(-1,latent.shape[1],-1)
                latent = torch.cat((latent,query_points),-1)
                del query_points
            if embed_camera_center is not None:
                embed_camera_center = embed_camera_center#.unsqueeze(1).expand(-1,int(latent.shape[0]/Vbatch),-1,-1)
                # embed_camera_center = embed_camera_center.reshape(-1,*embed_camera_center.shape[2:])
                latent = torch.split(latent,n_query)
                latent = torch.vstack([torch.cat((lat,embed.expand(lat.shape[0],-1,-1)),-1) for lat, embed in zip(latent, embed_camera_center)])
                del embed_camera_center

            if self._feature_encoder:
                latent = self._feature_encoder(latent)

            output = self.decoder(latent)#.reshape(Vbatch, n_query)

            output = torch.split(output, n_query)
            out = {"output": output, "volume": volume}
        else:

            n_chunk = int(torch.ceil(torch.tensor(n_query).sum() / self.val_n_query))
            uv = [torch.chunk(p, n_chunk, dim=1) for p in uv]
            query_points = torch.chunk(query_points, n_chunk) if query_points is not None else None
            output = [torch.empty(0,device=image_features[0].device)] * len(n_query)
            for chunk in range(n_chunk):
                uv_chunk = [p[chunk] for p in uv]
                n_split = [points.shape[1] for points in uv_chunk]
                latent_chunk = self._image_encoder.sample_roi(image_features, uv_chunk)#.transpose(1, 2)
                latent_chunk = torch.vstack(latent_chunk).transpose(0, 1)
                if query_points is not None:
                    assert Vbatch==1
                    query_points_chunk = query_points[chunk]
                    query_points_chunk = query_points_chunk.unsqueeze(1).expand(-1, latent_chunk.shape[1], -1)
                    latent_chunk = torch.cat((latent_chunk, query_points_chunk), -1)
                    del query_points_chunk
                if embed_camera_center is not None:
                    assert Vbatch == 1
                    embed_camera_center_chunk = embed_camera_center.unsqueeze(1).expand(-1, int(latent_chunk.shape[0] / Vbatch), -1, -1)
                    embed_camera_center_chunk = embed_camera_center_chunk.reshape(-1, *embed_camera_center_chunk.shape[2:])
                    latent_chunk = torch.cat((latent_chunk, embed_camera_center_chunk), -1)

                if self._feature_encoder:
                    latent_chunk = self._feature_encoder(latent_chunk)
                output_chunk = self.decoder(latent_chunk)
                output_chunk = torch.split(output_chunk, n_split)
                output = [torch.cat((out_i, out_chunk_i)) for out_i, out_chunk_i in zip(output, output_chunk)]
            out = {"output": output, "volume": volume, 'query_indices': query_indices}
        # del uv
        # del image_features
        # del embed_camera_center

        # latent = latent  # .reshape(len(cameras),Vbatch * query_points.shape[1], -1)

        # volume = torch.vstack(volume)


        return out


class CTnetMicrophysics(torch.nn.Module):
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
        # image_size = cfg.data.image_size
        # chunk_size_test = cfg.raysampler.chunk_size_test
        # n_harmonic_functions_xyz = cfg.ct_net.n_harmonic_functions_xyz
        # n_harmonic_functions_dir = cfg.ct_net.n_harmonic_functions_dir
        # n_hidden_neurons_xyz = cfg.ct_net.n_hidden_neurons_xyz
        # n_hidden_neurons_dir = cfg.ct_net.n_hidden_neurons_dir
        n_layers_xyz = cfg.ct_net.n_layers_xyz
        # n_layers_dir = cfg.ct_net.n_layers_dir
        # visualization = cfg.visualization.visdom
        append_xyz = cfg.ct_net.append_xyz
        # append_dir = cfg.ct_net.append_dir
        feature_encoding = cfg.ct_net.feature_encoding
        self.use_neighbours = cfg.ct_net.use_neighbours if hasattr(cfg.ct_net,'use_neighbours') else False


        # self._spatial_encoder = Backbone.from_conf(cfg)
        self._image_encoder = Backbone.from_cfg(cfg)
        # self.harmonic_embedding = PositionalEncoding(cfg.ct_net.n_harmonic_functions_xyz,
        #                                              cfg.ct_net.n_harmonic_functions_dir)
        # self._chunk_size_test = chunk_size_test
        # self._image_size = image_size
        # self.visualization = visualization
        self.stop_encoder_grad = cfg.ct_net.stop_encoder_grad
        # self.dir_at_camera_coordinates = cfg.ct_net.dir_at_camera_coordinates
        # self.norm_dir = cfg.ct_net.norm_dir
        self.n_query = cfg.ct_net.n_query
        self.val_n_query = cfg.ct_net.val_n_query
        self.n_cam = n_cam
        self.query_point_method = cfg.ct_net.query_point_method
        self.query_point_val_method = cfg.ct_net.query_point_val_method if hasattr(cfg.ct_net,'query_point_val_method') else 'all'

        self.mask_type = None if cfg.ct_net.mask_type == 'None' else cfg.ct_net.mask_type
        self.val_mask_type = None if cfg.ct_net.val_mask_type == 'None' else cfg.ct_net.val_mask_type

        self.decoder_input_size = self._image_encoder.latent_size
        # self.mask_net = MaskGenerator()

        if n_layers_xyz:
            self.mlp_xyz = MLPWithInputSkips(
                n_layers_xyz,
                3, #self.harmonic_embedding.embedding_dim_xyz,
                3,#self.harmonic_embedding.embedding_dim_xyz,
                64,
                input_skips=append_xyz,
            )
            self.mlp_cam_center = MLPWithInputSkips(
                n_layers_xyz,
                3,  # self.harmonic_embedding.embedding_dim_xyz,
                3,  # self.harmonic_embedding.embedding_dim_xyz,
                64,
                input_skips=append_xyz,
            )
            self.decoder_input_size += 64*2

        else:
            self.mlp_xyz = None
            self.mlp_cam_center = None
        self.decoder_input_size *= n_cam
        if feature_encoding:
            self._feature_encoder = FeatureEncoding(n_harmonic_functions=cfg.feature_encoder.n_harm)
            self.decoder_input_size *= (1 + 2 * cfg.feature_encoder.n_harm)
        else:
            self._feature_encoder = None

        self.decoder = Decoder.from_cfg(cfg, self.decoder_input_size, self.use_neighbours)

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
            if self.use_neighbours:
                NotImplementedError()
                # volume, query_points, _ = volume.get_query_points_and_neighbours(self.n_query, self.query_point_method, masks=masks)
            else:
                volume, query_points, _ = volume.get_query_points_microphysics(self.n_query, self.query_point_method, masks = masks)
        else:
            volume, query_points, query_indices = volume.get_query_points_microphysics(self.val_n_query, self.query_point_val_method, masks = masks)
        n_query = [points.shape[0] for points in query_points]
        uv = cameras.project_points(query_points,screen=True)

        if self.mlp_cam_center:
            cam_centers = cameras.get_camera_center()
            embed_camera_center = self.mlp_cam_center(cam_centers.view(-1,3),cam_centers.view(-1,3)).view(*cam_centers.shape[:-1],-1)

        else:
            embed_camera_center = None
        del cameras
        if self.mlp_xyz:
            query_points = torch.vstack(query_points).view(-1,3)
            query_points = self.mlp_xyz(query_points, query_points)
        else:
            query_points = None
        if self.training:

            latent = self._image_encoder.sample_roi(image_features, uv)#.transpose(1, 2)

            if self.stop_encoder_grad:
                latent = [lat.detach() for lat in latent]

            # latent = latent.reshape(Vbatch * n_query, -1)
            latent = torch.vstack(latent).transpose(0, 1)
            if query_points is not None:
                query_points = query_points.unsqueeze(1).expand(-1,latent.shape[1],-1)
                latent = torch.cat((latent,query_points),-1)
                del query_points
            if embed_camera_center is not None:
                embed_camera_center = embed_camera_center#.unsqueeze(1).expand(-1,int(latent.shape[0]/Vbatch),-1,-1)
                # embed_camera_center = embed_camera_center.reshape(-1,*embed_camera_center.shape[2:])
                latent = torch.split(latent,n_query)
                latent = torch.vstack([torch.cat((lat,embed.expand(lat.shape[0],-1,-1)),-1) for lat, embed in zip(latent, embed_camera_center)])
                del embed_camera_center

            if self._feature_encoder:
                latent = self._feature_encoder(latent)

            output = self.decoder(latent)#.reshape(Vbatch, n_query)

            output = torch.split(output, n_query)
            out = {"output": output, "volume": volume}
        else:

            n_chunk = int(torch.ceil(torch.tensor(n_query).sum() / self.val_n_query))
            uv = [torch.chunk(p, n_chunk, dim=1) for p in uv]
            query_points = torch.chunk(query_points, n_chunk) if query_points is not None else None
            output = [torch.empty(0,device=image_features[0].device)] * len(n_query)
            for chunk in range(n_chunk):
                uv_chunk = [p[chunk] for p in uv]
                n_split = [points.shape[1] for points in uv_chunk]
                latent_chunk = self._image_encoder.sample_roi(image_features, uv_chunk)#.transpose(1, 2)
                latent_chunk = torch.vstack(latent_chunk).transpose(0, 1)
                if query_points is not None:
                    assert Vbatch==1
                    query_points_chunk = query_points[chunk]
                    query_points_chunk = query_points_chunk.unsqueeze(1).expand(-1, latent_chunk.shape[1], -1)
                    latent_chunk = torch.cat((latent_chunk, query_points_chunk), -1)
                    del query_points_chunk
                if embed_camera_center is not None:
                    assert Vbatch == 1
                    embed_camera_center_chunk = embed_camera_center.unsqueeze(1).expand(-1, int(latent_chunk.shape[0] / Vbatch), -1, -1)
                    embed_camera_center_chunk = embed_camera_center_chunk.reshape(-1, *embed_camera_center_chunk.shape[2:])
                    latent_chunk = torch.cat((latent_chunk, embed_camera_center_chunk), -1)

                if self._feature_encoder:
                    latent_chunk = self._feature_encoder(latent_chunk)
                output_chunk = self.decoder(latent_chunk)
                output_chunk = torch.split(output_chunk, n_split)
                output = [torch.cat((out_i, out_chunk_i)) for out_i, out_chunk_i in zip(output, output_chunk)]
            out = {"output": output, "volume": volume, 'query_indices': query_indices}
        # del uv
        # del image_features
        # del embed_camera_center

        # latent = latent  # .reshape(len(cameras),Vbatch * query_points.shape[1], -1)

        # volume = torch.vstack(volume)


        return out


class CTnetAirMSPI(torch.nn.Module):
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
        # image_size = cfg.data.image_size
        # chunk_size_test = cfg.raysampler.chunk_size_test
        # n_harmonic_functions_xyz = cfg.ct_net.n_harmonic_functions_xyz
        # n_harmonic_functions_dir = cfg.ct_net.n_harmonic_functions_dir
        # n_hidden_neurons_xyz = cfg.ct_net.n_hidden_neurons_xyz
        # n_hidden_neurons_dir = cfg.ct_net.n_hidden_neurons_dir
        n_layers_xyz = cfg.ct_net.n_layers_xyz
        # n_layers_dir = cfg.ct_net.n_layers_dir
        # visualization = cfg.visualization.visdom
        append_xyz = cfg.ct_net.append_xyz
        # append_dir = cfg.ct_net.append_dir
        feature_encoding = cfg.ct_net.feature_encoding
        self.use_neighbours = cfg.ct_net.use_neighbours if hasattr(cfg.ct_net,'use_neighbours') else False


        # self._spatial_encoder = Backbone.from_conf(cfg)
        self._image_encoder = Backbone.from_cfg(cfg)
        # self.harmonic_embedding = PositionalEncoding(cfg.ct_net.n_harmonic_functions_xyz,
        #                                              cfg.ct_net.n_harmonic_functions_dir)
        # self._chunk_size_test = chunk_size_test
        # self._image_size = image_size
        # self.visualization = visualization
        self.stop_encoder_grad = cfg.ct_net.stop_encoder_grad
        # self.dir_at_camera_coordinates = cfg.ct_net.dir_at_camera_coordinates
        # self.norm_dir = cfg.ct_net.norm_dir
        self.n_query = cfg.ct_net.n_query
        self.val_n_query = cfg.ct_net.val_n_query
        self.n_cam = n_cam
        self.query_point_method = cfg.ct_net.query_point_method
        self.query_point_val_method = cfg.ct_net.query_point_val_method if hasattr(cfg.ct_net,'query_point_val_method') else 'all'

        self.mask_type = None if cfg.ct_net.mask_type == 'None' else cfg.ct_net.mask_type
        self.val_mask_type = None if cfg.ct_net.val_mask_type == 'None' else cfg.ct_net.val_mask_type

        self.decoder_input_size = self._image_encoder.latent_size
        # self.mask_net = MaskGenerator()

        if n_layers_xyz:
            self.mlp_xyz = MLPWithInputSkips(
                n_layers_xyz,
                3, #self.harmonic_embedding.embedding_dim_xyz,
                3,#self.harmonic_embedding.embedding_dim_xyz,
                64,
                input_skips=append_xyz,
            )
            # self.mlp_cam_center = MLPWithInputSkips(
            #     n_layers_xyz,
            #     3,  # self.harmonic_embedding.embedding_dim_xyz,
            #     3,  # self.harmonic_embedding.embedding_dim_xyz,
            #     64,
            #     input_skips=append_xyz,
            # )
            self.decoder_input_size += 64#*2

        else:
            self.mlp_xyz = None
            # self.mlp_cam_center = None
        self.decoder_input_size *= n_cam
        if feature_encoding:
            self._feature_encoder = FeatureEncoding(n_harmonic_functions=cfg.feature_encoder.n_harm)
            self.decoder_input_size *= (1 + 2 * cfg.feature_encoder.n_harm)
        else:
            self._feature_encoder = None

        self.decoder = Decoder.from_cfg(cfg, self.decoder_input_size, self.use_neighbours)

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



    def forward(
        self,
        cameras: AirMSPICameras,
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
            if self.use_neighbours:
                volume, query_points, query_indices = volume.get_query_points_and_neighbours(self.n_query, self.query_point_method, masks=masks)
            else:
                volume, query_points, query_indices = volume.get_query_points(self.n_query, self.query_point_method, masks = masks)
            uv = cameras.project_points(query_indices, screen=True)

        else:
            volume, query_points, query_indices = volume.get_query_points(self.val_n_query, self.query_point_val_method, masks = masks)
            uv = cameras.project_points(None, screen=True)

        n_query = [points.shape[0] for points in query_points]

        # if self.mlp_cam_center:
        #     cam_centers = cameras.get_camera_center()
        #     embed_camera_center = self.mlp_cam_center(cam_centers.view(-1,3),cam_centers.view(-1,3)).view(*cam_centers.shape[:-1],-1)

        # else:
        #     embed_camera_center = None
        del cameras
        if self.mlp_xyz:
            query_points = torch.vstack(query_points).view(-1,3)
            query_points = self.mlp_xyz(query_points, query_points)
        else:
            query_points = None
        if self.training:

            latent = self._image_encoder.sample_roi(image_features, uv)#.transpose(1, 2)
            del image_features
            if self.stop_encoder_grad:
                latent = [lat.detach() for lat in latent]

            # latent = latent.reshape(Vbatch * n_query, -1)
            latent = torch.vstack(latent).transpose(0, 1)
            if query_points is not None:
                query_points = query_points.unsqueeze(1).expand(-1,latent.shape[1],-1)
                latent = torch.cat((latent,query_points),-1)
                del query_points
            # if embed_camera_center is not None:
            #     embed_camera_center = embed_camera_center#.unsqueeze(1).expand(-1,int(latent.shape[0]/Vbatch),-1,-1)
            #     # embed_camera_center = embed_camera_center.reshape(-1,*embed_camera_center.shape[2:])
            #     latent = torch.split(latent,n_query)
            #     latent = torch.vstack([torch.cat((lat,embed.expand(lat.shape[0],-1,-1)),-1) for lat, embed in zip(latent, embed_camera_center)])
            #     del embed_camera_center

            if self._feature_encoder:
                latent = self._feature_encoder(latent)

            output = self.decoder(latent)#.reshape(Vbatch, n_query)

            output = torch.split(output, n_query)
            out = {"output": output, "volume": volume}
        else:

            n_chunk = int(torch.ceil(torch.tensor(n_query).sum() / self.val_n_query))
            uv = [torch.chunk(p, n_chunk, dim=1) for p in uv]
            query_points = torch.chunk(query_points, n_chunk) if query_points is not None else None
            output = [torch.empty(0,device=image_features[0].device)] * len(n_query)
            for chunk in range(n_chunk):
                uv_chunk = [p[chunk] for p in uv]
                n_split = [points.shape[1] for points in uv_chunk]
                latent_chunk = self._image_encoder.sample_roi(image_features, uv_chunk)#.transpose(1, 2)
                latent_chunk = torch.vstack(latent_chunk).transpose(0, 1)
                if query_points is not None:
                    assert Vbatch==1
                    query_points_chunk = query_points[chunk]
                    query_points_chunk = query_points_chunk.unsqueeze(1).expand(-1, latent_chunk.shape[1], -1)
                    latent_chunk = torch.cat((latent_chunk, query_points_chunk), -1)
                    del query_points_chunk
                # if embed_camera_center is not None:
                #     assert Vbatch == 1
                #     embed_camera_center_chunk = embed_camera_center.unsqueeze(1).expand(-1, int(latent_chunk.shape[0] / Vbatch), -1, -1)
                #     embed_camera_center_chunk = embed_camera_center_chunk.reshape(-1, *embed_camera_center_chunk.shape[2:])
                #     latent_chunk = torch.cat((latent_chunk, embed_camera_center_chunk), -1)

                if self._feature_encoder:
                    latent_chunk = self._feature_encoder(latent_chunk)
                output_chunk = self.decoder(latent_chunk)
                output_chunk = torch.split(output_chunk, n_split)
                output = [torch.cat((out_i, out_chunk_i)) for out_i, out_chunk_i in zip(output, output_chunk)]
            out = {"output": output, "volume": volume, 'query_indices': query_indices}



        return out
