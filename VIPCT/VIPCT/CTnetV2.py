# This file contains the main VIP-CT v2 framework code.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Roi Ronen and Yoav. Y. Schechner.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
from VIPCT.scene.volumes import Volumes
from typing import List, Optional, Tuple
import torch
from VIPCT.scene.cameras import PerspectiveCameras, AirMSPICameras
from VIPCT.VIPCT.mlp_function import MLPWithInputSkips, MLPIdentity
from VIPCT.VIPCT.encoder.encoder import Backbone
from VIPCT.VIPCT.decoder.decoder import Decoder
from .decoder.transformer import Transformer


class CTnetV2(torch.nn.Module):

    def __init__(
        self,
        cfg,
        n_cam
    ):

        super().__init__()

        n_layers_xyz = cfg.ct_net.n_layers_xyz
        append_xyz = cfg.ct_net.append_xyz
        self.use_neighbours = cfg.ct_net.use_neighbours if hasattr(cfg.ct_net,'use_neighbours') else False
        self._image_encoder = Backbone.from_cfg(cfg)
        self.stop_encoder_grad = cfg.ct_net.stop_encoder_grad
        self.n_query = cfg.ct_net.n_query
        self.val_n_query = cfg.ct_net.val_n_query
        self.n_cam = n_cam
        self.query_point_method = cfg.ct_net.query_point_method
        self.decoder_masked = False if cfg.decoder.masked == 'None' else cfg.decoder.masked
        self.num_masked = 0 if cfg.decoder.num_masked == 'None' else cfg.decoder.num_masked
        self.query_point_val_method = cfg.ct_net.query_point_val_method if hasattr(cfg.ct_net,'query_point_val_method') else 'all'
        self.mask_type = None if cfg.ct_net.mask_type == 'None' else cfg.ct_net.mask_type
        self.val_mask_type = None if cfg.ct_net.val_mask_type == 'None' else cfg.ct_net.val_mask_type
        self.decoder_input_size = self._image_encoder.latent_size* n_cam
        self.ce_bins = cfg.cross_entropy.bins
        if n_layers_xyz>0:
            if n_layers_xyz>1:
                self.mlp_xyz = MLPWithInputSkips(
                    n_layers_xyz,
                    3,
                    3,
                    cfg.ct_net.n_hidden_neurons_xyz,
                    input_skips=append_xyz,
                )
                self.mlp_cam_center = MLPWithInputSkips(
                    n_layers_xyz,
                    3,
                    3,
                    cfg.ct_net.n_hidden_neurons_dir,
                    input_skips=append_xyz,
                )
                self.decoder_input_size += cfg.ct_net.n_hidden_neurons_dir* n_cam + cfg.ct_net.n_hidden_neurons_xyz
            else:
                # insert raw coordinates
                self.mlp_xyz = MLPIdentity()
                self.mlp_cam_center = MLPIdentity()
                self.decoder_input_size += 3 * n_cam +3
        else:
            self.mlp_xyz = None
            self.mlp_cam_center = None
        # self.decoder_input_size *= n_cam
        if self.decoder_masked:
            len_mask = (cfg.ct_net.n_hidden_neurons_xyz+1) * self.num_masked
            self.decoder_input_size += len_mask

        self.decoder = Decoder.from_cfg(cfg, self.decoder_input_size, self.ce_bins, self.use_neighbours)



    def forward(
        self,
        cameras: PerspectiveCameras,
        image: torch.Tensor,
        volume: Volumes,
        masks: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """
        Args:
            cameras: Camera objects for geometry encoding and image feature projection & sampling.
            image: A batch of corresponding cloud images of shape
            ('batch_size', 'num_cameras', C, H, W).
            volume: A batch of corresponding ground truth 3D volumes of shape
            ('batch_size', Nx, Ny, Nz).
            masks: A batch of corresponding 3D masks of shape
            ('batch_size', Nx, Ny, Nz).
        """

        if len(image.shape)==4:
            image = image[:,:,None,...]
        Vbatch = len(volume)
        image = image.view(-1, *image.shape[2:])
        image_features = self._image_encoder(image)
        image_features = [features.view(Vbatch,self.n_cam,*features.shape[1:]) for features in image_features]

        del image
        if self.training:
            if self.use_neighbours:
                volume, query_points, _ = volume.get_query_points_and_neighbours(self.n_query, self.query_point_method, masks=masks)
            elif self.query_point_method== 'toa_random':
                volume, query_points, _ = volume.get_query_points_seq(self.n_query, self.query_point_method, masks=masks)
            else:
                volume, query_points, _ = volume.get_query_points(self.n_query, self.query_point_method, masks = masks)
        else:
            if self.query_point_method == 'toa_random':
                volume, query_points, query_indices = volume.get_query_points_seq(self.n_query, self.query_point_method,
                                                                      masks=masks)
            else:
                volume, query_points, query_indices = volume.get_query_points(self.val_n_query, self.query_point_val_method, masks = masks)
        n_query = [points.shape[0] for points in query_points]
        uv = cameras.project_points(query_points, screen=True)

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
            latent = torch.vstack(latent).transpose(0, 1)
            if query_points is not None:
                # query_points = query_points.unsqueeze(1).expand(-1,latent.shape[1],-1)
                latent = latent.reshape(latent.shape[0],-1)
                latent = torch.cat((latent,query_points),-1)
                # if self.decoder_masked:
                #     rate = self.num_masked / self.n_query
                #     masks = torch.rand((latent.shape[0],latent.shape[0]),device=latent.device) > rate#[:,:self.num_masked]
                #     masked_points = torch.cat((volume[0].unsqueeze(1),query_points),-1)
                #     points = torch.zeros((latent.shape[0],self.num_masked),device=latent.devcie)
                #     masked_points = [m[mask] for m, mask in zip(masked_points,masks)]
                #     masked_points = masked_points.unsqueeze(0).expand(latent.shape[0],-1,-1)
                #     masked_points = masked_points[mask[None]]
                #     masked_points = masked_points.reshape(latent.shape[0], -1)
                #     latent = torch.cat((latent, masked_points), -1)
                #     del masked_points
                del query_points
            if embed_camera_center is not None:
                embed_camera_center = embed_camera_center.reshape(embed_camera_center.shape[0],-1)
                embed_camera_center = embed_camera_center.expand(latent.shape[0],-1)
                latent = torch.cat((latent, embed_camera_center), -1)

                # latent = torch.split(latent,n_query)
                # latent = torch.vstack([torch.cat((lat,embed.expand(lat.shape[0],-1,-1)),-1) for lat, embed in zip(latent, embed_camera_center)])
                del embed_camera_center

            output = self.decoder(latent)
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
                latent_chunk = self._image_encoder.sample_roi(image_features, uv_chunk)
                latent_chunk = torch.vstack(latent_chunk).transpose(0, 1)
                latent_chunk = latent_chunk.reshape(latent_chunk.shape[0],-1)
                if query_points is not None:
                    assert Vbatch == 1
                    query_points_chunk = query_points[chunk]
                    # query_points_chunk = query_points_chunk.unsqueeze(1).expand(-1, latent_chunk.shape[1], -1)
                    latent_chunk = torch.cat((latent_chunk, query_points_chunk), -1)
                    del query_points_chunk
                if embed_camera_center is not None:
                    assert Vbatch == 1
                    embed_camera_center_chunk = embed_camera_center.unsqueeze(1).expand(-1, int(latent_chunk.shape[0] / Vbatch), -1, -1)
                    embed_camera_center_chunk = embed_camera_center_chunk.reshape(int(latent_chunk.shape[0] / Vbatch),-1)
                    latent_chunk = torch.cat((latent_chunk, embed_camera_center_chunk), -1)

                output_chunk = self.decoder(latent_chunk)
                output_chunk = torch.split(output_chunk, n_split)
                output = [torch.cat((out_i, out_chunk_i)) for out_i, out_chunk_i in zip(output, output_chunk)]

            out = {"output": output, "volume": volume, 'query_indices': query_indices}
        return out
