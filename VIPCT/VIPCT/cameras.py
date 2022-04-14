# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from .util.types import Device
from .util.renderer_utils import TensorProperties, convert_to_tensors_and_broadcast


# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)
norm = lambda x: x / torch.linalg.norm(x, axis=0)


class PerspectiveCameras(TensorProperties):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.

    Parameters for this camera are specified in NDC if `in_ndc` is set to True.
    If parameters are specified in screen space, `in_ndc` must be set to False.
    """

    def __init__(
            self,
            image_size: Union[List, Tuple, torch.Tensor],
            focal_length=1.0,
            camera_center: Optional[torch.Tensor] = None,
            R: Optional[torch.Tensor] = _R,
            T: Optional[torch.Tensor] = _T,
            K: Optional[torch.Tensor] = None,
            P: Optional[torch.Tensor] = None,
            device: Device = "cpu"
    ):
        """

        Args:
            image_size: (height, width) of image size.
                A tensor of shape (B, N, 2) or a list/tuple.
            focal_length: Focal length of the camera in world units.
                A tensor of shape (B, N, 1) or (B, N, 2) for
                square and non-square pixels respectively.
            R: Rotation matrix of shape (B, N, 3, 3)
            T: Translation matrix of shape (B, N, 3)
            K: (optional) camera intrinsic matrix of shape (B, N, 3, 3)
                If provided, don't need focal_length
            P: (optional) camera projection matrix of shape (B, N, 3, 4)
                If provided, don't need Rotation and Translation matrices

            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.

        if K is None:
            K = torch.eye(3, device=device)[None]
            K[:, 0, 0] = focal_length
            K[:, 1, 1] = focal_length
        else:
            K = K.to(device=device)
        if P is None:
            G = torch.hstack([R.T, -R.T @ T.T]).to(device=device)
            P = K @ G
        else:
            if not isinstance(P, torch.Tensor):
                P = torch.tensor(P)
            P = P.to(device=device)
            if len(P.shape) == 2:
                P= P[None, None]
            elif len(P.shape) == 3:
                P= P[None]
            R = P[...,:3]
            T = torch.squeeze(P[..., 3:],dim=-1)
        super().__init__(
            device=device,
            focal_length=focal_length,
            R=R,
            T=T,
            K=K,
            P=P,
            image_size=image_size
        )
        self.camera_center = camera_center
        self.index = torch.arange(self._N)
        if (self.image_size < 1).any():  # pyre-ignore
            raise ValueError("Image_size provided has invalid values")


    def get_camera_center(self) -> torch.Tensor:
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting T here will update the values set in init as this
        value may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        # the camera center is the translation component (the last column) of the transform matrix P (3x4 RT matrix)
        return self.camera_center #self.P[:, :, 3]


    def _transform_points_old(self, points, eps: Optional[float] = None) -> List[torch.Tensor]:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: List of Tensor of shape (Pi, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, num_cams, P, 2) or (num_cams,P, 2) depending
            on the dimensions of the transform
        """
        points_batch = []
        for p in points:
            ones = torch.ones(p.shape[0], 1, dtype=p.dtype, device=p.device)
            points_batch.append(torch.cat([p, ones], dim=1))

        # points_batch = [p.clone() ]
        # if points_batch.dim() == 2:
        #     points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        # if points_batch.dim() != 3:
        #     msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
        #     raise ValueError(msg % repr(points.shape))

        # N, P, _3 = points_batch.shape

        composed_matrices = self.P#[self.index,...]
        if composed_matrices.dim() == 3:
            composed_matrices = composed_matrices[None]
        composed_matrices = composed_matrices.transpose(-2, -1)
        points_out = [_broadcast_bmm(points, composed_matrix) for points, composed_matrix in zip(points_batch, composed_matrices)]
        denom = [out[..., 2:] for out in points_out]  # denominator
        if eps is not None:
            for i, d in enumerate(denom):
                d_sign = d.sign() + (d == 0.0).type_as(d)
                denom[i] = d_sign * torch.clamp(d.abs(), eps)
        points_out = [out[..., :2] / d for out, d in zip(points_out, denom)]

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        # if points_out.shape[0] == 1 and points.dim() == 2:
        #     points_out = points_out.reshape(points.shape[0],-1)

        return points_out


    def _transform_points(self, points, eps: Optional[float] = None) -> List[torch.Tensor]:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: List of Tensor of shape (Pi, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, num_cams, P, 2) or (num_cams,P, 2) depending
            on the dimensions of the transform
        """
        points_batch = []
        for p in points:
            ones = torch.ones(p.shape[0], 1, dtype=p.dtype, device=p.device)
            points_batch.append(torch.cat([p, ones], dim=1))

        # points_batch = [p.clone() ]
        # if points_batch.dim() == 2:
        #     points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        # if points_batch.dim() != 3:
        #     msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
        #     raise ValueError(msg % repr(points.shape))

        # N, P, _3 = points_batch.shape

        composed_matrices = self.P#[self.index,...]
        if composed_matrices.dim() == 3:
            composed_matrices = composed_matrices[None]
        composed_matrices = composed_matrices.transpose(-2, -1)
        points_out = [_broadcast_bmm(points, composed_matrix) for points, composed_matrix in zip(points_batch, composed_matrices)]
        denom = [out[..., 2:] for out in points_out]  # denominator
        if eps is not None:
            for i, d in enumerate(denom):
                d_sign = d.sign() + (d == 0.0).type_as(d)
                denom[i] = d_sign * torch.clamp(d.abs(), eps)
        points_out = [out[..., [1,0]] / d for out, d in zip(points_out, denom)]

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        # if points_out.shape[0] == 1 and points.dim() == 2:
        #     points_out = points_out.reshape(points.shape[0],-1)

        return points_out



    def project_points(
        self, points, eps: Optional[float] = None, screen: bool = False
    ) -> torch.Tensor:
        """
        Transforms points from world space to screen space.
        Input points follow the SHDOM coordinate system conventions: +X[km] (North), +Y [km] (East), +Z[km] (Up).
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: list of torch tensors, each of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            screen: if True returns the projected points in pixels
                    else in ndc space: X, Y in [-1 1]. For non square
                    images, we scale the points such that largest side
                    has range [-1, 1] and the smallest side has range
                    [-u, u], with u < 1.

        Returns
            new_points: transformed points with the same shape as the input
            except the last axis size is 2.
        """
        points_out = self._transform_points(points, eps=eps)
        if screen:
            points_out = self._to_screen_transform(points_out)
        self.index = torch.arange(self._N)
        return points_out

    def _to_screen_transform(self, points):
        image_size = torch.unsqueeze(self.image_size, 2)  # self.image_size[self.index].view(points.shape[1],1, 2)  # of shape (1 or B)x2
        for i, p in enumerate(points):
            if p.shape[-1]==3:
                p = p[...,:2]
            points[i] = p * (image_size[i] / 2) + (image_size[i] / 2)
        return points

    def clone(self):
        """
        Returns a copy of `self`.
        """
        cam_type = type(self)
        other = cam_type(device=self.device)
        return super().clone(other)





    # def unproject_points(
    #     self, xy_depth: torch.Tensor, world_coordinates: bool = True, **kwargs
    # ) -> torch.Tensor:
    #     if world_coordinates:
    #         to_camera_transform = self.get_full_projection_transform(**kwargs)
    #     else:
    #         to_camera_transform = self.get_projection_transform(**kwargs)
    #
    #     unprojection_transform = to_camera_transform.inverse()
    #     xy_inv_depth = torch.cat(
    #         (xy_depth[..., :2], 1.0 / xy_depth[..., 2:3]), dim=-1  # type: ignore
    #     )
    #     return unprojection_transform.transform_points(xy_inv_depth)




class AirMSPICameras(TensorProperties):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.

    Parameters for this camera are specified in NDC if `in_ndc` is set to True.
    If parameters are specified in screen space, `in_ndc` must be set to False.
    """

    def __init__(
            self,
            mapping: torch.tensor,
            centers: torch.tensor=None,
            device: Device = "cpu"
    ):

        super().__init__(
            device=device,
            mapping=mapping,
            centers=centers,
        )

        # if (self.image_size < 1).any():  # pyre-ignore
        #     raise ValueError("Image_size provided has invalid values")


    # def get_camera_center(self) -> torch.Tensor:
    #     """
    #     Return the 3D location of the camera optical center
    #     in the world coordinates.
    #
    #     Args:
    #         **kwargs: parameters for the camera extrinsics can be passed in
    #             as keyword arguments to override the default values
    #             set in __init__.
    #
    #     Setting T here will update the values set in init as this
    #     value may be needed later on in the rendering pipeline e.g. for
    #     lighting calculations.
    #
    #     Returns:
    #         C: a batch of 3D locations of shape (N, 3) denoting
    #         the locations of the center of each camera in the batch.
    #     """
    #     # the camera center is the translation component (the last column) of the transform matrix P (3x4 RT matrix)
    #     return self.camera_center #self.P[:, :, 3]



    def project_points(
        self, points, eps: Optional[float] = None, screen: bool = False
    ) -> torch.Tensor:
        """
        Transforms points from world space to screen space.
        Input points follow the SHDOM coordinate system conventions: +X[km] (North), +Y [km] (East), +Z[km] (Up).
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: list of torch tensors, each of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            screen: if True returns the projected points in pixels
                    else in ndc space: X, Y in [-1 1]. For non square
                    images, we scale the points such that largest side
                    has range [-1, 1] and the smallest side has range
                    [-u, u], with u < 1.

        Returns
            new_points: transformed points with the same shape as the input
            except the last axis size is 2.
        """
        if points is not None:
            points_out = [map[:,p,:] for map, p in zip([self.mapping], points)]
        else:
            points_out = [self.mapping]

        return points_out

    def project_pointsv2(
        self, points, eps: Optional[float] = None, screen: bool = False
    ) -> (torch.Tensor,torch.Tensor):
        """
        Transforms points from world space to screen space.
        Input points follow the SHDOM coordinate system conventions: +X[km] (North), +Y [km] (East), +Z[km] (Up).
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: list of torch tensors, each of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            screen: if True returns the projected points in pixels
                    else in ndc space: X, Y in [-1 1]. For non square
                    images, we scale the points such that largest side
                    has range [-1, 1] and the smallest side has range
                    [-u, u], with u < 1.

        Returns
            new_points: transformed points with the same shape as the input
            except the last axis size is 2.
        """
        if points is not None:
            points_out = []
            pixel_centers = []
            for map, center, p in zip([self.mapping],[self.centers], points):

                pixel_centers.append(center[:,p.to(center.device),:])
                points_out.append(map[:,p.to(map.device),:])
        else:
            points_out = [self.mapping]
            pixel_centers = [self.centers]

        return points_out, pixel_centers

    def clone(self):
        """
        Returns a copy of `self`.
        """
        cam_type = type(self)
        other = cam_type(device=self.device)
        return super().clone(other)








################################################
# Helper functions for world to view transforms
################################################


# def get_world_to_view_transform(
#     R: torch.Tensor = _R, T: torch.Tensor = _T
# ) -> Transform3d:
#     """
#     This function returns a Transform3d representing the transformation
#     matrix to go from world space to view space by applying a rotation and
#     a translation.
#
#     PyTorch3D uses the same convention as Hartley & Zisserman.
#     I.e., for camera extrinsic parameters R (rotation) and T (translation),
#     we map a 3D point `X_world` in world coordinates to
#     a point `X_cam` in camera coordinates with:
#     `X_cam = X_world R + T`
#
#     Args:
#         R: (N, 3, 3) matrix representing the rotation.
#         T: (N, 3) matrix representing the translation.
#
#     Returns:
#         a Transform3d object which represents the composed RT transformation.
#
#     """
#     # TODO: also support the case where RT is specified as one matrix
#     # of shape (N, 4, 4).
#
#     if T.shape[0] != R.shape[0]:
#         msg = "Expected R, T to have the same batch dimension; got %r, %r"
#         raise ValueError(msg % (R.shape[0], T.shape[0]))
#     if T.dim() != 2 or T.shape[1:] != (3,):
#         msg = "Expected T to have shape (N, 3); got %r"
#         raise ValueError(msg % repr(T.shape))
#     if R.dim() != 3 or R.shape[1:] != (3, 3):
#         msg = "Expected R to have shape (N, 3, 3); got %r"
#         raise ValueError(msg % repr(R.shape))
#
#     # Create a Transform3d object
#     T_ = Translate(T, device=T.device)
#     R_ = Rotate(R, device=R.device)
#     return R_.compose(T_)
#
#
# def camera_position_from_spherical_angles(
#     distance: float,
#     elevation: float,
#     azimuth: float,
#     degrees: bool = True,
#     device: Device = "cpu",
# ) -> torch.Tensor:
#     """
#     Calculate the location of the camera based on the distance away from
#     the target point, the elevation and azimuth angles.
#
#     Args:
#         distance: distance of the camera from the object.
#         elevation, azimuth: angles.
#             The inputs distance, elevation and azimuth can be one of the following
#                 - Python scalar
#                 - Torch scalar
#                 - Torch tensor of shape (N) or (1)
#         degrees: bool, whether the angles are specified in degrees or radians.
#         device: str or torch.device, device for new tensors to be placed on.
#
#     The vectors are broadcast against each other so they all have shape (N, 1).
#
#     Returns:
#         camera_position: (N, 3) xyz location of the camera.
#     """
#     broadcasted_args = convert_to_tensors_and_broadcast(
#         distance, elevation, azimuth, device=device
#     )
#     dist, elev, azim = broadcasted_args
#     if degrees:
#         elev = math.pi / 180.0 * elev
#         azim = math.pi / 180.0 * azim
#     x = dist * torch.cos(elev) * torch.sin(azim)
#     y = dist * torch.sin(elev)
#     z = dist * torch.cos(elev) * torch.cos(azim)
#     camera_position = torch.stack([x, y, z], dim=1)
#     if camera_position.dim() == 0:
#         camera_position = camera_position.view(1, -1)  # add batch dim.
#     return camera_position.view(-1, 3)
#
#
# def look_at_rotation(
#     camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: Device = "cpu"
# ) -> torch.Tensor:
#     """
#     This function takes a vector 'camera_position' which specifies the location
#     of the camera in world coordinates and two vectors `at` and `up` which
#     indicate the position of the object and the up directions of the world
#     coordinate system respectively. The object is assumed to be centered at
#     the origin.
#
#     The output is a rotation matrix representing the transformation
#     from world coordinates -> view coordinates.
#
#     Args:
#         camera_position: position of the camera in world coordinates
#         at: position of the object in world coordinates
#         up: vector specifying the up direction in the world coordinate frame.
#
#     The inputs camera_position, at and up can each be a
#         - 3 element tuple/list
#         - torch tensor of shape (1, 3)
#         - torch tensor of shape (N, 3)
#
#     The vectors are broadcast against each other so they all have shape (N, 3).
#
#     Returns:
#         R: (N, 3, 3) batched rotation matrices
#     """
#     # Format input and broadcast
#     broadcasted_args = convert_to_tensors_and_broadcast(
#         camera_position, at, up, device=device
#     )
#     camera_position, at, up = broadcasted_args
#     for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
#         if t.shape[-1] != 3:
#             msg = "Expected arg %s to have shape (N, 3); got %r"
#             raise ValueError(msg % (n, t.shape))
#     z_axis = F.normalize(at - camera_position, eps=1e-5)
#     x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
#     y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
#     is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
#         dim=1, keepdim=True
#     )
#     if is_close.any():
#         replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
#         x_axis = torch.where(is_close, replacement, x_axis)
#     R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
#     return R.transpose(1, 2)
#
#
# def look_at_view_transform(
#     dist=1.0,
#     elev=0.0,
#     azim=0.0,
#     degrees: bool = True,
#     eye: Optional[Sequence] = None,
#     at=((0, 0, 0),),  # (1, 3)
#     up=((0, 1, 0),),  # (1, 3)
#     device: Device = "cpu",
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     This function returns a rotation and translation matrix
#     to apply the 'Look At' transformation from world -> view coordinates [0].
#
#     Args:
#         dist: distance of the camera from the object
#         elev: angle in degrees or radians. This is the angle between the
#             vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
#         azim: angle in degrees or radians. The vector from the object to
#             the camera is projected onto a horizontal plane y = 0.
#             azim is the angle between the projected vector and a
#             reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
#         dist, elev and azim can be of shape (1), (N).
#         degrees: boolean flag to indicate if the elevation and azimuth
#             angles are specified in degrees or radians.
#         eye: the position of the camera(s) in world coordinates. If eye is not
#             None, it will override the camera position derived from dist, elev, azim.
#         up: the direction of the x axis in the world coordinate system.
#         at: the position of the object(s) in world coordinates.
#         eye, up and at can be of shape (1, 3) or (N, 3).
#
#     Returns:
#         2-element tuple containing
#
#         - **R**: the rotation to apply to the points to align with the camera.
#         - **T**: the translation to apply to the points to align with the camera.
#
#     References:
#     [0] https://www.scratchapixel.com
#     """
#
#     if eye is not None:
#         broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
#         eye, at, up = broadcasted_args
#         C = eye
#     else:
#         broadcasted_args = convert_to_tensors_and_broadcast(
#             dist, elev, azim, at, up, device=device
#         )
#         dist, elev, azim, at, up = broadcasted_args
#         C = (
#             camera_position_from_spherical_angles(
#                 dist, elev, azim, degrees=degrees, device=device
#             )
#             + at
#         )
#
#     R = look_at_rotation(C, at, up, device=device)
#     T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
#     return R, T
#
#
# def get_ndc_to_screen_transform(
#     cameras,
#     with_xyflip: bool = False,
#     image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
# ) -> Transform3d:
#     """
#     PyTorch3D NDC to screen conversion.
#     Conversion from PyTorch3D's NDC space (+X left, +Y up) to screen/image space
#     (+X right, +Y down, origin top left).
#
#     Args:
#         cameras
#         with_xyflip: flips x- and y-axis if set to True.
#     Optional kwargs:
#         image_size: ((height, width),) specifying the height, width
#         of the image. If not provided, it reads it from cameras.
#
#     We represent the NDC to screen conversion as a Transform3d
#     with projection matrix
#
#     K = [
#             [s,   0,    0,  cx],
#             [0,   s,    0,  cy],
#             [0,   0,    1,   0],
#             [0,   0,    0,   1],
#     ]
#
#     """
#     # We require the image size, which is necessary for the transform
#     if image_size is None:
#         msg = "For NDC to screen conversion, image_size=(height, width) needs to be specified."
#         raise ValueError(msg)
#
#     K = torch.zeros((cameras._N, 4, 4), device=cameras.device, dtype=torch.float32)
#     if not torch.is_tensor(image_size):
#         image_size = torch.tensor(image_size, device=cameras.device)
#     image_size = image_size.view(-1, 2)  # of shape (1 or B)x2
#     height, width = image_size.unbind(1)
#
#     # For non square images, we scale the points such that smallest side
#     # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
#     # This convention is consistent with the PyTorch3D renderer
#     scale = (image_size.min(dim=1).values - 1.0) / 2.0
#
#     K[:, 0, 0] = scale
#     K[:, 1, 1] = scale
#     K[:, 0, 3] = -1.0 * (width - 1.0) / 2.0
#     K[:, 1, 3] = -1.0 * (height - 1.0) / 2.0
#     K[:, 2, 2] = 1.0
#     K[:, 3, 3] = 1.0
#
#     # Transpose the projection matrix as PyTorch3D transforms use row vectors.
#     transform = Transform3d(
#         matrix=K.transpose(1, 2).contiguous(), device=cameras.device
#     )
#
#     if with_xyflip:
#         # flip x, y axis
#         xyflip = torch.eye(4, device=cameras.device, dtype=torch.float32)
#         xyflip[0, 0] = -1.0
#         xyflip[1, 1] = -1.0
#         xyflip = xyflip.view(1, 4, 4).expand(cameras._N, -1, -1)
#         xyflip_transform = Transform3d(
#             matrix=xyflip.transpose(1, 2).contiguous(), device=cameras.device
#         )
#         transform = transform.compose(xyflip_transform)
#     return transform
#
#
# def get_screen_to_ndc_transform(
#     cameras,
#     with_xyflip: bool = False,
#     image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
# ) -> Transform3d:
#     """
#     Screen to PyTorch3D NDC conversion.
#     Conversion from screen/image space (+X right, +Y down, origin top left)
#     to PyTorch3D's NDC space (+X left, +Y up).
#
#     Args:
#         cameras
#         with_xyflip: flips x- and y-axis if set to True.
#     Optional kwargs:
#         image_size: ((height, width),) specifying the height, width
#         of the image. If not provided, it reads it from cameras.
#
#     We represent the screen to NDC conversion as a Transform3d
#     with projection matrix
#
#     K = [
#             [1/s,    0,    0,  cx/s],
#             [  0,  1/s,    0,  cy/s],
#             [  0,    0,    1,     0],
#             [  0,    0,    0,     1],
#     ]
#
#     """
#     transform = get_ndc_to_screen_transform(
#         cameras,
#         with_xyflip=with_xyflip,
#         image_size=image_size,
#     ).inverse()
#     return transform

def _broadcast_bmm(a, b):
    """
    Batch multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K) or (M, P, K)
        b: torch tensor of shape (N, K, K)

    Returns:
        a and b broadcast multiplied. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.
    """
    if a.dim() == 2:
        a = a[None]
    if b.dim() == 2:
        b = b[None]
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = a.expand(len(b), -1, -1)
        if len(b) == 1:
            b = b.expand(len(a), -1, -1)
    return a.bmm(b)

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    from volumes import Volumes


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

    if False:
        with open('/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/train/cloud_results_0.pkl', 'rb') as outfile:
            x = pickle.load(outfile)
        image_sizes = np.array([image.shape for image in x['images']])
        cameras = PerspectiveCameras(image_sizes, P=x['cameras_P'])
        camera_position_list = x['cameras_pos']
        layers = 4
        images = [torch.arange(int(128/(i+1))**2).reshape(1,1,int(128/(i+1)),-1).double().repeat(image_sizes.shape[0],1,1,1) for i in range(layers)]
        indices = torch.topk(torch.tensor(x['ext']).reshape(-1), 10).indices
        print(torch.tensor(x['ext']).reshape(-1)[indices])
        grid = x['grid']
        volume = Volumes(torch.tensor(x['ext'])[None, None].double(), grid)
        projected_to_world_points = volume.get_coord_grid()[0][indices].double()
        projected_to_screen = cameras.project_points(projected_to_world_points, screen=True)
        projected_to_ndc = cameras.project_points(projected_to_world_points, screen=False)
        for im, screen_points in zip(x['images'], projected_to_screen):
            plt.imshow(im / np.max(im))
            plt.scatter(screen_points[:, 1].cpu().numpy(), screen_points[:, 0].cpu().numpy(), s=1, c='red',
                        marker='x')
            plt.show()
        A = sample_features(images, projected_to_ndc.double())
        print(sample_features(images, projected_to_ndc.double()))
        print()
    else:
        from .cameras import AirMSPICameras
        DEFAULT_DATA_ROOT = '/home/roironen/Data'
        data_root = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train'
        image_root = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/AIRMSPI_IMAGES_LWC_LOW_SC/'
        mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        images_mapping_list = []
        for _, map in mapping.items():
            voxels_list = []
            v = map.values()
            voxels = np.array(list(v), dtype=object)
            ctr = 0
            for i, voxel in enumerate(voxels):
                if len(voxel) > 0:
                    pixels = np.unravel_index(voxel, np.array([350,350]))
                    mean_px = np.mean(pixels, 1)
                    voxels_list.append(mean_px)
                else:
                    ctr += 1
                    voxels_list.append([-100000, -100000])
            images_mapping_list.append(voxels_list)
        with open(image_root, 'rb') as f:
            images = pickle.load(f)['images']
        image_sizes = np.array([[350,350]]*9)
        device = 'cuda'
        cameras = AirMSPICameras(image_size=torch.tensor(image_sizes),mapping=torch.tensor(mapping, device=device).float(),
                                         device=device)
        with open(data_root, 'rb') as f:
            x = pickle.load(f)
        layers = 4
        indices = torch.topk(torch.tensor(x['ext']).reshape(-1), 10).indices
        print(torch.tensor(x['ext']).reshape(-1)[indices])
        grid = x['grid']
        volume = Volumes(torch.tensor(x['ext'])[None, None].double(), grid)
        projected_to_screen = cameras.project_points(indices, screen=True)
        for im, screen_points in zip(x['images'], projected_to_screen):
            plt.imshow(im / np.max(im))
            plt.scatter(screen_points[:, 1].cpu().numpy(), screen_points[:, 0].cpu().numpy(), s=1, c='red',
                        marker='x')
            plt.show()
        A = sample_features(images, projected_to_screen.double())
        print(sample_features(images, projected_to_screen.double()))
        print()


