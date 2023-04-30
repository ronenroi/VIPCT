import os, time
import numpy as np
import argparse
import at3d
import shdom
import dill as pickle
import torch
from VIPCT.renderer.shdom_util import Monotonous
import matplotlib.pyplot as plt
import xarray as xr
from collections import OrderedDict

def perspective_projection(wavelength, fov, x_resolution, y_resolution,
                           position_vector, rotation_matrix,
                           stokes='I'):
    """
    Generates a sensor dataset that observes a target location with
    a perspective (pinhole camera) projection.

    Parameters
    ----------
    wavelength: float,
        Wavelength in [micron]
    fov: float
        Field of view [deg]
    x_resolution: int
        Number of pixels in camera x axis
    y_resolution: int
        Number of pixels in camera y axis
    position_vector: list of 3 float elements
        [x , y , z] which are:
        Location in global x coordinates [km] (North)
        Location in global y coordinates [km] (East)
        Location in global z coordinates [km] (Up)
    lookat_vector: list of 3 float elements
        [x , y , z] which are:
        Point in global x coordinates [km] (North) where the camera is pointing at
        Point in global y coordinates [km] (East) where the camera is pointing at
        Point in global z coordinates [km] (Up) where the camera is pointing at
    up_vector: list of 3 float elements
        The up vector determines the roll of the camera.
    stokes: list or string
       list or string of stokes components to observe ['I', 'Q', 'U', 'V'].
    sub_pixel_ray_args : dict
        dictionary defining the method for generating sub-pixel rays. The callable
        which generates the position_perturbations and weights (e.g. at3d.sensor.gaussian)
        should be set as the 'method', while arguments to that callable, should be set as
        other entries in the dict. Each argument have two values, one for each of the
        x and y axes of the image plane, respectively.
        E.g. sub_pixel_ray_args={'method':at3d.sensor.gaussian, 'degree': (2, 3)}

    Returns
    -------
    sensor : xr.Dataset
        A dataset containing all of the information required to define a sensor
        for which synthetic measurements can be simulated;
        positions and angles of all pixels, sub-pixel rays and their associated weights,
        and the sensor's observables.

    """
    norm = lambda x: x / np.linalg.norm(x, axis=0)

    #assert samples>=1, "Sample per pixel is an integer >= 1"
    #assert int(samples) == samples, "Sample per pixel is an integer >= 1"

    assert int(x_resolution) == x_resolution, "x_resolution is an integer >= 1"
    assert int(y_resolution) == y_resolution, "y_resolution is an integer >= 1"

    # The bounding_box is not nessesary in the prespactive projection, but we still may consider
    # to use if we project the rays on the bounding_box when the differences in mu , phi angles are below certaine precision.
    #     if(bounding_box is not None):

    #         xmin, ymin, zmin = bounding_box.x.data.min(),bounding_box.y.data.min(),bounding_box.z.data.min()
    #         xmax, ymax, zmax = bounding_box.x.data.max(),bounding_box.y.data.max(),bounding_box.z.data.max()

    nx = x_resolution
    ny = y_resolution
    position = np.array(position_vector, dtype=np.float32)

    M = max(nx, ny)
    npix = nx*ny
    R = np.array([nx, ny])/M # R will be used to scale the sensor meshgrid.
    dy = 2*R[1]/ny # pixel length in y direction in the normalized image plane.
    dx = 2*R[0]/nx # pixel length in x direction in the normalized image plane.
    x_s, y_s, z_s = np.meshgrid(np.linspace(-R[0]+dx/2, R[0]-dx/2, nx),
                                np.linspace(-R[1]+dy/2, R[1]-dy/2, ny), 1.0)

    # Here x_c, y_c, z_c coordinates on the image plane before transformation to the requaired observation angle
    focal = 1.0 / np.tan(np.deg2rad(fov) / 2.0) # focal (normalized) length when the sensor size is 2 e.g. r in [-1,1).
    fov_x = np.rad2deg(2*np.arctan(R[0]/focal))
    fov_y = np.rad2deg(2*np.arctan(R[1]/focal))

    k = np.array([[focal, 0, 0],
                  [0, focal, 0],
                  [0, 0, 1]], dtype=np.float32)
    inv_k = np.linalg.inv(k)

    homogeneous_coordinates = np.stack([x_s.ravel(), y_s.ravel(), z_s.ravel()])

    x_c, y_c, z_c = norm(np.matmul(
        rotation_matrix, np.matmul(inv_k, homogeneous_coordinates)))
    # Here x_c, y_c, z_c coordinates on the image plane after transformation to the requaired observation

    # x,y,z mu, phi in the global coordinates:
    mu = -z_c.astype(np.float64)
    phi = (np.arctan2(y_c, x_c) + np.pi).astype(np.float64)
    x = np.full(npix, position[0], dtype=np.float32)
    y = np.full(npix, position[1], dtype=np.float32)
    z = np.full(npix, position[2], dtype=np.float32)

    image_shape = [nx,ny]
    sensor = at3d.sensor.make_sensor_dataset(x.ravel(), y.ravel(), z.ravel(),
                                 mu.ravel(), phi.ravel(), stokes, wavelength)
    # compare to orthographic projection, prespective projection may not have bounding box.
    #     if(bounding_box is not None):
    #         sensor['bounding_box'] = xr.DataArray(np.array([xmin,ymin,zmin,xmax,ymax,zmax]),
    #                                               coords={'bbox': ['xmin','ymin','zmin','xmax','ymax','zmax']},dims='bbox')

    sensor['image_shape'] = xr.DataArray(image_shape,
                                         coords={'image_dims': ['nx', 'ny']},
                                         dims='image_dims')
    sensor.attrs = {
        'projection': 'Perspective',
        'fov_deg': fov,
        'fov_x_deg': fov_x,
        'fov_y_deg': fov_y,
        'x_resolution': x_resolution,
        'y_resolution': y_resolution,
        'position': position,
        # 'lookat': lookat,
        'rotation_matrix': rotation_matrix.ravel(),
        'sensor_to_camera_transform_matrix':k.ravel()

    }

    sensor = at3d.sensor._add_null_subpixel_rays(sensor)
    return sensor

class LossAT3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, optimizer):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # gradient = np.zeros(input.shape)
        # gradient[optimizer.mask.data] = optimizer.gradient
        gradient = torch.tensor(optimizer.gradient, dtype=input.dtype, device=input.device)
        ctx.save_for_backward(gradient)
        return torch.tensor(optimizer.loss, dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        gradient, = ctx.saved_tensors
        return gradient * grad_output.clone(), None


class DiffRendererAT3D(object):
    """
    Optimize: Extinction
    --------------------
    Estimate the extinction coefficient based on monochrome radiance measurements.
    In this script, the phase function, albedo and rayleigh scattering are assumed known and are not estimated.

    Measurements are simulated measurements using a forward rendering script
    (e.g. scripts/render_radiance_toa.py).

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/optimize_extinction_lbfgs.py --help

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, cfg):
        self.scatterer_name = 'cloud'
        self._init_solution = False
        self.load_solver(cfg)


        self.state_gen = None

        self.n_jobs = cfg.shdom.n_jobs
        self.min_bound = cfg.cross_entropy.min
        self.max_bound = cfg.cross_entropy.max
        self.use_forward_grid = cfg.shdom.use_forward_grid


        # L2 Loss
        self.image_mean = cfg.data.mean
        self.image_std = cfg.data.std
        self.loss_at3d = LossAT3D().apply
        self.loss_operator = torch.nn.Identity()


    def get_grid(self, grid):
        grid = shdom.Grid(x=grid[0],y=grid[1],z=grid[2])
        return grid

    def load_solver(self,cfg):
        # if cfg.data.dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
        #     path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/solver2.pkl'
        # elif cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m' or cfg.data.dataset_name == 'HAWAII_2000CCN_10cameras_20m':
        #     path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/at3d.nc'
        # else:
        #     NotImplementedError()
        if cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m':
            path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/at3d.nc'
        elif cfg.data.dataset_name == 'HAWAII_2000CCN_10cameras_20m':
            path = '/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/at3d.nc'
        self.sensors, self.solvers, self.rte_grid = at3d.util.load_forward_model(path)
        self.wavelength = self.sensors.get_unique_solvers()[0]


        return

    # def get_projections(self,cfg):
    #     if cfg.data.dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
    #         path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/shdom_projections2.pkl'
    #     elif cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m' or cfg.data.dataset_name == 'BOMEX_50CCN_aux_10cameras_20m' or cfg.data.dataset_name == 'BOMEX_10cameras_20m':
    #         path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/shdom_projections.pkl'
    #     elif cfg.data.dataset_name == 'HAWAII_2000CCN_10cameras_20m':
    #         path = '/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/shdom_projections.pkl'
    #     else:
    #         NotImplementedError()
    #     with open(path, 'rb') as pickle_file:
    #         projection_list = pickle.load(pickle_file)['projections']
    #     sensor_dict = at3d.containers.SensorsDict()
    #
    #     for projection in projection_list:
    #         sensor_dict.add_sensor('SideViewCamera',
    #                                perspective_projection(self.wavelength, projection._fov, projection.resolution[0], projection.resolution[1], projection.position,
    #                                                                   projection._rotation_matrix,
    #                                                                   stokes='I')
    #                                )
    #
    #     return sensor_dict



    def get_medium_estimator(self, mask,volume):
        """
        Generate the medium estimator for optimization.

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired measurements.
        ground_truth: shdom.Scatterer
            The ground truth scatterer


        Returns
        -------
        medium_estimator: shdom.MediumEstimator
            A medium estimator object which defines the optimized parameters.
        """

        # Define the grid for reconstruction
        optical_properties = self.solvers[self.wavelength].medium['cloud'].copy(deep=True)
        optical_properties = optical_properties.drop_vars('extinction')


        grid_to_optical_properties = at3d.medium.GridToOpticalProperties(
            self.rte_grid, 'cloud', self.wavelength, optical_properties
        )

        # UnknownScatterers is a container for all of the unknown variables.
        # Each unknown_scatterer also records the transforms from the abstract state vector
        # to the gridded data in physical coordinates.
        unknown_scatterers = at3d.containers.UnknownScatterers(
            at3d.medium.UnknownScatterer(grid_to_optical_properties,
                                         extinction=(
                                             None, at3d.transforms.StateToGridMask(mask=mask)))
        )



        return unknown_scatterers


    def render(self, cloud, mask, volume, gt_images):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        cloud[0,:,:] = 0
        cloud[-1,:,:] = 0
        cloud[:, 0, :] = 0
        cloud[:, -1, :] = 0
        cloud[:, :,-1] = 0
        cloud[cloud<self.min_bound] = self.min_bound
        cloud[cloud>self.max_bound] = self.max_bound
        gt_images = gt_images.squeeze() # view x H x W
        gt_images *= self.image_std
        gt_images += self.image_mean
        gt_images = list(gt_images)

        unknown_scatterers = self.get_medium_estimator(mask.cpu().numpy(),volume)

        # now we form state_gen which updates the solvers with an input_state.
        solvers_reconstruct = at3d.containers.SolversDict()
        cloud_state = cloud[mask]
        if self.state_gen is None:
            # prepare all of the static inputs to the solver just copy pasted from forward model
            surfaces = OrderedDict()
            numerical_parameters = OrderedDict()
            sources = OrderedDict()
            num_stokes = OrderedDict()
            background_optical_scatterers = OrderedDict()
            for key in self.sensors.get_unique_solvers():
                surfaces[key] = self.solvers[key].surface
                numerical_params = self.solvers[key].numerical_params
                numerical_params['num_mu_bins'] = 2
                numerical_params['num_phi_bins'] = 4
                numerical_parameters[key] = numerical_params
                sources[key] = self.solvers[key].source
                num_stokes[key] = self.solvers[key]._nstokes
                background_optical_scatterers[key] = {'rayleigh': self.solvers[key].medium['rayleigh']}
            # now we form state_gen which updates the solvers with an input_state.
            self.state_gen = at3d.medium.StateGenerator(solvers_reconstruct,
                                                        unknown_scatterers, surfaces,
                                                        numerical_parameters, sources,
                                                        background_optical_scatterers,
                                                        num_stokes)

        else:
            self.state_gen = at3d.medium.StateGenerator(solvers_reconstruct,
                                               unknown_scatterers, self.state_gen._surfaces,
                                               self.state_gen._numerical_parameters, self.state_gen._sources,
                                                    self.state_gen._background_optical_scatterers, self.state_gen._num_stokes)

        self.state_gen(cloud_state.detach().cpu().numpy())
        objective_function = at3d.optimize.ObjectiveFunction.LevisApproxUncorrelatedL2(
    self.sensors, solvers_reconstruct, self.sensors, unknown_scatterers, self.state_gen,
            self.state_gen.project_gradient_to_state,
    parallel_solve_kwargs={'n_jobs': self.n_jobs, 'verbose': True},
  gradient_kwargs={'cost_function': 'L2', 'exact_single_scatter':False},
  uncertainty_kwargs={'add_noise': False},
  min_bounds=self.min_bound, max_bounds=self.max_bound)

        optimizer = at3d.optimize.Optimizer(objective_function)
        loss, gradient = optimizer.objective(cloud_state.detach().cpu().numpy())

        gt_images = np.array(gt_images)
        images = self.sensors.get_images('SideViewCamera')
        images = [image.I.data for image in images]
        # images = []
        # for im, im_gt,masked_im in zip(pixels,gt_images[0],mask_images[0]):
        #     masked_image = np.zeros(im_gt.size)
        #     masked_image[masked_im.ravel('F')] = im
        #     images.append(masked_image.reshape(masked_im.shape).T)
        # images = np.array(images)
        # vmax = max(gt_images[5].max().item(),images[5].max())
        # f, axarr = plt.subplots(1, 3)
        # axarr[0].imshow(gt_images[5],vmin=0,vmax=vmax)
        # axarr[1].imshow(images[5], vmin=0, vmax=vmax)
        # axarr[2].imshow(np.abs(gt_images[5] - images[5]))
        # plt.show()

        plt.scatter(np.array(gt_images).ravel(),np.array(images).ravel())
        plt.axis('square')
        plt.show()
        f, axarr = plt.subplots(3, len(images))
        for ax, image,gt in zip(axarr.T, images,gt_images):
            ax[0].imshow(image)
            ax[1].imshow(gt)
            ax[2].imshow(np.abs(gt-image))
            # ax.invert_xaxis()
            # ax.invert_yaxis()
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
        plt.title(f'loss={loss}')
        plt.tight_layout()
        plt.show()

        # print(np.sum(np.array(gt_images)-images)**2)
        self.loss = loss #/ np.sum(gt_images**2) #/ mask_images.sum()
        self.images = images
        self.gt_images = gt_images
        self.gradient = gradient
        l2_loss = self.loss_at3d(cloud_state,self) #/ gt_images.size #/ np.sum(gt_images**2)
        return self.loss_operator(l2_loss)




