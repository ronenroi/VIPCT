
# This file contains auxiliry code for result visualization.
# It is based on pySHDOM source code ('https://github.com/aviadlevis/pyshdom') by Aviad Levis
# Copyright (c) Aviad Levis et al.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper that describes the readme file:
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


from typing import List, Optional, Tuple

import torch
# from pytorch3d.vis.plotly_vis import plot_scene
import tensorboardX as tb
import matplotlib.pyplot as plt
import numpy as np

class SummaryWriter(object):
    """
    A wrapper for tensorboardX summarywriter with some basic summary writing implementation.
    This wrapper enables logging of images, error measures and loss with pre-determined temporal intervals into tensorboard.

    To view the summary of this run (and comparisons to all subdirectories):
        tensorboard --logdir LOGDIR

    Parameters
    ----------
    log_dir: str
        The directory where the log will be saved
    """
    def __init__(self, log_dir=None):
        self._dir = log_dir
        self._tf_writer = tb.SummaryWriter(log_dir) if log_dir is not None else None
        self._callback_fns = []
        self._kwargs = []
        self._iter = 0
        self._dataset = 'train'

    def add_callback_fn(self, callback_fn, kwargs=None):
        """
        Add a callback function to the callback function list

        Parameters
        ----------
        callback_fn: bound method
            A callback function to push into the list
        kwargs: dict, optional
            A dictionary with optional keyword arguments for the callback function
        """
        self._callback_fns.append(callback_fn)
        self._kwargs.append(kwargs)


    def monitor_loss(self, loss):
        """
        Monitor the loss.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'loss': loss,
            'ckpt_time': self._iter,
            'title': 'loss',
        }
        self.loss_cbfn(kwargs)




    def monitor_scatterer_error(self, delta, epsilon, name='extinction'):
        """

        """
        kwargs = {
            'delta': delta,
            'epsilon': epsilon,
            'title': [f'{name}_delta/', f'{name}_epsilon/'],
        }
        self.error_cbfn(kwargs)

    def monitor_scatter_plot(self, est_param, gt_param, ind=0, dilute_percent=0.8, name='extinction'):
        """
        Monitor scatter plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'est_param': est_param,
            'gt_param': gt_param,
            'title': '{}_scatter_plot/{}-{}'.format(name,self._dataset,ind),
            'percent': dilute_percent,
        }
        self.scatter_plot_cbfn(kwargs)


    # def monitor_horizontal_mean(self, estimator_name, ground_truth, ground_truth_mask=None, ckpt_period=-1):
    #     """
    #     Monitor horizontally averaged quantities and compare to ground truth over iterations.
    #
    #     Parameters
    #     ----------
    #     estimator_name: str
    #         The name of the scatterer to monitor
    #     ground_truth: shdom.Scatterer
    #         The ground truth medium.
    #     ground_truth_mask: shdom.GridData
    #         The ground-truth mask of the estimator
    #     ckpt_period: float
    #        time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
    #     """
    #     kwargs = {
    #         'ckpt_period': ckpt_period,
    #         'ckpt_time': time.time(),
    #         'title': '{}/horizontal_mean/{}',
    #         'mask': ground_truth_mask
    #     }
    #     self.add_callback_fn(self.horizontal_mean_cbfn, kwargs)
    #     if hasattr(self, '_ground_truth'):
    #         self._ground_truth[estimator_name] = ground_truth
    #     else:
    #         self._ground_truth = OrderedDict({estimator_name: ground_truth})

    # def monitor_domain_mean(self, estimator_name, ground_truth, ckpt_period=-1):
    #     """
    #     Monitor domain mean and compare to ground truth over iterations.
    #
    #     Parameters
    #     ----------
    #     estimator_name: str
    #         The name of the scatterer to monitor
    #     ground_truth: shdom.Scatterer
    #         The ground truth medium.
    #     ckpt_period: float
    #        time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
    #     """
    #     kwargs = {
    #         'ckpt_period': ckpt_period,
    #         'ckpt_time': time.time(),
    #         'title': '{}/mean/{}'
    #     }
    #     self.add_callback_fn(self.domain_mean_cbfn, kwargs)
    #     if hasattr(self, '_ground_truth'):
    #         self._ground_truth[estimator_name] = ground_truth
    #     else:
    #         self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_images(self, gt_images):
        """
        Monitor the GT images

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        """
        num_images = gt_images.shape[0]

        vmax = [image.max() * 1.25 for image in gt_images]

        kwargs = {
            'images': gt_images,
            'title': ['Acquired/view{}'.format(view) for view in range(num_images)],
            'vmax': vmax,
        }
        self.images_cbfn(kwargs)

    # def monitor_power_spectrum(self, estimator_name, ground_truth, ckpt_period=-1):
    #     """
    #     TODO
    #     """
    #     kwargs = {
    #         'ckpt_period': ckpt_period,
    #         'ckpt_time': time.time(),
    #         'title': '{}/isotropic_power_spectrum/{}',
    #     }
    #     self.add_callback_fn(self.isotropic_power_spectrum_cbfn, kwargs)
    #     if hasattr(self, '_ground_truth'):
    #         self._ground_truth[estimator_name] = ground_truth
    #     else:
    #         self._ground_truth = OrderedDict({estimator_name: ground_truth})


    def loss_cbfn(self,  kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalars(kwargs['title'], {
            self._dataset: kwargs['loss']}, self._iter)



    def images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.write_image_list(self._iter, kwargs['images'], kwargs['title'], kwargs['vmax'])


    def error_cbfn(self, kwargs):
        """
        Callback function for monitoring parameter error measures.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """


        # delta = (torch.linalg.norm(kwargs['est_param'], 1) - torch.linalg.norm(kwargs['gt_param'], 1)) / torch.linalg.norm(kwargs['gt_param'], 1)
        # epsilon = torch.linalg.norm((kwargs['est_param'] - kwargs['gt_param']), 1) / torch.linalg.norm(kwargs['gt_param'],1)
        self.tf_writer.add_scalars(kwargs['title'][0], {
            self._dataset: kwargs['delta']}, self._iter)
        self.tf_writer.add_scalars(kwargs['title'][1], {
            self._dataset: kwargs['epsilon']}, self._iter)

    # def domain_mean_cbfn(self, kwargs):
    #     """
    #     Callback function for monitoring domain averages of parameters.
    #
    #     Parameters
    #     ----------
    #     kwargs: dict,
    #         keyword arguments
    #     """
    #     for scatterer_name, gt_scatterer in self._ground_truth.items():
    #         est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
    #         for parameter_name, parameter in est_scatterer.estimators.items():
    #             if parameter.type == 'Homogeneous':
    #                 est_param = parameter.data
    #             else:
    #                 est_param = parameter.data.mean()
    #
    #             ground_truth = getattr(gt_scatterer, parameter_name)
    #             if ground_truth.type == 'Homogeneous':
    #                 gt_param = ground_truth.data
    #             else:
    #                 gt_param = ground_truth.data.mean()
    #
    #             self.tf_writer.add_scalars(
    #                 main_tag=kwargs['title'].format(scatterer_name, parameter_name),
    #                 tag_scalar_dict={'estimated': est_param, 'true': gt_param},
    #                 global_step=self._iter
    #             )

    # def horizontal_mean_cbfn(self, kwargs):
    #     """
    #     Callback function for monitoring horizontal averages of parameters.
    #
    #     Parameters
    #     ----------
    #     kwargs: dict,
    #         keyword arguments
    #     """
    #     for scatterer_name, gt_scatterer in self._ground_truth.items():
    #         est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
    #
    #         common_grid = est_scatterer.grid + gt_scatterer.grid
    #         a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
    #         b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
    #         common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)
    #
    #         for parameter_name, parameter in est_scatterer.estimators.items():
    #             ground_truth = getattr(gt_scatterer, parameter_name)
    #
    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("ignore", category=RuntimeWarning)
    #
    #
    #                 est_parameter_masked = copy.deepcopy(parameter).resample(common_grid)
    #                 est_parameter_masked.apply_mask(common_mask)
    #                 est_param = est_parameter_masked.data
    #                 est_param[np.bitwise_not(common_mask.data)] = np.nan
    #                 est_param = np.nan_to_num(np.nanmean(est_param,axis=(0,1)))
    #
    #                 gt_param_masked = copy.deepcopy(ground_truth).resample(common_grid)
    #                 gt_param_masked.apply_mask(common_mask)
    #                 gt_param = gt_param_masked.data
    #                 gt_param[np.bitwise_not(common_mask.data)] = np.nan
    #                 gt_param = np.nan_to_num(np.nanmean(gt_param,axis=(0,1)))
    #
    #             fig, ax = plt.subplots()
    #             ax.set_title('{} {}'.format(scatterer_name, parameter_name), fontsize=16)
    #             ax.plot(est_param, common_grid.z, label='Estimated')
    #             ax.plot(gt_param, common_grid.z, label='True')
    #             ax.legend()
    #             ax.set_ylabel('Altitude [km]', fontsize=14)
    #             self.tf_writer.add_figure(
    #                 tag=kwargs['title'].format(scatterer_name, parameter_name),
    #                 figure=fig,
    #                 global_step=self._iter
    #             )

    def scatter_plot_cbfn(self, kwargs):
        """
        Callback function for monitoring scatter plot of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        est_param = kwargs['est_param'].cpu().detach().numpy().ravel()
        gt_param = kwargs['gt_param'].cpu().detach().numpy().ravel()
        rho = np.corrcoef(est_param, gt_param)[1, 0]
        num_params = gt_param.size
        rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
        max_val = max(gt_param.max(), est_param.max())
        fig, ax = plt.subplots()
        ax.set_title(r' ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(100 * kwargs['percent'], rho),
                     fontsize=16)
        ax.scatter(gt_param[rand_ind], est_param[rand_ind], facecolors='none', edgecolors='b')
        ax.set_xlim([0, 1.1*max_val])
        ax.set_ylim([0, 1.1*max_val])
        ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
        ax.set_ylabel('Estimated', fontsize=14)
        ax.set_xlabel('True', fontsize=14)

        self.tf_writer.add_figure(
            tag=kwargs['title'],
            figure=fig,
            global_step=self._iter
        )

    # def isotropic_power_spectrum_cbfn(self, kwargs):
    #     """
    #     TODO
    #     """
    #     for scatterer_name, gt_scatterer in self._ground_truth.items():
    #
    #         est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
    #         grid = est_scatterer.grid + gt_scatterer.grid
    #         a = est_scatterer.get_mask(threshold=0.0).resample(grid,method='nearest')
    #         b = gt_scatterer.get_mask(threshold=0.0).resample(grid,method='nearest')
    #         common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=grid)
    #
    #         x,y,z = np.fft.fftfreq(grid.nx,d=grid.x[1]-grid.x[0]),np.fft.fftfreq(grid.ny,d=grid.y[1]-grid.y[0]), \
    #             np.fft.fftfreq(grid.nz,d=grid.z[1]-grid.z[0])
    #         X,Y,Z = np.meshgrid(np.fft.fftshift(y),np.fft.fftshift(x),np.fft.fftshift(z))
    #         isotropic_wavenumber = np.sqrt(X**2+Y**2+Z**2)
    #         nyquist_mask = (np.abs(X)<0.5/(grid.x[1]-grid.x[0])) & (np.abs(Y)<0.5/(grid.y[1]-grid.y[0])) & (np.abs(Z)<0.5/(grid.z[1]-grid.z[0]))
    #         bins = stats.mstats.mquantiles(isotropic_wavenumber[nyquist_mask],[i/20 for i in range(21)])
    #         bin_centres = np.array([(bins[i+1]+bins[i])/2.0 for i in range(len(bins)-1)])
    #
    #
    #         for parameter_name, parameter in est_scatterer.estimators.items():
    #             ground_truth = getattr(gt_scatterer, parameter_name)
    #
    #             grid_gt = copy.copy(ground_truth).resample(grid)
    #             grid_gt.apply_mask(common_mask)
    #             grid_parameter = copy.copy(parameter).resample(grid)
    #             grid_parameter.apply_mask(common_mask)
    #
    #             gt_spec = np.abs(np.fft.fftshift(np.fft.fftn((grid_gt.data - grid_gt.data.mean())/grid_gt.data.std())))**2
    #             param_spec = np.abs(np.fft.fftshift(np.fft.fftn((grid_parameter.data - grid_parameter.data.mean())/grid_parameter.data.std())))**2
    #
    #             gt_resampled, bin_edge, number = stats.binned_statistic_dd(isotropic_wavenumber[nyquist_mask], gt_spec[nyquist_mask],
    #                             bins=[bins,],statistic='mean')
    #             param_resampled, bin_edge, number = stats.binned_statistic_dd(isotropic_wavenumber[nyquist_mask], param_spec[nyquist_mask],
    #                             bins=[bins,],statistic='mean')
    #
    #             fig, ax = plt.subplots()
    #             ax.set_title(r'{} {}: Isotropic power spectrum'.format(scatterer_name, parameter_name ),
    #                          fontsize=16)
    #             ax.loglog(bin_centres, gt_resampled, '-o',label='True')
    #             ax.loglog(bin_centres, param_resampled, 'x-', label='Estimated')
    #             ax.legend()
    #             self.tf_writer.add_figure(
    #                 tag=kwargs['title'].format(scatterer_name, parameter_name),
    #                 figure=fig,
    #                 global_step=self._iter
    #             )

    def write_image_list(self, global_step, images, titles, vmax=None):
        """
        Write an image list to tensorboardX.

        Parameters
        ----------
        global_step: integer,
            The global step of the optimizer.
        images: list
            List of images to be logged onto tensorboard.
        titles: list
            List of strings that will title the corresponding images on tensorboard.
        vmax: list or scalar, optional
            List or a single of scaling factor for the image contrast equalization
        """
        if np.isscalar(vmax) or vmax is None:
            vmax = [vmax]*len(images)

        assert len(images) == len(titles), 'len(images) != len(titles): {} != {}'.format(len(images), len(titles))
        assert len(vmax) == len(titles), 'len(vmax) != len(images): {} != {}'.format(len(vmax), len(titles))

        for image, title, vm in zip(images, titles, vmax):
            if (image.shape[0] in (1,3,4)): #polarized
                #there is some overlap in this condition with a multispectral unpolarized case
                # with a very small number of pixels in the first dimension.
                image = image[0]

            if image.ndim==3: # polychromatic
                img_tensor = image[:,:,:]/ image.max()

            else:
                img_tensor = image[:,:,np.newaxis] / image.max()
            self.tf_writer.add_images(tag=title,
                img_tensor=img_tensor,
                dataformats='HWN',
                global_step=global_step
                )


    @property
    def callback_fns(self):
        return self._callback_fns

    @property
    def dir(self):
        return self._dir

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def tf_writer(self):
        return self._tf_writer

# def visualize_ct_outputs(
#     out: dict, output_cache: List, viz: Visdom, visdom_env: str
# ):
#     """
#     Visualizes the outputs of the `RadianceFieldRenderer`.
#
#     Args:
#         out: An output of the validation rendering pass.
#         output_cache: A list with outputs of several training render passes.
#         viz: A visdom connection object.
#         visdom_env: The name of visdom environment for visualization.
#     """
#
#     # Show the training images.
#     ims = torch.stack([o["image"] for o in output_cache])[:,0,...]
#     # ims = torch.cat(list(ims), dim=0)
#     viz.images(
#         ims.clamp(0., 1.),#.permute(2, 0, 1),
#         env=visdom_env,
#         win="images",
#         opts={"title": "train_images"},
#     )
#
#     # # Show the coarse and fine renders together with the ground truth images.
#     # ims_full = torch.cat(
#     #     [
#     #         nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
#     #         for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
#     #     ],
#     #     dim=2,
#     # )
#     # viz.image(
#     #     ims_full,
#     #     env=visdom_env,
#     #     win="images_full",
#     #     opts={"title": "coarse | fine | target"},
#     # )
#
#     # Make a 3D plot of training cameras and their emitted rays.
#     # camera_trace = {
#     #     f"camera_{ci:03d}": o.cpu() for ci, o in enumerate(output_cache[0]["camera"])
#     # }
#     # ray_pts_trace = {
#     #     f"ray_pts_{ci:03d}": Pointclouds(
#     #         ray_bundle_to_ray_points(o["coarse_ray_bundle"])
#     #         .detach()
#     #         .cpu()
#     #         .view(1, -1, 3)validation_epoch_interval
#     #     )
#     #     for ci, o in enumerate(output_cache)
#     # }
#     # plotly_plot = plot_scene(
#     #     {
#     #         "training_scene": {
#     #             **camera_trace,
#     #             # **ray_pts_trace,
#     #         },
#     #     },
#     #     pointcloud_max_points=5000,
#     #     pointcloud_marker_size=1,
#     #     camera_scale=0.3,
#     # )
#     # viz.plotlyplot(plotly_plot, env=visdom_env, win="scenes")
#
#     # scatter plots
#     viz.scatter(
#         X=torch.cat((out['volume'],out['output']),1),
#         env=visdom_env, win="scatter",
#         opts={"title": "Training scatter"}
#     )
