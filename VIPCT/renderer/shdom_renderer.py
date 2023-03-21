import os, time
import numpy as np
import argparse
import shdom
import dill as pickle
import torch
from VIPCT.renderer.shdom_util import Monotonous
class ForwardModel(object):
    def __init__(self, input_dir):
        super(ForwardModel, self).__init__()
        medium, solver, measurements = shdom.load_forward_model(input_dir)
        # self.data = np.load(input_dir + '.npz')
        self.camera = measurements.camera
        self.atmospheric_grid = medium.grid
        self.air = medium.get_scatterer('air')
        self.droplets = medium.get_scatterer('cloud')
        self.x_shape = self.droplets.lwc.shape
        self.mask = self.droplets.lwc.data > 0
        self.scene_parameters = solver._scene_parameters
        self.numerical_parameters = solver._numerical_parameters
        self.solvers_list = []

    def init_solution(self):
        self.solvers_list = []
    def forward(self, x):
        # x.shape = (N_batch, Nx, Ny, Nz)
        # solvers_list = []
        if isinstance(x, torch.Tensor):
            x_type = 'tensor'
            device = x.device
            x = x.detach().cpu().numpy()

        else:
            x_type = 'numpy'
        if len(x.shape) == 3:
            x = x[None, ...]
        solvers_list = self.solvers_list
        if len(solvers_list) == 0:
            for xi in x:
                lwc = self.droplets._lwc
                lwc._data = xi.reshape(self.x_shape)
                rte_solver = shdom.RteSolver(self.scene_parameters, self.numerical_parameters)
                atmosphere = shdom.Medium(self.atmospheric_grid)
                atmosphere.add_scatterer(self.air, name='air')
                atmosphere.add_scatterer(self.droplets, name='cloud')
                rte_solver.set_medium(atmosphere)
                solvers_list.append(rte_solver)
        else:
            for xi, solver in zip(x, solvers_list):
                lwc = self.droplets._lwc
                lwc._data = xi.reshape(self.x_shape)
                atmosphere = shdom.Medium(self.atmospheric_grid)
                atmosphere.add_scatterer(self.air, name='air')
                atmosphere.add_scatterer(self.droplets, name='cloud')
                solver.set_medium(atmosphere)

        rte_solver_array = shdom.RteSolverArray(solvers_list)
        rte_solver_array.solve(maxiter=1, verbose=False)
        images = np.array(self.camera.render(rte_solver_array, n_jobs=8))
        if len(images.shape) == 3:
            images = images[..., None]
        y = images.transpose((3, 0, 1, 2))  # .view((x.shape[0],-1))
        if x_type == 'tensor':
            y = torch.tensor(y).to(device=device)
        return y

    def __call__(self, x):
        return self.forward(x)


class DiffRenderer(object):

    def __init__(self, n_jobs=1, init_solution=True):
        self._medium = None
        self._rte_solver = None
        self._measurements = None
        self._writer = None
        self._images = None
        self._iteration = 0
        self._loss = None
        self._n_jobs = n_jobs
        self._init_solution = init_solution
        self._mask = None

    def set_measurements(self, measurements):
        """
        Set the measurements (data-fit constraints)

        Parameters
        ----------
        measurements: shdom.Measurements
            A measurements object storing the acquired images and sensor geometry
        """
        self._measurements = measurements

    def set_medium_estimator(self, medium_estimator):
        """
        Set the MediumEstimator for the optimizer.

        Parameters
        ----------
        medium_estimator: shdom.MediumEstimator
            The MediumEstimator
        """
        self._medium = medium_estimator

    def set_mask(self, mask):
        """
        Set the Mask for the optimizer.

        Parameters
        ----------
        mask: numpy array
            The Mask
        """
        self._mask = mask

    def objective_fun(self, state):
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
        if self.iteration == 0:
            self.init_optimizer()
        # self._iteration += 1
        self.set_state(state)
        gradient, loss, images = self.medium.compute_gradient(
            rte_solvers=self.rte_solver,
            measurements=self.measurements,
            n_jobs=self.n_jobs
        )
        # import matplotlib.pyplot as plt
        # f, axarr = plt.subplots(1, len(images), figsize=(16, 16))
        # for ax, image in zip(axarr, images):
        #     ax.imshow(image)
        #     ax.invert_xaxis()
        #     ax.invert_yaxis()
        #     ax.axis('off')
        # plt.show()
        self._loss = loss
        self._images = images
        return loss, gradient



    def init_optimizer(self):
        """
        Initialize the optimizer.
        This means:
          1. Setting the RteSolver medium
          2. Initializing a solution
          3. Computing the direct solar flux derivatives
          4. Counting the number of unknown parameters
        """

        assert self.rte_solver.num_solvers == self.measurements.num_channels == self.medium.num_wavelengths, \
            'RteSolver has {} solvers, Measurements have {} channels and Medium has {} wavelengths'.format(
                self.rte_solver.num_solvers, self.measurements.num_channels, self.medium.num_wavelengths)

        self.rte_solver.set_medium(self.medium)
        self.rte_solver.init_solution()
        self.medium.compute_direct_derivative(self.rte_solver)
        self._num_parameters = self.medium.num_parameters

    def callback(self, state):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector
        """
        self._iteration += 1

        # Writer callback functions
        if self.writer is not None:
            for callbackfn, kwargs in zip(self.writer.callback_fns, self.writer.kwargs):
                time_passed = time.time() - kwargs['ckpt_time']
                if time_passed > kwargs['ckpt_period']:
                    kwargs['ckpt_time'] = time.time()
                    callbackfn(kwargs)
    def get_bounds(self):
        """
        Retrieve the bounds for every parameter from the MediumEstimator (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        return self.medium.get_bounds()

    def get_state(self):
        """
        Retrieve MediumEstimator state

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        return self.medium.get_state()

    def set_state(self, state):
        """
        Set the state of the optimization. This means:
          1. Setting the MediumEstimator state
          2. Updating the RteSolver medium
          3. Computing the direct solar flux
          4. Computing the current RTE solution with the previous solution as an initialization

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        self.medium.set_state(state)
        self.rte_solver.set_medium(self.medium)
        if self._init_solution is False:
            self.rte_solver.make_direct()
        self.rte_solver.solve(maxiter=100, init_solution=self._init_solution, verbose=False)

    def save_state(self, path):
        """
        Save Optimizer state to file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self.get_state(), -1))
        file.close()

    def load_state(self, path):
        """
        Load Optimizer from file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'rb')
        data = file.read()
        file.close()
        state = pickle.loads(data)
        self.set_state(state)

    @property
    def rte_solver(self):
        return self._rte_solver

    @property
    def medium(self):
        return self._medium

    @property
    def measurements(self):
        return self._measurements

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def writer(self):
        return self._writer

    @property
    def iteration(self):
        return self._iteration

    @property
    def mask(self):
        return self._mask

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def loss(self):
        return self._loss

    @property
    def images(self):
        return self._images


class LossSHDOM(torch.autograd.Function):
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


class DiffRendererSHDOM(object):
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
        self.projections = self.get_projections(cfg)
        self.cameras = shdom.Camera(shdom.RadianceSensor(), self.projections)
        self.rte_solver = self.get_rte_solver(cfg)
        self.n_jobs = cfg.shdom.n_jobs
        self.min_bound = cfg.cross_entropy.min
        self.max_bound = cfg.cross_entropy.max
        self.use_forward_grid = cfg.shdom.use_forward_grid
        self.mie_base_path = '../../../pyshdom/mie_tables/polydisperse/Water_<wavelength>nm.scat'
        self.add_rayleigh = cfg.shdom.add_rayleigh

        parser = argparse.ArgumentParser()
        # CloudGenerator = getattr(shdom.generate, 'Homogenous')
        CloudGenerator = Monotonous
        parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if self.add_rayleigh:
            AirGenerator = shdom.generate.AFGLSummerMidLatAir
            parser = AirGenerator.update_parser(parser)
        self.args = parser.parse_args()
        self.args.air_path = '../../../pyshdom/ancillary_data/AFGL_summer_mid_lat.txt'
        self.args.air_max_alt = 5
        self.args.extinction = 0
        self.cloud_generator = CloudGenerator(self.args)
        self.table_path = self.mie_base_path.replace('<wavelength>', '{}'.format(shdom.int_round(self.wavelength)))
        self.cloud_generator.add_mie(self.table_path)
        self.air_generator = AirGenerator(self.args) if AirGenerator is not None else None



        # L2 Loss
        self.image_mean = cfg.data.mean
        self.image_std = cfg.data.std
        self.loss_shdom = LossSHDOM().apply
        self.loss_operator = torch.nn.Identity()

    def get_grid(self, grid):
        grid = shdom.Grid(x=grid[0],y=grid[1],z=grid[2])
        return grid

    def get_rte_solver(self,cfg):
        if cfg.data.dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
            path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/solver2.pkl'
        elif cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m':
            path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/train/solver.pkl'
        else:
            NotImplementedError()
        solver = shdom.RteSolver()
        solver.load_params(path)
        self.wavelength = solver.wavelength
        return shdom.RteSolverArray([solver])

    def get_projections(self,cfg):
        if cfg.data.dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
            path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/shdom_projections2.pkl'
        elif cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m':
            path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/train/shdom_projections.pkl'
        else:
            NotImplementedError()
        with open(path, 'rb') as pickle_file:
            projection_list = pickle.load(pickle_file)['projections']
        return shdom.MultiViewProjection(projection_list)



    def get_medium_estimator(self, cloud_extinction, mask, volume):
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
        if self.use_forward_grid:
            extinction_grid = albedo_grid = phase_grid = self.get_grid(volume._grid[0])
        else:
            NotImplementedError()
            # extinction_grid = albedo_grid = phase_grid = self.get_grid()
        grid = extinction_grid + albedo_grid + phase_grid


        # Define the known albedo and phase: either ground-truth or specified, but it is not optimized.


        albedo = self.cloud_generator.get_albedo(self.wavelength, albedo_grid)

        path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/medium_3827.pkl'
        medium = shdom.Medium()
        medium.load(path)


        medium.scatterers['cloud']._reff = self.cloud_generator.get_reff(medium.scatterers['cloud'].grid)
        # medium.scatterers['cloud']._veff._data[:,:,:] = 0.01
        medium.scatterers['cloud']._veff._data[cloud_extinction>0] = self.cloud_generator.get_veff(medium.scatterers['cloud'].grid)._data[cloud_extinction>0]
        phase = medium.scatterers['cloud'].get_phase(self.wavelength)

        # phase = self.cloud_generator.get_phase(self.wavelength, mask=cloud_extinction>1.0, grid = phase_grid)

        ext = self.cloud_generator.get_extinction(grid=grid)
        ext._data[mask] = cloud_extinction[mask]
        ext._data[np.bitwise_not(mask)] = 0
        extinction = shdom.GridDataEstimator(ext,
                                             min_bound=self.min_bound + 1e-3,
                                             max_bound=self.max_bound)
        cloud_estimator = shdom.OpticalScattererEstimator(self.wavelength, extinction, albedo, phase)
        cloud_mask = shdom.GridData(cloud_estimator.grid, mask)
        cloud_estimator.set_mask(cloud_mask)

        # Create a medium estimator object (optional Rayleigh scattering)
        medium_estimator = shdom.MediumEstimator()
        medium_estimator.add_scatterer(cloud_estimator, self.scatterer_name) #MUST BE FIRST!!!

        if self.add_rayleigh:
            air = self.air_generator.get_scatterer(self.wavelength)
            medium_estimator.set_grid(cloud_estimator.grid + air.grid)
            medium_estimator.add_scatterer(air, 'air')
        else:
            medium_estimator.set_grid(cloud_estimator.grid)


        ext_fixed = self.cloud_generator.get_extinction(grid=grid)
        ext_fixed._data[np.bitwise_not(mask)] = cloud_extinction[np.bitwise_not(mask)]
        ext_fixed._data[mask] = 0
        cloud_fixed = shdom.OpticalScatterer(self.wavelength, ext_fixed, albedo, phase)
        medium_estimator.add_scatterer(cloud_fixed, 'cloud_fixed')


        return medium_estimator

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter from the MediumEstimator (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        return self.medium.get_bounds()

    def get_state(self):
        """
        Retrieve MediumEstimator state

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        return self.medium.get_state()

    def set_state(self, state):
        """
        Set the state of the optimization. This means:
          1. Setting the MediumEstimator state
          2. Updating the RteSolver medium
          3. Computing the direct solar flux
          4. Computing the current RTE solution with the previous solution as an initialization

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        self.medium.set_state(state)
        self.rte_solver.set_medium(self.medium)
        if self._init_solution is False:
            self.rte_solver.make_direct()
        self.rte_solver.solve(maxiter=100, init_solution=self._init_solution, verbose=False)

    def init_optimizer(self):
        """
        Initialize the optimizer.
        This means:
          1. Setting the RteSolver medium
          2. Initializing a solution
          3. Computing the direct solar flux derivatives
          4. Counting the number of unknown parameters
        """

        assert self.measurements.num_channels == self.medium.num_wavelengths, \
            'Measurements have {} channels and Medium has {} wavelengths'.format(
                self.measurements.num_channels, self.medium.num_wavelengths)

        self.rte_solver.set_medium(self.medium)
        self.rte_solver.init_solution()
        self.medium.compute_direct_derivative(self.rte_solver)
        self._num_parameters = self.medium.num_parameters

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

        gt_images *= self.image_std
        gt_images += self.image_mean
        gt_images = gt_images.cpu().numpy()
        self.measurements = shdom.Measurements(self.cameras,images=gt_images,wavelength=self.wavelength)
        cloud_state = cloud[mask]

        self.medium = self.get_medium_estimator(cloud.detach().cpu().numpy(),mask.cpu().numpy(), volume)
        # cloud_estimator = self.medium.scatterers['cloud']
        # cloud_mask = shdom.GridData(cloud_estimator.grid, (mask.cpu().numpy()))
        # cloud_estimator.set_mask(cloud_mask)
        self.init_optimizer()

        self.set_state(cloud_state.detach().cpu().numpy())

        # self._iteration += 1
        gradient, loss, images = self.medium.compute_gradient(
            rte_solvers=self.rte_solver,
            measurements=self.measurements,
            n_jobs=self.n_jobs
        )
        # import matplotlib.pyplot as plt
        # f, axarr = plt.subplots(1, len(images), figsize=(16, 16))
        # for ax, image in zip(axarr, images):
        #     ax.imshow(image)
        #     ax.invert_xaxis()
        #     ax.invert_yaxis()
        #     ax.axis('off')
        # plt.show()
        self.loss = loss
        self.images = images
        self.gt_images = gt_images
        self.gradient = gradient
        l2_loss = self.loss_shdom(cloud_state,self) / gt_images.size
        return self.loss_operator(l2_loss)




