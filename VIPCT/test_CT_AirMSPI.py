# This file contains the main script for VIP-CT evaluation on AirMSPI data.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
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

import os, time
import warnings
import hydra
import numpy as np
from VIPCT.airmspi_dataset import get_real_world_airmspi_datasets, trivial_collate
from VIPCT.CTnet import *
from omegaconf import OmegaConf
from omegaconf import DictConfig
from VIPCT.cameras import AirMSPICameras
import scipy.io as sio

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

@hydra.main(config_path=CONFIG_DIR, config_name="vip-ct_test_airmspi")
def main(cfg: DictConfig):

    # Set the relevant seeds for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run
    if torch.cuda.is_available() and cfg.debug == False:
        n_device = torch.cuda.device_count()
        cfg.gpu = 0 if n_device==1 else cfg.gpu
        device = f"cuda:{cfg.gpu}"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
        )
        device = "cpu"

    log_dir = os.getcwd()
    log_dir = log_dir.replace('outputs','test_results')
    results_dir = log_dir
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)
    if len(results_dir) > 0:
        # Make the root of the experiment directory
        os.makedirs(results_dir, exist_ok=True)

    resume_cfg_path = os.path.join(checkpoint_resume_path.split('/checkpoints')[0],'.hydra/config.yaml')
    net_cfg = OmegaConf.load(resume_cfg_path)
    cfg = OmegaConf.merge(net_cfg,cfg)

    # Initialize VIP-CT model
    model = CTnetAirMSPI(cfg=cfg, n_cam=cfg.data.n_cam)

    # Load model
    assert os.path.isfile(checkpoint_resume_path)
    print(f"Resuming from checkpoint {checkpoint_resume_path}.")
    loaded_data = torch.load(checkpoint_resume_path, map_location=device)
    model.load_state_dict(loaded_data["model"])
    model.to(device)
    model.eval().float()


    batch_time_net = []

    val_dataset = get_real_world_airmspi_datasets(cfg=cfg)
    val_image, grid, images_mapping_list, pixel_centers_list, masks = val_dataset[0]
    val_image = torch.tensor(val_image,device=device).float()[None]

    # masks = sio.loadmat('/wdata/yaelsc/AirMSPI_raw_data/raw_data/mask_72x72x32_vox50x50x40m.mat')['mask']
    # mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
    # mapping_path = '/wdata/yaelsc/AirMSPI_raw_data/raw_data/voxel_pixel_list72x72x32_BOMEX_img350x350.pkl'
    # mapping_path = '/wdata/roironen/Data/voxel_pixel_list72x72x32_BOMEX_img350x350_processed.pkl'
    # with open(mapping_path, 'rb') as f:
    #     mapping = pickle.load(f)
    # images_mapping_list = sio.loadmat(mapping_path)['map']
    # pixel_center_path = '/wdata/roironen/Data/AirMSPI-Varying/test/20130206_202754Z_NorthPacificOcean-32N123W_vadim_measurement.mat'
    # image_size = [350, 350]
    # with open(mapping_path, 'rb') as f:
    #     mapping = pickle.load(f)
    # images_mapping_list = []
    # pixel_centers_list = []
    # pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    # camera_ind = 0
    # for _, map in mapping.items():
    #     voxels_list = []
    #     pixel_list = []
    #     v = map.values()
    #     voxels = np.array(list(v),dtype=object)
    #     for i, voxel in enumerate(voxels):
    #         if len(voxel)>0:
    #             pixels = np.unravel_index(voxel, np.array(image_size))
    #             mean_px = np.mean(pixels,1)
    #             voxels_list.append(mean_px)
    #             pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #         else:
    #             voxels_list.append([-100000,-100000])
    #             pixel_list.append([-10000, -10000, -10000])
    #
    #     camera_ind += 1
    #     images_mapping_list.append(voxels_list)
    #     pixel_centers_list.append(pixel_list)
    #
    # with open('/wdata/roironen/Data/AirMSPI-Varying/test/rebat_images_mapping_lists72x72x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(images_mapping_list, f, pickle.HIGHEST_PROTOCOL)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/test/rebat_pixel_centers_lists72x72x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(pixel_centers_list, f, pickle.HIGHEST_PROTOCOL)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/test/rebat_images_mapping_lists72x72x32_BOMEX_img350x350.pkl', 'rb') as f:
    #     images_mapping_list = pickle.load(f)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/test/rebat_pixel_centers_lists72x72x32_BOMEX_img350x350.pkl', 'rb') as f:
    #     pixel_centers_list = pickle.load(f)

    # if cfg.data.n_cam != 9:
    #     # print(len(images_mapping_list))
    #     images_mapping_list.pop(cfg.data.drop_index)
    #     # print(len(images_mapping_list))
    #     pixel_centers_list = np.delete(pixel_centers_list, cfg.data.drop_index, 0)



    masks = torch.tensor(masks,device=device)[None]
    val_volume = Volumes(torch.unsqueeze(torch.tensor(masks, device=device).float(), 1), grid)
    val_camera = AirMSPICameras(mapping=torch.tensor(images_mapping_list, device=device).float(),
                                  centers=torch.tensor(pixel_centers_list).float(), device=device)


# Activate eval mode of the model (lets us do a full rendering pass).
    with torch.no_grad():
        est_vols = torch.zeros(masks.shape, device=masks.device)
        # n_points_mask = torch.sum(torch.stack(masks)*1.0) if isinstance(masks, list) else masks.sum()
        # if n_points_mask > cfg.min_mask_points:
        net_start_time = time.time()

        val_out = model(
            val_camera,
            val_image,
            val_volume,
            masks
        )
        for i, (out_vol, m) in enumerate(zip(val_out["output"],masks)):
            if m is None:
                est_vols[i] = out_vol.squeeze(1)
            else:
                # m = m.reshape(-1)
                est_vols[i][m] = out_vol.squeeze(1)

        time_net = time.time() - net_start_time
        assert len(est_vols)==1 ##TODO support validation with batch larger than 1
        est_vols[est_vols<0] = 0

        airmspi_cloud = {'cloud':est_vols[0].cpu().numpy()}

        sio.savemat('airmspi_recovery.mat', airmspi_cloud)
        batch_time_net.append(time_net)

    batch_time_net = np.array(batch_time_net)

    print(f'Mean time = {np.mean(batch_time_net)}')


if __name__ == "__main__":
    main()


