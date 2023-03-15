import pickle
import matplotlib.pyplot as plt
import glob, os
import scipy.io as sio
import numpy as np






if __name__ == "__main__":
    datasets = [
        "/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/test",
        "/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/test",
        "/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/test",
        "/wdata/roironen/Data/processed_HAWAII_2000CCN_32x64_50m/10cameras_20m",

    ]
    exts = []
    for dataset in datasets:
        data_paths = [f for f in glob.glob(os.path.join(dataset, "cloud*.pkl"))]
        ext = []
        for cloud_path in data_paths:
            with open(cloud_path, 'rb') as f:
                ext.append(pickle.load(f)['ext'])

        exts.append(ext)

    fig, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    legend = ["BOMEX500CCN", "CASS600CCN", "BOMEX50CCN", "HAWAII2000CCN"]
    for e, ax, l in zip(exts,axs,legend):
        clouds = np.array(e)
        clouds = clouds[clouds>0]
        ax.hist(clouds,100,density=True)
        ax.legend([l])
        ax.set(xlabel='Voxel extinction value [1/km]', ylabel='Probability')
    fig.tight_layout()
    plt.show()
    print()