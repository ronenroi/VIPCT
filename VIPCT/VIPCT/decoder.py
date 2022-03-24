import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from .util import nn_util as util
from .roi_align import ROIAlign
# from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)

class Decoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            type,
            average_cams,
            feature_flatten,
            n_cam,
            latent_size,
    ):
        """
        :param type Decoder network type.
        """
        super().__init__()
        self.average_cams = average_cams
        self.feature_flatten = feature_flatten
        if type == 'FixCT':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, 1)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)

            self.decoder = torch.nn.Sequential(
                linear1,
                # torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(True),
                linear2,
                # torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'FixCTv2':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, 1)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)

            self.decoder = torch.nn.Sequential(
                linear1,
                torch.nn.LayerNorm(2048),
                torch.nn.ReLU(True),
                linear2,
                torch.nn.LayerNorm(512),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'FixCTv3':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, 1)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)
            nn.init.constant_(linear4.bias.data, 10)

            self.decoder = torch.nn.Sequential(
                linear1,
                torch.nn.LayerNorm(2048),
                torch.nn.ReLU(True),
                linear2,
                torch.nn.LayerNorm(512),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'VIPCT':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, 1)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)

            self.decoder = torch.nn.Sequential(
                linear1,
                # torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(True),
                linear2,
                # torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.ReLU(True),
                linear4
            )

    def forward(self, x):
        if self.average_cams:
            x = torch.mean(x,1)
        if self.feature_flatten:
            x = x.view(x.shape[0],-1)
        return self.decoder(x)

    @classmethod
    def from_cfg(cls, cfg, latent_size):
        return cls(
            type = cfg.decoder.name,
            average_cams=cfg.decoder.average_cams,
            feature_flatten=cfg.decoder.feature_flatten,
            n_cam = cfg.data.n_cam,
            latent_size = latent_size,
        )
