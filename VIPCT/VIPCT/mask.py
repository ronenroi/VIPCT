"""
Implements image encoders from
https://github.com/sxyu/pixel-nerf/blob/a5a514224272a91e3ec590f215567032e1f1c260/src/model/encoder.py#L180
"""
import torch
from torch import nn



class MaskGenerator(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self

    ):
        """

        """
        super().__init__()


        self.up = []
        self.up.append(nn.ModuleList([
            nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(2, mode='bilinear')
            ]
        )
        )
        block = []
        for i in range(2):
            block+=[
                nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(2, mode='bilinear')
            ]
        self.up.append(nn.ModuleList(block))
        block = []
        for i in range(3):
            block+=[
                nn.Conv2d(int(128 / (2 ** (i))), int(128 / (2 ** (i + 1))), kernel_size=3, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(2, mode='bilinear')
            ]
        self.up.append(nn.ModuleList(block))
        for i in range(4):
            block+=[
                nn.Conv2d(int(256 / (2 ** (i))), int(256 / (2 ** (i + 1))), kernel_size=3, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(2, mode='bilinear')
            ]
        self.up.append(nn.ModuleList(block))



        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, features):
        """

        """
        pred_masks = torch.empty(0,device=features[0].device)
        for feature, up in zip(features, self.up):
            pred_masks += up(feature.view(-1,*feature.shape[2:]))
        return self.out(pred_masks)
    # @classmethod
    # def from_cfg(cls, cfg):
    #     return cls(
    #         cfg.backbone.name,
    #         pretrained=cfg.backbone.pretrained,
    #         num_layers=cfg.backbone.num_layers,
    #         index_interp=cfg.backbone.index_interp,
    #         index_padding=cfg.backbone.index_padding,
    #         upsample_interp=cfg.backbone.upsample_interp,
    #         feature_scale=cfg.backbone.feature_scale,
    #         use_first_pool=cfg.backbone.use_first_pool,
    #     )

