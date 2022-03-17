"""
Implements image encoders from
https://github.com/sxyu/pixel-nerf/blob/a5a514224272a91e3ec590f215567032e1f1c260/src/model/encoder.py#L180
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from .util import nn_util as util
from .roi_align import ROIAlign
# from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler


class Backbone(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            backbone="resnet34",
            pretrained=True,
            num_layers=4,
            index_interp="bilinear",
            index_padding="border",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
            sampling_support=8,
            to_flatten = False
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )

            self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size,
                                         stride=self.model.conv1.stride, padding=self.model.conv1.padding,
                                         bias=self.model.conv1.bias!=None)
            # Following 2 lines need to be uncommented for older cfgigs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.sampling_support = sampling_support
            if backbone=='resnet34':
                self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
            elif backbone=='resnet50':
                self.latent_size = [0, 64, 320, 832, 1856][num_layers]
            else:
                NotImplementedError()
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.to_flatten = to_flatten
        self.samplers = [ROIAlign((sampling_support,sampling_support), 0.5**scale,0) for scale in range(1,1+self.num_layers)]
        self.net = nn.Linear(sampling_support*sampling_support,1)
            # nn.Sequential(nn.Conv2d(512,512,kernel_size=8,padding=0,bias=False),
            #                      nn.ReLU(inplace=True),
                                 # nn.MaxPool2d(2),
                                 # nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                 # nn.ReLU(inplace=True),
                                 # nn.MaxPool2d(2),
                                 # nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                 # nn.ReLU(inplace=True),
                                 # nn.MaxPool2d(2),
                                 # )

        # self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        # self.register_buffer(
        #     "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        # )
        # self.latent (B, L, H, W)

    # def sample_images_old(self, latent, uv, image_size=()):
    #     """
    #     Get pixel-aligned image features at 2D image coordinates
    #     :param latent (B, C, H, W) images features
    #     :param uv (B, N, 2) image points (x,y)
    #     :param image_size image size, either (width, height) or single int.
    #     if not specified, assumes coords are in [-1, 1]
    #     :return (B, C, N) L is latent size
    #     """
    #     with profiler.record_function("encoder_index"):
    #         if uv.shape[0] == 1 and latent.shape[0] > 1:
    #             uv = uv.expand(latent.shape[0], -1, -1)
    #
    #         with profiler.record_function("encoder_index_pre"):
    #             if len(image_size) > 0:
    #                 if len(image_size) == 1:
    #                     image_size = (image_size, image_size)
    #                 scale = self.latent_scaling / image_size
    #                 uv = uv * scale - 1.0
    #
    #         uv = uv.unsqueeze(2)  # (B, N, 1, 2)
    #         samples = F.grid_sample(
    #             latent,
    #             uv,
    #             align_corners=True,
    #             mode=self.index_interp,
    #             padding_mode=self.index_padding,
    #         )
    #         return samples[:, :, :, 0]  # (B, C, N)
    def sample_roi(self, latents, box_centers):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param latent (B, C, H, W) images features
        :param box_center (B, N, 2) box center in pixels (x,y)
        :return (B, C, N) L is latent size
        """
        # if uv.shape[0] == 1 and latents.shape[0] > 1:
        #     uv = uv.expand(latents.shape[0], -1, -1)
        # image_size = torch.tensor(image_size, device=uv.device)
        # m = (image_size -1) / 2
        # uv /= m
        # uv -= 1

        boxes = [self.make_boxes(box_center) for box_center in box_centers]
        samples = torch.empty(0,device=latents[0].device)
        for latent, sampler in zip(latents, self.samplers):
            latent = latent.view(-1,*latent.shape[2:])
            roi_features = sampler(latent,boxes)#.reshape(box_centers.shape[0],box_centers.shape[1],-1) #
            samples = torch.cat((samples, roi_features),dim=1)

        samples = torch.squeeze(self.net(samples.reshape(samples.shape[0],samples.shape[1],-1)),-1)
        chuncks =  [box.shape[0] for box in boxes]
        samples = torch.split(samples,chuncks)
        if self.to_flatten:
            samples = [sample.reshape(*box_center.shape[:-1], -1).transpose(0, 1).reshape(box_center.shape[1], -1) for
                       sample, box_center in zip(samples, box_centers)]
        else:
            samples = [sample.reshape(*box_center.shape[:-1], -1).transpose(0, 1) for
                       sample, box_center in zip(samples, box_centers)]
            # samples.splitreshape(*box_centers.shape[:-1],-1)
        # samples = self.net(samples.reshape(samples.shape[0],samples.shape[1],-1)).reshape(*box_centers.shape[:-1],-1)
        return samples #.transpose(2,3) # (B, Cams,points,features)

    def make_boxes(self, box_centers):
        d = (self.sampling_support-1)/2
        x1y1 = box_centers - d
        x2y2 = box_centers + d
        # boxes = list(torch.cat((x1y1,x2y2),dim=-1).view(-1,box_centers.shape[-2],4))
        boxes = torch.cat((x1y1,x2y2),dim=-1).view(-1,4)
        return boxes
    def sample_features(self, latents, uv):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param latent (B, C, H, W) images features
        :param uv (B, N, 2) image points (x,y)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :return (B, C, N) L is latent size
        """
        # if uv.shape[0] == 1 and latents.shape[0] > 1:
        #     uv = uv.expand(latents.shape[0], -1, -1)
        # image_size = torch.tensor(image_size, device=uv.device)
        # m = (image_size -1) / 2
        # uv /= m
        # uv -= 1

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = torch.empty(0,device=uv.device)
        for latent in latents:
            samples = torch.cat((samples, torch.squeeze(F.grid_sample(
                latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            ))),dim=1)
        return samples # (Cams,cum_channels, N)
    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        # if self.feature_scale != 1.0:
        #     x = F.interpolate(
        #         x,
        #         scale_factor=self.feature_scale,
        #         mode="bilinear" if self.feature_scale > 1.0 else "area",
        #         align_corners=True if self.feature_scale > 1.0 else None,
        #         recompute_scale_factor=True,
        #     )
        # x = x.to(device=self.latent.device)
        input_size = torch.tensor(x.shape[-2:])
        if self.use_custom_resnet:
            latents = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            # latents = latents
            # align_corners = None if self.index_interp == "nearest " else True
            # latent_sz = latents[0].shape[-2:]
            # for i in range(len(latents)):
            #     latents[i] = F.interpolate(
            #         latents[i],
            #         latent_sz,
            #         mode=self.upsample_interp,
            #         align_corners=align_corners,
            #     )
            # latent = torch.cat(latents, dim=1)
        # self.latent_scaling[0] = latent.shape[-1]
        # self.latent_scaling[1] = latent.shape[-2]
        # self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0 #self.latent_scaling / (self.latent_scaling - 1) * 2.0
        self.latent_scaling = [(torch.tensor(latent.shape[-2:]) / input_size).to(device=x.device) for latent in latents]
        # return latent
        return latents
    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            cfg.backbone.name,
            pretrained=cfg.backbone.pretrained,
            num_layers=cfg.backbone.num_layers,
            index_interp=cfg.backbone.index_interp,
            index_padding=cfg.backbone.index_padding,
            upsample_interp=cfg.backbone.upsample_interp,
            feature_scale=cfg.backbone.feature_scale,
            use_first_pool=cfg.backbone.use_first_pool,
            sampling_support=cfg.backbone.sampling_support,
            to_flatten = cfg.backbone.feature_flatten
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size,
                                     stride=self.model.conv1.stride, padding=self.model.conv1.padding,bias=self.model.bias)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            cfg.get_string("backbone"),
            pretrained=cfg.get_bool("pretrained", True),
            latent_size=cfg.get_int("latent_size", 128),
        )


class ConvEncoder(nn.Module):
    """
    Basic, extremely simple convolutional encoder
    """

    def __init__(
        self,
        dim_in=1,
        norm_layer=util.get_norm_layer("group"),
        padding_type="reflect",
        use_leaky_relu=True,
        use_skip_conn=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.norm_layer = norm_layer
        self.activation = nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        self.padding_type = padding_type
        self.use_skip_conn = use_skip_conn

        # TODO: make these cfgigurable
        first_layer_chnls = 64
        mid_layer_chnls = 128
        last_layer_chnls = 128
        n_down_layers = 3
        self.n_down_layers = n_down_layers

        self.conv_in = nn.Sequential(
            nn.Conv2d(dim_in, first_layer_chnls, kernel_size=7, stride=2, bias=False),
            norm_layer(first_layer_chnls),
            self.activation,
        )

        chnls = first_layer_chnls
        for i in range(0, n_down_layers):
            conv = nn.Sequential(
                nn.Conv2d(chnls, 2 * chnls, kernel_size=3, stride=2, bias=False),
                norm_layer(2 * chnls),
                self.activation,
            )
            setattr(self, "conv" + str(i), conv)

            deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    4 * chnls, chnls, kernel_size=3, stride=2, bias=False
                ),
                norm_layer(chnls),
                self.activation,
            )
            setattr(self, "deconv" + str(i), deconv)
            chnls *= 2

        self.conv_mid = nn.Sequential(
            nn.Conv2d(chnls, mid_layer_chnls, kernel_size=4, stride=4, bias=False),
            norm_layer(mid_layer_chnls),
            self.activation,
        )

        self.deconv_last = nn.ConvTranspose2d(
            first_layer_chnls, last_layer_chnls, kernel_size=3, stride=2, bias=True
        )

        self.dims = [last_layer_chnls]

    def forward(self, x):
        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_in)
        x = self.conv_in(x)

        inters = []
        for i in range(0, self.n_down_layers):
            conv_i = getattr(self, "conv" + str(i))
            x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=conv_i)
            x = conv_i(x)
            inters.append(x)

        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_mid)
        x = self.conv_mid(x)
        x = x.reshape(x.shape[0], -1, 1, 1).expand(-1, -1, *inters[-1].shape[-2:])

        for i in reversed(range(0, self.n_down_layers)):
            if self.use_skip_conn:
                x = torch.cat((x, inters[i]), dim=1)
            deconv_i = getattr(self, "deconv" + str(i))
            x = deconv_i(x)
            x = util.same_unpad_deconv2d(x, layer=deconv_i)
        x = self.deconv_last(x)
        x = util.same_unpad_deconv2d(x, layer=self.deconv_last)
        return x
