import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_readout import ForegroundTopKDepthReadout
from .transformer import TransformerEncoder, TransformerEncoderLayer
import math


class BaselinePreservingDynamicUpsample(nn.Module):
    """Learn small sampling offsets while exactly preserving bilinear at zero.

    The offset predictor runs on the low-resolution feature and pixel-shuffles
    two coordinates per output point.  Offsets are expressed in input-feature
    pixels and applied to the same ``align_corners=True`` grid used by the
    original StereoDETR bilinear upsampling path.
    """

    def __init__(self, channels, scale_factor=2, max_offset=0.5):
        super().__init__()
        self.channels = int(channels)
        self.scale_factor = int(scale_factor)
        self.max_offset = float(max_offset)
        if self.channels <= 0:
            raise ValueError("dynamic upsampling channels must be positive")
        if self.scale_factor <= 1:
            raise ValueError("dynamic upsampling scale_factor must exceed one")
        if not math.isfinite(self.max_offset) or self.max_offset <= 0.0:
            raise ValueError("dynamic upsampling max_offset must be finite and positive")

        output_channels = 2 * self.scale_factor * self.scale_factor
        self.offset_predictor = nn.Conv2d(
            self.channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # The uploaded V09 checkpoint has no offset parameters.  Zero
        # initialization makes a newly constructed V11 module reproduce the
        # previous bilinear path before any training update.
        nn.init.zeros_(self.offset_predictor.weight)
        nn.init.zeros_(self.offset_predictor.bias)

    def predict_offsets(self, feature):
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                "dynamic upsampling expects [B, {}, H, W], got {}".format(
                    self.channels, tuple(feature.shape)
                )
            )
        raw_offsets = F.pixel_shuffle(
            self.offset_predictor(feature), self.scale_factor
        )
        return torch.tanh(raw_offsets) * self.max_offset

    @staticmethod
    def _normalized_base_grid(batch, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y = y.view(1, height, 1).expand(batch, height, width)
        grid_x = x.view(1, 1, width).expand(batch, height, width)
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(self, feature):
        batch, _, height, width = feature.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        offsets = self.predict_offsets(feature)

        grid = self._normalized_base_grid(
            batch,
            output_height,
            output_width,
            feature.device,
            feature.dtype,
        )
        # Convert offsets from input-feature pixels to the normalized
        # align-corners coordinate system.  Degenerate one-pixel dimensions
        # keep zero displacement on that axis.
        x_scale = 2.0 / float(width - 1) if width > 1 else 0.0
        y_scale = 2.0 / float(height - 1) if height > 1 else 0.0
        grid = grid + torch.stack(
            (offsets[:, 0] * x_scale, offsets[:, 1] * y_scale),
            dim=-1,
        )
        return F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


class MobileV2Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        # v2
        self.pwconv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        # v2
        feat = self.pwconv(x)
        feat = self.dwconv(feat)
        feat = self.pwliner(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat


class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost

class GhostModule(nn.Module):
    """
        Ghost Module from https://github.com/iamhankai/ghostnet.pytorch.

    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Sequential(),
            nn.Conv2d(inp, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class ResGhostModule(GhostModule):
    """Some Information about ResGhostModule"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, relu=True, stride=1):
        assert(ratio > 2)
        super(ResGhostModule, self).__init__(inp, oup-inp, kernel_size, ratio-1, dw_size, relu=relu, stride=stride)
        self.oup = oup
        if stride > 1:
            self.downsampling = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsampling = None

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        if not self.downsampling is None:
            x = self.downsampling(x)
        out = torch.cat([x, x1, x2], dim=1)
        return out[:,:self.oup,:,:]


class Aggregation(nn.Module):
    def __init__(self, in_channels, in_channels_s8, in_channels_s16, left_att, blocks, expanse_ratio, backbone_channels, cat_left=False):
        super(Aggregation, self).__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio[0]
        self.expanse_ratio_8 = expanse_ratio[1]
        self.expanse_ratio_16 = expanse_ratio[2]
        self.cat_left = cat_left

        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        conv0_8 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio_8)
                 for i in range(blocks[0])]
        self.conv0_8 = nn.Sequential(*conv0_8)

        conv0_16 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio_16)
                 for i in range(blocks[0])]
        self.conv0_16 = nn.Sequential(*conv0_16)

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_add = [MobileV2Residual(in_channels * 2 + in_channels_s8, 
                                      in_channels * 2 + in_channels_s8, 
                                      stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[1] - 1)]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = MobileV2Residual(in_channels * 2 + in_channels_s8, 
                                      in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_add = [MobileV2Residual(in_channels * 4 + in_channels_s16, 
                                      in_channels * 4 + in_channels_s16, 
                                      stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[2] - 1)]
        self.conv4 = nn.Sequential(*conv4_add)

        if self.left_att:
            self.att0 = AttentionModule(in_channels, backbone_channels[0])
            self.att2 = AttentionModule(in_channels * 2 + in_channels_s8, backbone_channels[1])
            self.att4 = AttentionModule(in_channels * 4 + in_channels_s16, backbone_channels[2])
        
        if not self.cat_left:   
            input_features = in_channels * 4 + in_channels_s16 
            self.depth_reason = nn.Sequential(
                ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
                MobileV2Residual(3 * input_features, 
                                512, stride=1, expanse_ratio=self.expanse_ratio)
            )
        else:
            input_features = in_channels * 4 + in_channels_s16 + 256
            self.depth_reason = nn.Sequential(
                ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
                MobileV2Residual(3 * input_features, 
                                512, stride=1, expanse_ratio=self.expanse_ratio)
            )
    def forward(self, x, x_8, x_16, features_left):

        x = self.conv0(x)
        if self.left_att:
            x = self.att0(x, features_left[0])
        conv1 = self.conv1(x) # b, 48, 36, 160
       
        conv1 = torch.concat([x_8, conv1], dim=1) # b, 48+24, 36, 160
        conv2 = self.conv2(conv1)
    
        if self.left_att:
            conv2 = self.att2(conv2, features_left[1])
        conv3 = self.conv3(conv2)
        
        conv3 = torch.concat([x_16, conv3], dim=1) # b, 48+24+12, 36, 160
      
        conv4 = self.conv4(conv3)
        if self.left_att:
            conv4 = self.att4(conv4, features_left[2])
        if not self.cat_left:   
            conv5 = self.depth_reason(conv4)
        else:
            conv5 = self.depth_reason(torch.cat([conv4, features_left[2]], dim=1))
        return conv5


def correlation_volume(left_feature, right_feature, max_disp):
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume

def correlation_volume_flip(left_feature, right_feature, max_disp):
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, :-i] * right_feature[:, :, :, i:]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume


def spatially_regularize_correlation(
    cost_volume,
    enabled=False,
    kernel_size=3,
    blend=1.0,
    passes=1,
):
    """Apply a parameter-free spatial low-pass filter per disparity slice.

    The disparity axis stays untouched: ``[B, D, H, W]`` is interpreted as
    a 2D tensor with D channels, so average pooling only aggregates local
    image-space neighbours.  ``count_include_pad=False`` avoids attenuating
    valid costs at image borders.  Disabling the option or setting blend to
    zero returns the original tensor object, which makes the V09-off path
    exactly identical rather than merely numerically close.
    """
    if not enabled or float(blend) == 0.0:
        return cost_volume
    kernel_size = int(kernel_size)
    passes = int(passes)
    blend = float(blend)
    if cost_volume.ndim != 4:
        raise ValueError("correlation volume must be [B,D,H,W]")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("correlation smoothing kernel must be odd and > 1")
    if passes <= 0:
        raise ValueError("correlation smoothing passes must be positive")
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("correlation smoothing blend must be in [0,1]")

    smoothed = cost_volume
    padding = kernel_size // 2
    for _ in range(passes):
        smoothed = F.avg_pool2d(
            smoothed,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
    return cost_volume + blend * (smoothed - cost_volume)


class BaselinePreservingGroupwiseCorrelation(nn.Module):
    """Compress group-wise correlations back to the baseline disparity volume.

    StereoDETR averages every feature channel before cost aggregation.  This
    module first keeps ``num_groups`` correlation proposals and then predicts
    a spatially varying convex combination from the left feature.  The final
    1x1 gate is initialized to zero, so softmax is exactly uniform and the
    initial output is mathematically the same channel average as the baseline.

    Only one disparity slice is constructed at a time.  Consequently the
    implementation never materializes a ``[B, G, D, H, W]`` tensor.
    """

    def __init__(self, channels, num_groups=16, gate_kernel_size=3):
        super().__init__()
        self.channels = int(channels)
        self.num_groups = int(num_groups)
        self.gate_kernel_size = int(gate_kernel_size)
        if self.channels <= 0 or self.num_groups <= 1:
            raise ValueError("groupwise correlation needs channels > 0 and groups > 1")
        if self.channels % self.num_groups != 0:
            raise ValueError("feature channels must be divisible by num_groups")
        if self.gate_kernel_size <= 0 or self.gate_kernel_size % 2 == 0:
            raise ValueError("gate_kernel_size must be a positive odd number")
        self.channels_per_group = self.channels // self.num_groups

        padding = self.gate_kernel_size // 2
        self.context = nn.Conv2d(
            self.num_groups,
            self.num_groups,
            kernel_size=self.gate_kernel_size,
            padding=padding,
            groups=self.num_groups,
            bias=False,
        )
        self.group_logits = nn.Conv2d(
            self.num_groups,
            self.num_groups,
            kernel_size=1,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Identity depth-wise context gives the zero-initialized 1x1 gate a
        # non-zero input and therefore a useful gradient on the first update.
        nn.init.zeros_(self.context.weight)
        center = self.gate_kernel_size // 2
        with torch.no_grad():
            self.context.weight[:, 0, center, center] = 1.0
        nn.init.zeros_(self.group_logits.weight)
        nn.init.zeros_(self.group_logits.bias)

    def gate_weights(self, left_feature):
        if left_feature.ndim != 4 or left_feature.shape[1] != self.channels:
            raise ValueError(
                "groupwise gate expects [B, {}, H, W], got {}".format(
                    self.channels, tuple(left_feature.shape)
                )
            )
        batch, _, height, width = left_feature.shape
        grouped = left_feature.reshape(
            batch,
            self.num_groups,
            self.channels_per_group,
            height,
            width,
        )
        # RMS energy is sign-stable and summarizes whether a feature group is
        # locally active without introducing an additional feature backbone.
        summary = grouped.square().mean(dim=2).clamp_min(1.0e-12).sqrt()
        context = F.gelu(self.context(summary))
        return F.softmax(self.group_logits(context), dim=1)

    def _compress(self, product, weights):
        batch, _, height, width = product.shape
        grouped = product.reshape(
            batch,
            self.num_groups,
            self.channels_per_group,
            height,
            width,
        ).mean(dim=2)
        return (grouped * weights).sum(dim=1)

    def forward(self, left_feature, right_feature, max_disp, flip=False):
        if left_feature.shape != right_feature.shape:
            raise ValueError("left and right features must have identical shapes")
        batch, channels, height, width = left_feature.shape
        if channels != self.channels:
            raise ValueError(
                "configured {} channels but received {}".format(
                    self.channels, channels
                )
            )
        max_disp = int(max_disp)
        if max_disp <= 0:
            raise ValueError("max_disp must be positive")

        weights = self.gate_weights(left_feature)
        # The original implementation reshaped and reduced the correlation
        # product into groups for every disparity slice.  The same convex
        # combination can be moved in front of the disparity loop:
        #
        #   sum_g w_g * mean_cg(left_g * right_g)
        # = sum_c (w_group(c) / channels_per_group) * left_c * right_c.
        #
        # Expanding the learned group weights to channels once keeps the
        # checkpoint and mathematical definition unchanged, while each
        # disparity slice now uses the same single channel reduction as the
        # StereoDETR baseline.
        channel_weights = weights.repeat_interleave(
            self.channels_per_group, dim=1
        )
        weighted_left = (
            left_feature * channel_weights / self.channels_per_group
        )
        cost_volume = left_feature.new_zeros(
            batch, max_disp, height, width
        )
        for disparity in range(max_disp):
            if disparity == 0:
                cost_volume[:, disparity] = (
                    weighted_left * right_feature
                ).sum(dim=1)
            elif flip:
                compressed = (
                    weighted_left[:, :, :, :-disparity]
                    * right_feature[:, :, :, disparity:]
                ).sum(dim=1)
                cost_volume[:, disparity, :, disparity:] = compressed
            else:
                compressed = (
                    weighted_left[:, :, :, disparity:]
                    * right_feature[:, :, :, :-disparity]
                ).sum(dim=1)
                cost_volume[:, disparity, :, disparity:] = compressed
        return cost_volume.contiguous()

class LightStereoDepthPredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"])
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"])
        self.depth_max = depth_max
        self.decoder_type = model_cfg['decoder_type']
        self.depth_map_size_mode = model_cfg['depth_map_size_mode']

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)
        depth_readout_cfg = model_cfg.get("depth_readout", {})
        self.depth_readout = ForegroundTopKDepthReadout(
            enabled=bool(depth_readout_cfg.get("enabled", False)),
            top_k=int(depth_readout_cfg.get("top_k", 2)),
        )
        dynamic_upsampling_cfg = model_cfg.get(
            "dynamic_depth_upsampling", {}
        )
        self.dynamic_depth_upsampling_enabled = bool(
            dynamic_upsampling_cfg.get("enabled", False)
        )
        self.dynamic_depth_upsampling_stages = tuple(
            sorted(
                set(
                    int(stage)
                    for stage in dynamic_upsampling_cfg.get("stages", [])
                )
            )
        )
        invalid_stages = set(self.dynamic_depth_upsampling_stages) - {1, 2}
        if invalid_stages:
            raise ValueError(
                "dynamic_depth_upsampling.stages only accepts 1 and 2"
            )
        if (
            self.dynamic_depth_upsampling_enabled
            and not self.dynamic_depth_upsampling_stages
        ):
            raise ValueError(
                "dynamic depth upsampling requires at least one stage"
            )
        self.dynamic_depth_max_offset = float(
            dynamic_upsampling_cfg.get("max_offset", 0.5)
        )
        if (
            self.dynamic_depth_upsampling_enabled
            and self.depth_map_size_mode != "downsample_x4"
        ):
            raise ValueError(
                "dynamic depth upsampling currently requires "
                "depth_map_size_mode=downsample_x4"
            )

        groupwise_cfg = model_cfg.get("groupwise_correlation", {})
        self.groupwise_correlation_enabled = bool(
            groupwise_cfg.get("enabled", False)
        )
        self.groupwise_correlation_scale = int(
            groupwise_cfg.get("scale", 4)
        )
        self.groupwise_correlation_groups = int(
            groupwise_cfg.get("num_groups", 16)
        )
        if (
            self.groupwise_correlation_enabled
            and self.groupwise_correlation_scale != 4
        ):
            raise ValueError(
                "V12 currently supports only the G0-validated s4 scale"
            )
        self.groupwise_correlation_s4 = None
        if self.groupwise_correlation_enabled:
            self.groupwise_correlation_s4 = (
                BaselinePreservingGroupwiseCorrelation(
                    channels=int(model_cfg["hidden_dim"]),
                    num_groups=self.groupwise_correlation_groups,
                    gate_kernel_size=int(
                        groupwise_cfg.get("gate_kernel_size", 3)
                    ),
                )
            )

        smoothing_cfg = model_cfg.get("correlation_smoothing", {})
        self.correlation_smoothing_enabled = bool(
            smoothing_cfg.get("enabled", False)
        )
        self.correlation_smoothing_scale = int(
            smoothing_cfg.get("scale", 4)
        )
        self.correlation_smoothing_kernel_size = int(
            smoothing_cfg.get("kernel_size", 3)
        )
        self.correlation_smoothing_blend = float(
            smoothing_cfg.get("blend", 1.0)
        )
        self.correlation_smoothing_passes = int(
            smoothing_cfg.get("passes", 1)
        )
        if self.correlation_smoothing_scale != 4:
            raise ValueError(
                "G4-authorized correlation smoothing supports only s4"
            )
        if (
            self.correlation_smoothing_kernel_size <= 1
            or self.correlation_smoothing_kernel_size % 2 == 0
        ):
            raise ValueError(
                "correlation smoothing kernel_size must be odd and > 1"
            )
        if self.correlation_smoothing_passes <= 0:
            raise ValueError("correlation smoothing passes must be positive")
        if (
            not math.isfinite(self.correlation_smoothing_blend)
            or not 0.0 <= self.correlation_smoothing_blend <= 1.0
        ):
            raise ValueError("correlation smoothing blend must be in [0,1]")

        # Create modules
        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())
        
        # self.dims_head = nn.Sequential(
        #     nn.Conv2d(512, d_model, kernel_size=(3, 3), padding=1),
        #     nn.GroupNorm(32, num_channels=d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
        #     nn.GroupNorm(32, num_channels=d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(d_model, 1, kernel_size=(1, 1)))
        
        self.output_channel_num = 256
        if self.depth_map_size_mode == "downsample_x16":
            self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))
        elif self.depth_map_size_mode == "downsample_x4":
            first_upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True
            )
            second_upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True
            )
            if (
                self.dynamic_depth_upsampling_enabled
                and 1 in self.dynamic_depth_upsampling_stages
            ):
                first_upsample = BaselinePreservingDynamicUpsample(
                    d_model,
                    scale_factor=2,
                    max_offset=self.dynamic_depth_max_offset,
                )
            if (
                self.dynamic_depth_upsampling_enabled
                and 2 in self.dynamic_depth_upsampling_stages
            ):
                second_upsample = BaselinePreservingDynamicUpsample(
                    int(self.output_channel_num / 2),
                    scale_factor=2,
                    max_offset=self.dynamic_depth_max_offset,
                )
            self.depth_classifier = nn.Sequential(
            first_upsample,
            nn.Conv2d(d_model, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            second_upsample,
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), depth_num_bins + 1, kernel_size=(1, 1)),
        )
        else:
            raise NotImplementedError
        
        
        self.disp_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 96, 1),
        )
        if self.decoder_type == "depthaware":
            depth_encoder_layer = TransformerEncoderLayer(
                d_model, nhead=8, dim_feedforward=256, dropout=0.1)
            self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

            self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)
        

        # aggregation
        self.cost_agg = Aggregation(in_channels=24,
                                    in_channels_s8=24, #24
                                    in_channels_s16=12,
                                    left_att=True,
                                    blocks=[ 1, 2, 4 ],
                                    expanse_ratio=[4, 4, 4],
                                    backbone_channels=[ 256, 256, 256 ],
                                    cat_left=model_cfg["cat_left"])
        
    def forward(self, feature_stereo, mask, pos, targets=None):
        debug = False
        if debug:
            print("debuging!!!!")
        assert len(feature_stereo) == 4
        batch_size_half = feature_stereo[0].shape[0] //2
        # foreground depth map
        #ipdb.set_trace()
        # src_16 = self.proj(feature_stereo[2])
        # src_16_left = src_16[:batch_size_half]
        # src_32 = self.upsample(F.interpolate(feature_stereo[2][:batch_size_half], size=src_16.shape[-2:], mode='bilinear'))
        # src_8 = self.downsample(feature_stereo[1][:batch_size_half])
        if self.training :
            gwc_volume_list4 = []
            gwc_volume_list8 = []
            gwc_volume_list16 = []
            for  batch_id in range(batch_size_half):
                flip_flag = targets[batch_id]["random_flip_flag"]
                switch_flag = targets[batch_id]["random_switch_flag"]
                f_left_i_s4 = feature_stereo[0][batch_id].unsqueeze(0)
                f_right_i_s4 = feature_stereo[0][batch_id+batch_size_half].unsqueeze(0)
                f_left_i_s8 = feature_stereo[1][batch_id].unsqueeze(0)
                f_right_i_s8 = feature_stereo[1][batch_id+batch_size_half].unsqueeze(0)
                f_left_i_s16 = feature_stereo[2][batch_id].unsqueeze(0)
                f_right_i_s16 = feature_stereo[2][batch_id+batch_size_half].unsqueeze(0)
                if flip_flag or switch_flag:
                    if self.groupwise_correlation_enabled:
                        gwc_volume_list4.append(
                            self.groupwise_correlation_s4(
                                f_left_i_s4,
                                f_right_i_s4,
                                96 // 4,
                                flip=True,
                            )
                        )
                    else:
                        gwc_volume_list4.append(correlation_volume_flip(f_left_i_s4, f_right_i_s4, 96 // 4)) 
                    gwc_volume_list8.append(correlation_volume_flip(f_left_i_s8, f_right_i_s8, 192 // 8))
                    gwc_volume_list16.append(correlation_volume_flip(f_left_i_s16, f_right_i_s16, 192 // 16))
                else:
                    if self.groupwise_correlation_enabled:
                        gwc_volume_list4.append(
                            self.groupwise_correlation_s4(
                                f_left_i_s4,
                                f_right_i_s4,
                                96 // 4,
                            )
                        )
                    else:
                        gwc_volume_list4.append(correlation_volume(f_left_i_s4, f_right_i_s4, 96 // 4))
                    gwc_volume_list8.append(correlation_volume(f_left_i_s8, f_right_i_s8, 192 // 8)) # 192/8
                    gwc_volume_list16.append(correlation_volume(f_left_i_s16, f_right_i_s16, 192 // 16))
            gwc_volume_s4 = torch.cat(gwc_volume_list4, 0)
            gwc_volume_s8 = torch.cat(gwc_volume_list8, 0)
            gwc_volume_s16 = torch.cat(gwc_volume_list16, 0)
        else:         
            if self.groupwise_correlation_enabled:
                gwc_volume_s4 = self.groupwise_correlation_s4(
                    feature_stereo[0][:batch_size_half],
                    feature_stereo[0][batch_size_half:],
                    96 // 4,
                )
            else:
                gwc_volume_s4 = correlation_volume(feature_stereo[0][:batch_size_half], 
                                                   feature_stereo[0][batch_size_half:], 
                                                   96 // 4)
            gwc_volume_s8 = correlation_volume(feature_stereo[1][:batch_size_half], 
                                               feature_stereo[1][batch_size_half:], 
                                               192 // 8)  # 192/8
            gwc_volume_s16 = correlation_volume(feature_stereo[2][:batch_size_half], 
                                               feature_stereo[2][batch_size_half:], 
                                               192 // 16)
        gwc_volume_s4 = spatially_regularize_correlation(
            gwc_volume_s4,
            enabled=self.correlation_smoothing_enabled,
            kernel_size=self.correlation_smoothing_kernel_size,
            blend=self.correlation_smoothing_blend,
            passes=self.correlation_smoothing_passes,
        )
        features_left = []
        for i in range(len(feature_stereo)):
            # features_left.append(feature_stereo[i][batch_size_half:])
            features_left.append(feature_stereo[i][:batch_size_half])
        PSV_features = self.cost_agg(gwc_volume_s4, gwc_volume_s8, gwc_volume_s16, features_left)
        features_for_depth = PSV_features
        src = self.depth_head(features_for_depth)
        feature_depth_dead = src
        #ipdb.set_trace()
        if self.training or debug:
            pred_disp = self.disp_output(src)
            # dims_pre = self.dims_head(features_for_depth)
        else:
            pred_disp = None
            # dims_pre = None
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = self.depth_readout(
            depth_logits,
            self.depth_bin_values,
            depth_probs=depth_probs,
        )
        #ipdb.set_trace()
        # depth embeddings with depth positional encodings
        if self.decoder_type == "depthaware":
            B, C, H, W = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
            pos = pos.flatten(2).permute(2, 0, 1)
            depth_embed = self.depth_encoder(src, mask, pos)
            depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
            #ipdb.set_trace()
            depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
            depth_embed = depth_embed + depth_pos_embed_ip
        else:
            depth_embed = None
            depth_pos_embed_ip = None

        return depth_logits, depth_embed, weighted_depth, depth_pos_embed_ip, PSV_features, pred_disp, feature_depth_dead

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
