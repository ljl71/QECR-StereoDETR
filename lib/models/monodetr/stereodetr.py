"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import math
import copy
import cv2

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)
from utils.box_ops import  box_iou, box_cxcylrtb_to_xyxy

from .backbone import build_backbone
from .bifpn import BiFPN
from .matcherstable_fg_only import build_fg_Stablematcher
from .depthaware_transformer import build_depthaware_transformer_stereo
from .anchor_transformer import build_anchor_transformer_stereo
from .detr_transformer import build_detr_transformer_stereo
from .depth_predictor import LightStereoDepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from .depth_predictor import DisparityLoss, MultiCandidateDisparityLoss
from .query_depth import (
    BaselineSafeDepthFusion,
    QueryDepthDistributionRefiner,
    QueryDepthDistributionSampler,
    UncertaintyGeometryGate,
)
from .quality_ranking import (
    Query3DQualityHead,
    build_3d_iou_quality_targets,
    pairwise_quality_loss,
    pointwise_quality_loss,
    quality_probability_from_logits,
)
from .geometry_alignment import (
    corner_alignment_loss,
    cuboid_corners,
    decode_camera_geometry,
    decode_predicted_alpha,
    decode_target_alpha,
    encoded_boxes_to_original_xyxy,
    progressive_weight,
    project_corners,
    projected_enclosing_boxes,
    projection_alignment_loss,
)
from .geometry_depth_residual import QueryGeometryDepthResidual
from .axial_depth_iou import depth_only_axial_giou_loss
from lib.losses.focal_loss import sigmoid_focal_loss, SigmoidFocalLoss
from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def freeze_except_parameter_prefixes(module, prefixes):
    """Freeze a model except parameters whose names start with a whitelist."""
    prefixes = tuple(str(prefix) for prefix in prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("trainable parameter prefixes must be non-empty")
    trainable = []
    for name, parameter in module.named_parameters():
        enabled = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise ValueError(
            "no parameters matched trainable prefixes {}".format(prefixes)
        )
    return trainable


class StereoDETR(nn.Module):
    """ This is the StereoDETR module that performs Stereo 3D object detection """
    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, with_center_refine=False,two_stage=False, init_box=False, use_dab=False, 
                 group_num=11, two_stage_dino=False, depth_pre_type="one", depth_sample_mode="3Dcenter",
                 fusion_mode=None, num_scale=1, FPN_type=None, query_depth_cfg=None,
                 quality_ranking_cfg=None, geometry_alignment_cfg=None,
                 dynamic_depth_upsampling_cfg=None,
                 groupwise_correlation_cfg=None,
                 geometry_depth_residual_cfg=None,
                 axial_depth_iou_cfg=None):

        super().__init__()
 
        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.with_center_refine = with_center_refine
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.two_stage_dino = two_stage_dino
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
  
        self.depth_pre_type = depth_pre_type
        self.depth_sample_mode = depth_sample_mode
        self.query_depth_cfg = query_depth_cfg or {}
        self.query_depth_enabled = bool(self.query_depth_cfg.get("enabled", False))
        self.baseline_safe_fusion_enabled = bool(
            self.query_depth_cfg.get("baseline_safe_fusion", False)
        )
        if self.baseline_safe_fusion_enabled and not self.query_depth_enabled:
            raise ValueError(
                "baseline_safe_fusion requires query_depth.enabled"
            )
        self.geometry_gate_enabled = bool(
            self.query_depth_cfg.get("geometry_gate", False)
        )
        self.geometry_alignment_cfg = geometry_alignment_cfg or {}
        self.geometry_alignment_enabled = bool(
            self.geometry_alignment_cfg.get("enabled", False)
        )
        self.quality_ranking_cfg = quality_ranking_cfg or {}
        self.quality_ranking_enabled = bool(
            self.quality_ranking_cfg.get("enabled", False)
        )
        self.quality_freeze_detector = bool(
            self.quality_ranking_cfg.get("freeze_detector", False)
        )
        self.dynamic_depth_upsampling_cfg = (
            dynamic_depth_upsampling_cfg or {}
        )
        self.depth_upsampling_train_only = bool(
            self.dynamic_depth_upsampling_cfg.get(
                "train_only_depth_classifier", False
            )
        )
        self.depth_upsampling_trainable_prefixes = tuple(
            str(prefix)
            for prefix in self.dynamic_depth_upsampling_cfg.get(
                "trainable_prefixes",
                ["depth_predictor.depth_classifier."],
            )
        )
        self.groupwise_correlation_cfg = groupwise_correlation_cfg or {}
        self.groupwise_correlation_train_only = bool(
            self.groupwise_correlation_cfg.get(
                "train_only_cost_aggregation", False
            )
        )
        self.groupwise_correlation_trainable_prefixes = tuple(
            str(prefix)
            for prefix in self.groupwise_correlation_cfg.get(
                "trainable_prefixes",
                ["depth_predictor.cost_agg."],
            )
        )
        self.geometry_depth_residual_cfg = geometry_depth_residual_cfg or {}
        self.geometry_depth_residual_enabled = bool(
            self.geometry_depth_residual_cfg.get("enabled", False)
        )
        self.geometry_depth_residual_train_only = bool(
            self.geometry_depth_residual_cfg.get("train_only_head", False)
        )
        self.geometry_depth_residual_trainable_prefixes = tuple(
            str(prefix)
            for prefix in self.geometry_depth_residual_cfg.get(
                "trainable_prefixes", ["depth_residual_head."]
            )
        )
        if (
            self.geometry_depth_residual_train_only
            and not self.geometry_depth_residual_enabled
        ):
            raise ValueError(
                "geometry_depth_residual.train_only_head requires enabled=true"
            )
        if self.geometry_depth_residual_enabled:
            self.depth_residual_head = QueryGeometryDepthResidual(
                query_dim=hidden_dim,
                hidden_dim=int(
                    self.geometry_depth_residual_cfg.get("hidden_dim", 32)
                ),
                max_correction=float(
                    self.geometry_depth_residual_cfg.get(
                        "max_correction", 3.0
                    )
                ),
                use_geometry_prior=bool(
                    self.geometry_depth_residual_cfg.get(
                        "use_geometry_prior", True
                    )
                ),
                detach_inputs=bool(
                    self.geometry_depth_residual_cfg.get("detach_inputs", True)
                ),
            )
        self.axial_depth_iou_cfg = axial_depth_iou_cfg or {}
        self.axial_depth_train_only = bool(
            self.axial_depth_iou_cfg.get(
                "train_only_depth_classifier", False
            )
        )
        self.axial_depth_trainable_prefixes = tuple(
            str(prefix)
            for prefix in self.axial_depth_iou_cfg.get(
                "trainable_prefixes",
                ["depth_predictor.depth_classifier."],
            )
        )
        restricted_scopes = sum(
            bool(value)
            for value in (
                self.quality_freeze_detector,
                self.depth_upsampling_train_only,
                self.groupwise_correlation_train_only,
                self.geometry_depth_residual_train_only,
                self.axial_depth_train_only,
            )
        )
        if restricted_scopes > 1:
            raise ValueError(
                "quality, V11 depth-only, V12 cost-aggregation and geometry "
                "depth-residual and axial-depth training scopes are mutually "
                "exclusive"
            )
        self.quality_score_power = float(
            self.quality_ranking_cfg.get("score_power", 1.0)
        )
        if not math.isfinite(self.quality_score_power) or self.quality_score_power < 0.0:
            raise ValueError(
                "quality_ranking.score_power must be finite and non-negative"
            )
        if self.quality_ranking_enabled:
            self.quality_head = Query3DQualityHead(hidden_dim=hidden_dim)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        if self.depth_sample_mode in [ "offset", '3Dcenter', 'reference']:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 8, 3)
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        if self.with_center_refine:
            self.center_refine_offset = MLP(hidden_dim, hidden_dim, 2, 2)  # center refine for depth sample
        
        if self.depth_pre_type in [ "sample_cat" ]:
            self.depth_fuse_conv =  MLP(256+256, 256, 256, 2)
            
        self.use_dab = use_dab
        
        self.fusion_mode = fusion_mode
        self.num_scale = num_scale
        self.FPN_type  = FPN_type
        if self.FPN_type == "BiFPN":
            self.bifpn = BiFPN(size=[256, 256, 256], feature_size=256, num_layers=3)
        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            if self.with_center_refine:
                nn.init.constant_(self.center_refine_offset.layers[-1].weight.data, 0)
                nn.init.constant_(self.center_refine_offset.layers[-1].bias.data, 0)

        if not two_stage:
            if two_stage_dino:
                self.query_embed = None
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries * group_num, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries * group_num, 6)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        
        self.two_stage = two_stage
        self.num_classes = num_classes

        if self.two_stage_dino:        
            _class_embed = nn.Linear(hidden_dim, num_classes)
            if self.depth_sample_mode in [ "offset", '3Dcenter', 'reference']:
                _bbox_embed = MLP(hidden_dim, hidden_dim, 8, 3)
            else:
                _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)    
            # init the two embed layers
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)   
            self.depthaware_transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.depthaware_transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        if self.query_depth_enabled:
            total_depth_bins = depth_predictor.depth_bin_values.numel() - 1
            self.query_depth_sampler = QueryDepthDistributionSampler(
                hidden_dim=hidden_dim,
                num_decoder_layers=num_pred,
                num_points=int(self.query_depth_cfg.get("num_points", 1)),
                temperature=float(self.query_depth_cfg.get("temperature", 1.0)),
                point_radius=float(self.query_depth_cfg.get("point_radius", 0.25)),
                entropy_weight=float(self.query_depth_cfg.get("entropy_weight", 1.0)),
                background_weight=float(self.query_depth_cfg.get("background_weight", 1.0)),
                align_corners=bool(self.query_depth_cfg.get("align_corners", True)),
            )
            self.query_depth_refiner = QueryDepthDistributionRefiner(
                hidden_dim=hidden_dim,
                num_decoder_layers=num_pred,
                total_bins=total_depth_bins,
                local_bins=int(
                    self.query_depth_cfg.get("local_bins", total_depth_bins)
                ),
                enabled=bool(
                    self.query_depth_cfg.get("cross_layer_refine", False)
                ),
                prior_weight=float(self.query_depth_cfg.get("prior_weight", 0.5)),
                detach_window=bool(
                    self.query_depth_cfg.get("detach_window", True)
                ),
                posterior_floor=float(
                    self.query_depth_cfg.get("posterior_floor", 0.05)
                ),
            )
            if self.baseline_safe_fusion_enabled:
                self.baseline_safe_depth_fusion = BaselineSafeDepthFusion(
                    num_decoder_layers=num_pred,
                    initial_query_weight=float(
                        self.query_depth_cfg.get(
                            "initial_query_weight", 0.05
                        )
                    ),
                    max_query_weight=float(
                        self.query_depth_cfg.get("max_query_weight", 0.5)
                    ),
                )
            if self.geometry_gate_enabled:
                self.uncertainty_geometry_gate = UncertaintyGeometryGate(
                    hidden_dim=hidden_dim,
                    hidden_gate_dim=int(
                        self.query_depth_cfg.get("gate_hidden_dim", 32)
                    ),
                    initial_stereo_weight=float(
                        self.query_depth_cfg.get("initial_stereo_weight", 0.9)
                    ),
                    max_correction=float(
                        self.query_depth_cfg.get("max_geometry_correction", 10.0)
                    ),
                )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d  
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
            if self.depth_pre_type in [ "sample_cat" ]:
                self.depth_fuse_conv =  _get_clones(self.depth_fuse_conv, num_pred)
            
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            if self.depth_pre_type in [ "sample_cat" ]:
                self.depth_fuse_conv =  nn.ModuleList([self.depth_fuse_conv for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if self.with_center_refine:
            if with_box_refine:
                self.center_refine_offset = _get_clones(self.center_refine_offset, num_pred)
                nn.init.constant_(self.center_refine_offset[0].layers[-1].bias.data[2:], -2.0)
            else:
                nn.init.constant_(self.center_refine_offset.layers[-1].bias.data[2:], -2.0)
                self.center_refine_offset = nn.ModuleList([self.center_refine_offset for _ in range(num_pred)])

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        # 二分之一平均池化
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.fusion_mode == "concat":
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),)
            self.fuse_conv1 = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),)
            self.fuse_conv2 = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),)
        elif self.fusion_mode == "replace":
              self.fuse_conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True))
            
        self.dnn_loss_temp = DDNLoss(depth_sort_reverse=False, 
                                     bg_value=60, 
                                     shrink_ratio=1,
                                     align_by_3d_center=True)

        if self.quality_ranking_enabled and self.quality_freeze_detector:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("quality_head."))
        elif self.depth_upsampling_train_only:
            self.depth_upsampling_trainable_parameters = (
                freeze_except_parameter_prefixes(
                    self, self.depth_upsampling_trainable_prefixes
                )
            )
        elif self.groupwise_correlation_train_only:
            self.groupwise_correlation_trainable_parameters = (
                freeze_except_parameter_prefixes(
                    self, self.groupwise_correlation_trainable_prefixes
                )
            )
        elif self.geometry_depth_residual_train_only:
            self.geometry_depth_residual_trainable_parameters = (
                freeze_except_parameter_prefixes(
                    self, self.geometry_depth_residual_trainable_prefixes
                )
            )
        elif self.axial_depth_train_only:
            self.axial_depth_trainable_parameters = (
                freeze_except_parameter_prefixes(
                    self, self.axial_depth_trainable_prefixes
                )
            )

    def train(self, mode=True):
        super().train(mode)
        if self.quality_ranking_enabled and self.quality_freeze_detector:
            # Transformer ``training`` state must stay enabled because Group
            # DETR reshapes the 11 query groups only in training mode.  Freeze
            # just the detector's BatchNorm statistics; its parameters already
            # have requires_grad=False.
            for module_name, module in self.named_modules():
                if (
                    not module_name.startswith("quality_head")
                    and isinstance(module, nn.modules.batchnorm._BatchNorm)
                ):
                    module.eval()
            self.quality_head.train(mode)
        elif self.depth_upsampling_train_only:
            # Keep all frozen BatchNorm statistics fixed.  The original
            # depth-classifier BatchNorm layers remain trainable in both V11O
            # and V11A/B, which makes the control comparison symmetric.
            for module_name, module in self.named_modules():
                if (
                    isinstance(module, nn.modules.batchnorm._BatchNorm)
                    and not any(
                        module_name.startswith(prefix.rstrip("."))
                        for prefix in self.depth_upsampling_trainable_prefixes
                    )
                ):
                    module.eval()
        elif self.groupwise_correlation_train_only:
            # V12O and V12A update the same cost aggregation branch.  Frozen
            # BatchNorm statistics stay fixed; only BatchNorm modules inside
            # the explicitly trainable aggregation prefixes may update.
            for module_name, module in self.named_modules():
                if (
                    isinstance(module, nn.modules.batchnorm._BatchNorm)
                    and not any(
                        module_name.startswith(prefix.rstrip("."))
                        for prefix in self.groupwise_correlation_trainable_prefixes
                    )
                ):
                    module.eval()
        elif self.geometry_depth_residual_train_only:
            # V20 freezes every existing V09 tensor.  There is no BatchNorm in
            # the residual head, so all detector BatchNorm statistics must also
            # remain fixed during the five-epoch paired screen.
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            self.depth_residual_head.train(mode)
        elif self.axial_depth_train_only:
            # V22O and V22A train exactly the same depth-classifier tensors.
            # All other BatchNorm statistics, including the frozen V09 quality
            # head path, remain fixed during the paired five-epoch screen.
            for module_name, module in self.named_modules():
                if (
                    isinstance(module, nn.modules.batchnorm._BatchNorm)
                    and not any(
                        module_name.startswith(prefix.rstrip("."))
                        for prefix in self.axial_depth_trainable_prefixes
                    )
                ):
                    module.eval()
        return self

    def forward(self, images, calibs, targets, img_sizes, img_sizes_ori, img_sizes_upper, dn_args=None):
        """?The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        batch_size = images.shape[0]
        left_images = images[:,0:3,:,:]
        right_images = images[:,3:,:,:]
        images = torch.cat([left_images, right_images], dim=0)
        features, pos_stereo = self.backbone(images)

        srcs_stereo = []
        masks_stereo = []

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_left_right = self.input_proj[l](src)
            
            srcs_stereo.append(src_left_right)
            masks_stereo.append(mask)
            
            assert mask is not None

        if self.num_feature_levels > len(srcs_stereo):
            _len_srcs = len(srcs_stereo)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs_stereo[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs_stereo.append(src)
                masks_stereo.append(mask)
                pos_stereo.append(pos_l)
        srcs_left = []
        masks_left = []
        srcs_right = []
        masks_right = []
        pos_left = []
        for src_stereo, mask_stereo, pos_stereo_i in zip(srcs_stereo, masks_stereo, pos_stereo):
        
            src_left = src_stereo[:batch_size]
            mask_left = mask_stereo[:batch_size]
            pos_left_i = pos_stereo_i[:batch_size]
            srcs_left.append(src_left)
            masks_left.append(mask_left)
            pos_left.append(pos_left_i)

            src_right = src_stereo[batch_size:]
            mask_right = mask_stereo[batch_size:]
            srcs_right.append(src_right)
            masks_right.append(mask_right)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.training:
                tgt_all_embed=tgt_embed = self.tgt_embed.weight           # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1) 
                
            else:
                tgt_all_embed=tgt_embed = self.tgt_embed.weight[:self.num_queries]         
                refanchor = self.refpoint_embed.weight[:self.num_queries]  
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1) 
        elif self.two_stage_dino:
            query_embeds = None
        else:
            if self.training:
                query_embeds = self.query_embed.weight
            else:
                # only use one group in inference
                query_embeds = self.query_embed.weight[:self.num_queries]
        
        if self.FPN_type == "BiFPN":
            srcs_left_bifpn = self.bifpn(srcs_left[1:])
            srcs_left[1] = srcs_left_bifpn[0]
            srcs_left[2] = srcs_left_bifpn[1]
            srcs_left[3] = srcs_left_bifpn[2]

        pred_depth_map_logits, depth_pos_embed, weighted_depth,\
              depth_pos_embed_ip, features_for_depth, pred_disp, feature_depth_dead = self.depth_predictor(
            srcs_stereo, masks_left[2], pos_left[2], targets)

        if self.fusion_mode == "concat":
            srcs_left[2] = self.fuse_conv1(torch.concat([srcs_left[2] , features_for_depth], dim=1)) 
        elif self.fusion_mode == "replace":
            srcs_left[2] = self.fuse_conv(features_for_depth)
        elif self.fusion_mode is None:
            pass
        else:
            raise NotImplementedError

        if self.num_scale == 3:
            hs, init_reference, inter_references, inter_references_dim, \
                enc_outputs_class, enc_outputs_coord_unact = self.depthaware_transformer(
                    srcs_left[1:4], masks_left[1:4], pos_left[1:4], query_embeds, depth_pos_embed, depth_pos_embed_ip)
        else:
            hs, init_reference, inter_references, inter_references_dim, \
                enc_outputs_class, enc_outputs_coord_unact = self.depthaware_transformer(
                    srcs_left[2:3], masks_left[2:3], pos_left[2:3], query_embeds, depth_pos_embed, depth_pos_embed_ip)#, attn_mask)

        outputs_coords = []
        outputs_samples = []
        outputs_classes = []
        outputs_classes_fg = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []
        outputs_depth_dist_logits = []
        outputs_depth_dist_probs = []
        outputs_depth_dist_bins = []
        outputs_depth_entropies = []
        outputs_depth_background_probs = []
        outputs_query_points = []
        outputs_query_point_weights = []
        outputs_query_depth_means = []
        outputs_point_depth_variances = []
        previous_query_full_probabilities = None
        final_gate_output = None
        final_safe_fusion_output = None

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            elif reference.shape[-1] == 8:
                tmp[..., :6] += reference[..., :6]
                tmp[..., 6:] += reference[..., :2]
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord[..., 0:6])

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)
            
            if self.depth_sample_mode == "offset": #从refer 预测到采样点的偏移量
                sample_offset = outputs_coord[..., 6:8]
                sample_ponit = sample_offset
            elif self.depth_sample_mode == "reference": #refer点
                sample_ponit = reference[..., :2]
                sample_ponit = sample_ponit.sigmoid()
            elif self.depth_sample_mode == "3Dcenter": #从3D center采样
                sample_ponit = outputs_coord[..., :2].detach()
            else:
                raise  ValueError("Error: depth_sample_mode data type is not supported")
            outputs_samples.append(sample_ponit)
            baseline_depth_map = None
            if self.baseline_safe_fusion_enabled:
                baseline_samples_grid = (
                    (sample_ponit[..., :2] - 0.5) * 2
                ).unsqueeze(2)
                baseline_depth_map = F.grid_sample(
                    weighted_depth.unsqueeze(1),
                    baseline_samples_grid,
                    mode='bilinear',
                    align_corners=True,
                ).squeeze(1)
            if self.query_depth_enabled:
                box_extents = torch.stack(
                    [
                        outputs_coord[..., 2] + outputs_coord[..., 3],
                        outputs_coord[..., 4] + outputs_coord[..., 5],
                    ],
                    dim=-1,
                )
                sampled_query_depth = self.query_depth_sampler(
                    depth_logits=pred_depth_map_logits,
                    depth_bin_values=self.depth_predictor.depth_bin_values,
                    query_features=hs[lvl],
                    center_points=sample_ponit,
                    box_extents=box_extents,
                    layer_index=lvl,
                )
                refined_query_depth = self.query_depth_refiner(
                    observed_probabilities=sampled_query_depth["probabilities"],
                    depth_bin_values=self.depth_predictor.depth_bin_values,
                    query_features=hs[lvl],
                    layer_index=lvl,
                    previous_full_probabilities=previous_query_full_probabilities,
                )
                previous_query_full_probabilities = refined_query_depth[
                    "full_probabilities"
                ]
                raw_query_depth = refined_query_depth["expected_depth"]
                if self.baseline_safe_fusion_enabled:
                    final_safe_fusion_output = (
                        self.baseline_safe_depth_fusion(
                            baseline_depth=baseline_depth_map,
                            query_depth=raw_query_depth,
                            layer_index=lvl,
                        )
                    )
                    depth_map = final_safe_fusion_output["depth"]
                else:
                    depth_map = raw_query_depth
                outputs_depth_dist_logits.append(refined_query_depth["logits"])
                outputs_depth_dist_probs.append(refined_query_depth["probabilities"])
                outputs_depth_dist_bins.append(refined_query_depth["bins"])
                outputs_depth_entropies.append(refined_query_depth["entropy"])
                outputs_depth_background_probs.append(
                    sampled_query_depth["background_probability"]
                )
                outputs_query_points.append(sampled_query_depth["points"])
                outputs_query_point_weights.append(sampled_query_depth["point_weights"])
                outputs_query_depth_means.append(depth_map)
                outputs_point_depth_variances.append(
                    sampled_query_depth["point_depth_variance"]
                )
            else:
                outputs_samples_grid = ((sample_ponit[..., :2] - 0.5) * 2).unsqueeze(2)
                depth_map = F.grid_sample(
                    weighted_depth.unsqueeze(1),
                    outputs_samples_grid,
                    mode='bilinear',
                    align_corners=True).squeeze(1)
            
            # depth_reg
            if self.depth_pre_type in ["sample_cat"] :
                feature_sample_pos = ((sample_ponit[..., :2] - 0.5) * 2).unsqueeze(2).detach()
                feature_for_depth_sampled = F.grid_sample(
                feature_depth_dead,
                feature_sample_pos,
                mode='bilinear',
                align_corners=True).squeeze(-1).permute(0, 2, 1)

                feature_2d_left = F.grid_sample(
                srcs_left[2],
                feature_sample_pos,
                mode='bilinear',
                align_corners=True).squeeze(-1).permute(0, 2, 1)

                feature_for_depth_reg = torch.cat([feature_for_depth_sampled, feature_2d_left], dim=2)
                feature_for_depth_reg = self.depth_fuse_conv[lvl](feature_for_depth_reg)

            elif self.depth_pre_type in ["one_w_uncertainty", "one_wo_uncertainty", "detrhead_w_uncertainty", "detrhead_wo_uncertainty"]:
                feature_for_depth_reg = hs[lvl]

            else:
                raise NotImplementedError
            
            depth_reg = self.depth_embed[lvl](feature_for_depth_reg)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)
            scales =  img_sizes_ori[:, 1: 2] / (img_sizes[:, 1: 2] + img_sizes_upper.unsqueeze(1))
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2]*scales, min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1) +(size3d[:, :, 2])/2
            
            depth_for_output = depth_map
            if (
                self.query_depth_enabled
                and self.geometry_gate_enabled
                and lvl == hs.shape[0] - 1
            ):
                final_gate_output = self.uncertainty_geometry_gate(
                    query_features=hs[lvl],
                    stereo_depth=depth_map,
                    geometry_depth=depth_geo.unsqueeze(-1),
                    entropy=outputs_depth_entropies[-1],
                    background_probability=outputs_depth_background_probs[-1],
                    log_variance=depth_reg[:, :, 1:2],
                    point_depth_variance=outputs_point_depth_variances[-1],
                )
                depth_for_output = final_gate_output["depth"]

            final_depth_residual_output = None
            if self.geometry_depth_residual_enabled:
                final_depth_residual_output = self.depth_residual_head(
                    query_features=hs[lvl],
                    stereo_depth=depth_for_output,
                    geometry_depth=depth_geo.unsqueeze(-1),
                )
                depth_for_output = final_depth_residual_output["depth"]

            # depth average + sigma
            if self.query_depth_enabled:
                depth_ave = torch.cat([depth_for_output, depth_reg[:, :, 1:2]], -1)
            elif self.depth_pre_type == "avg":
                depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) \
                                        + depth_geo.unsqueeze(-1) \
                                            + depth_map) / 3,
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "avg2":
                depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) \
                                        + depth_map) / 2,
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "one_wo_uncertainty":
                depth_ave = torch.cat([depth_for_output,
                                    depth_reg[:, :, 1: 2]*0], -1)
            elif self.depth_pre_type == "detrhead_w_uncertainty":
                depth_ave = torch.cat([(1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.),
                                       depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "detrhead_wo_uncertainty":
                depth_ave = torch.cat([(1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.),
                                       depth_reg[:, :, 1: 2]*0], -1)
            elif self.depth_pre_type in[ "one_w_uncertainty"]:
                depth_ave = torch.cat([depth_for_output,
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type in ["sample_cat" ]:
                depth_ave = torch.cat([(1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.),
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "depthmap_and_geo_w_uncertainty":
                # comepare depth_map and depth_geo to get the max depth
                max_map_geo = (depth_map + depth_geo.unsqueeze(-1))/2
                depth_ave = torch.cat([depth_map_ref,
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "err_w_uncertainty":
                depth_ave = torch.cat([depth_map + depth_reg[:, :, 0: 1],
                                    depth_reg[:, :, 1: 2]], -1)
            else:
                raise NotImplementedError
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)
        outputs_coord = torch.stack(outputs_coords)
        outputs_samples = torch.stack(outputs_samples)
        outputs_class = torch.stack(outputs_classes)
        
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)
        if self.query_depth_enabled:
            outputs_depth_dist_logits = torch.stack(outputs_depth_dist_logits)
            outputs_depth_dist_probs = torch.stack(outputs_depth_dist_probs)
            outputs_depth_dist_bins = torch.stack(outputs_depth_dist_bins)
            outputs_depth_entropies = torch.stack(outputs_depth_entropies)
            outputs_depth_background_probs = torch.stack(
                outputs_depth_background_probs
            )
            outputs_query_points = torch.stack(outputs_query_points)
            outputs_query_point_weights = torch.stack(outputs_query_point_weights)
            outputs_query_depth_means = torch.stack(outputs_query_depth_means)
            outputs_point_depth_variances = torch.stack(
                outputs_point_depth_variances
            )

        out = {'pred_logits': outputs_class[-1],
                'pred_boxes': outputs_coord[-1],
                'pred_sample_points': outputs_samples[-1],
                }
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        out['pred_disp'] = pred_disp
        if self.quality_ranking_enabled:
            quality_features = hs[-1]
            if self.quality_freeze_detector:
                quality_features = quality_features.detach()
            quality_logits = self.quality_head(quality_features)
            out['pred_quality_logits'] = quality_logits
            out['pred_quality'] = quality_probability_from_logits(
                quality_logits,
                score_power=self.quality_score_power,
            )
            # These tensors are consumed only by the training-time IoU target
            # builder.  Inference merely reads pred_quality.
            out['quality_calibs'] = calibs
            out['quality_img_sizes'] = img_sizes
            out['quality_img_sizes_ori'] = img_sizes_ori
            out['quality_img_sizes_upper'] = img_sizes_upper
        if self.geometry_alignment_enabled:
            out["geometry_calibs"] = calibs
            out["geometry_img_sizes"] = img_sizes
            out["geometry_img_sizes_ori"] = img_sizes_ori
            out["geometry_img_sizes_upper"] = img_sizes_upper
        if self.query_depth_enabled:
            out.update(
                {
                    "pred_depth_dist_logits": outputs_depth_dist_logits[-1],
                    "pred_depth_dist_probs": outputs_depth_dist_probs[-1],
                    "pred_depth_dist_bins": outputs_depth_dist_bins[-1],
                    "pred_depth_entropy": outputs_depth_entropies[-1],
                    "pred_depth_background_probability": outputs_depth_background_probs[-1],
                    "pred_query_points": outputs_query_points[-1],
                    "pred_query_point_weights": outputs_query_point_weights[-1],
                    "pred_query_depth_mean": outputs_query_depth_means[-1],
                    "pred_point_depth_variance": outputs_point_depth_variances[-1],
                }
            )
            if final_safe_fusion_output is not None:
                out.update(
                    {
                        "pred_baseline_depth": final_safe_fusion_output[
                            "baseline_depth"
                        ],
                        "pred_raw_query_depth": final_safe_fusion_output[
                            "query_depth"
                        ],
                        "pred_query_depth_blend_weight": (
                            final_safe_fusion_output["query_weight"]
                        ),
                    }
                )
            if final_gate_output is not None:
                out["pred_depth_gate"] = final_gate_output["stereo_weight"]
                out["pred_stereo_depth"] = final_gate_output["stereo_depth"]
                out["pred_geometry_depth"] = final_gate_output["geometry_depth"]
        if self.geometry_depth_residual_enabled:
            out["pred_depth_residual"] = final_depth_residual_output[
                "residual"
            ]
            out["pred_depth_geometry_context"] = final_depth_residual_output[
                "geometry_context"
            ]
            out["pred_depth_before_residual"] = final_depth_residual_output[
                "stereo_depth"
            ]
            out["pred_geometry_depth"] = final_depth_residual_output[
                "geometry_depth"
            ]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
                outputs_3d_dim,
                outputs_angle,
                outputs_depth,
                outputs_samples,
                outputs_depth_dist_logits if self.query_depth_enabled else None,
                outputs_depth_dist_probs if self.query_depth_enabled else None,
                outputs_depth_dist_bins if self.query_depth_enabled else None,
                outputs_depth_entropies if self.query_depth_enabled else None,
                outputs_depth_background_probs if self.query_depth_enabled else None,
                outputs_query_points if self.query_depth_enabled else None,
                outputs_query_point_weights if self.query_depth_enabled else None,
                outputs_query_depth_means if self.query_depth_enabled else None,
                outputs_point_depth_variances if self.query_depth_enabled else None,
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out #, mask_dict

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class,
        outputs_coord,
        outputs_3d_dim,
        outputs_angle,
        outputs_depth,
        outputs_samples,
        outputs_depth_dist_logits=None,
        outputs_depth_dist_probs=None,
        outputs_depth_dist_bins=None,
        outputs_depth_entropies=None,
        outputs_depth_background_probs=None,
        outputs_query_points=None,
        outputs_query_point_weights=None,
        outputs_query_depth_means=None,
        outputs_point_depth_variances=None,
    ):
        auxiliary_outputs = [
            {
                'pred_logits': a,
                'pred_boxes': b,
                'pred_3d_dim': c,
                'pred_angle': d,
                'pred_depth': e,
                'pred_sample_points': f,
            }
            for a, b, c, d, e, f in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_3d_dim[:-1],
                outputs_angle[:-1],
                outputs_depth[:-1],
                outputs_samples[:-1],
            )
        ]
        if outputs_depth_dist_logits is not None:
            for layer_index, layer_output in enumerate(auxiliary_outputs):
                layer_output.update(
                    {
                        "pred_depth_dist_logits": outputs_depth_dist_logits[layer_index],
                        "pred_depth_dist_probs": outputs_depth_dist_probs[layer_index],
                        "pred_depth_dist_bins": outputs_depth_dist_bins[layer_index],
                        "pred_depth_entropy": outputs_depth_entropies[layer_index],
                        "pred_depth_background_probability": outputs_depth_background_probs[layer_index],
                        "pred_query_points": outputs_query_points[layer_index],
                        "pred_query_point_weights": outputs_query_point_weights[layer_index],
                        "pred_query_depth_mean": outputs_query_depth_means[layer_index],
                        "pred_point_depth_variance": outputs_point_depth_variances[layer_index],
                    }
                )
        return auxiliary_outputs


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, 
                 group_num=11, depth_sort_reverse=False, depth_bg=0,
                 shrink_ratio=1, align_by_3d_center=False, query_depth_cfg=None,
                 disparity_supervision_cfg=None, quality_ranking_cfg=None,
                 geometry_alignment_cfg=None, axial_depth_iou_cfg=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss(depth_sort_reverse=depth_sort_reverse, 
                                bg_value=depth_bg, 
                                shrink_ratio=shrink_ratio,
                                align_by_3d_center=align_by_3d_center,
                                )  # for depth map
        self.disparity_loss = DisparityLoss(maxdisp=96)
        self.disparity_supervision_cfg = disparity_supervision_cfg or {}
        self.disparity_supervision_mode = self.disparity_supervision_cfg.get(
            'mode', 'scalar'
        )
        if self.disparity_supervision_mode not in [
            'scalar', 'multi_candidate'
        ]:
            raise ValueError(
                "model.disparity_supervision.mode must be 'scalar' or "
                "'multi_candidate'"
            )
        self.multi_candidate_disparity_loss = MultiCandidateDisparityLoss(
            maxdisp=96,
            variance=float(
                self.disparity_supervision_cfg.get('variance', 0.5)
            ),
            smoothing_radius=int(
                self.disparity_supervision_cfg.get('smoothing_radius', 4)
            ),
            use_soft_confidence=bool(
                self.disparity_supervision_cfg.get(
                    'lrc_soft_weight', False
                )
            ),
        )
        self.loss_cls_2d = SigmoidFocalLoss(gamma=2.0, balance_weights=torch.tensor([4.0, 2.0, 4.0]))
        self.group_num = group_num
        self.depth_match_type =  "filter" # "filter" "rematch" 
        self.query_depth_cfg = query_depth_cfg or {}
        self.quality_aligned_cls = bool(
            self.query_depth_cfg.get("quality_aligned_cls", False)
        )
        self.quality_iou_alpha = float(
            self.query_depth_cfg.get("quality_iou_alpha", 0.5)
        )
        self.quality_depth_tau = float(
            self.query_depth_cfg.get("quality_depth_tau", 0.1)
        )
        self.query_dist_target_temperature = float(
            self.query_depth_cfg.get("target_temperature", 1.0)
        )
        self.gate_oracle_temperature = float(
            self.query_depth_cfg.get("gate_oracle_temperature", 1.0)
        )
        self.teacher_confidence_threshold = float(
            self.query_depth_cfg.get("teacher_confidence_threshold", 0.5)
        )
        self.teacher_target_temperature = float(
            self.query_depth_cfg.get("teacher_target_temperature", 1.0)
        )
        self.query_align_corners = bool(
            self.query_depth_cfg.get("align_corners", True)
        )
        self.geometry_alignment_cfg = geometry_alignment_cfg or {}
        self.geometry_alignment_enabled = bool(
            self.geometry_alignment_cfg.get("enabled", False)
        )
        self.geometry_corner_enabled = bool(
            self.geometry_alignment_cfg.get("corner_enabled", False)
        )
        self.geometry_projection_enabled = bool(
            self.geometry_alignment_cfg.get("projection_enabled", False)
        )
        if self.geometry_alignment_enabled and not (
            self.geometry_corner_enabled or self.geometry_projection_enabled
        ):
            raise ValueError(
                "geometry_alignment.enabled requires corner_enabled or projection_enabled"
            )
        self.geometry_epoch = 0
        self.quality_ranking_cfg = quality_ranking_cfg or {}
        self.quality_ranking_enabled = bool(
            self.quality_ranking_cfg.get("enabled", False)
        )
        self.quality_pairwise_enabled = bool(
            self.quality_ranking_cfg.get("pairwise_enabled", False)
        )
        self.quality_only = bool(
            self.quality_ranking_enabled
            and self.quality_ranking_cfg.get("freeze_detector", False)
            and self.quality_ranking_cfg.get(
                "quality_only_when_frozen", True
            )
        )
        self.axial_depth_iou_cfg = axial_depth_iou_cfg or {}
        self.axial_depth_iou_enabled = bool(
            self.axial_depth_iou_cfg.get("enabled", False)
        )
        self.axial_depth_min_half_extent = float(
            self.axial_depth_iou_cfg.get("min_half_extent", 0.05)
        )
        self.axial_depth_eps = float(
            self.axial_depth_iou_cfg.get("eps", 1.0e-6)
        )
        if self.axial_depth_min_half_extent <= 0.0:
            raise ValueError("axial_depth_iou.min_half_extent must be positive")
        if self.axial_depth_eps <= 0.0:
            raise ValueError("axial_depth_iou.eps must be positive")

    def loss_labels(self, outputs, targets, indices, indices_filted, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if self.quality_aligned_cls:
            return self.loss_labels_quality(
                outputs, targets, indices, indices_filted, num_boxes, log=log
            )
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_quality(
        self, outputs, targets, indices, indices_filted, num_boxes, log=True
    ):
        """Align class confidence with 2D overlap and relative depth quality."""
        src_logits = outputs["pred_logits"]
        probabilities = src_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        positive_weights = torch.zeros_like(src_logits)
        negative_weights = probabilities.pow(2.0)
        idx = self._get_src_permutation_idx(indices)

        if idx[0].numel() > 0:
            target_classes = torch.cat(
                [target["labels"][target_index] for target, (_, target_index) in zip(targets, indices)]
            ).view(-1).long()
            source_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [target["boxes_3d"][target_index] for target, (_, target_index) in zip(targets, indices)],
                dim=0,
            )
            overlap = torch.diag(
                box_iou(
                    box_cxcylrtb_to_xyxy(source_boxes),
                    box_cxcylrtb_to_xyxy(target_boxes),
                )[0]
            ).clamp(0.0, 1.0)

            source_depth = outputs["pred_depth"][idx][:, 0]
            target_depth = torch.cat(
                [target["depth"][target_index] for target, (_, target_index) in zip(targets, indices)],
                dim=0,
            ).view(-1)
            relative_depth_error = (
                (source_depth - target_depth).abs() / target_depth.clamp_min(1.0)
            )
            depth_quality = torch.exp(
                -relative_depth_error / max(self.quality_depth_tau, 1e-6)
            )
            quality = (
                overlap.pow(self.quality_iou_alpha)
                * depth_quality.pow(1.0 - self.quality_iou_alpha)
            ).clamp(0.01, 1.0).detach()

            positive_weights[idx[0], idx[1], target_classes] = quality
            negative_weights[idx[0], idx[1], target_classes] = 1.0 - quality

        loss = (
            -positive_weights * probabilities.log()
            - negative_weights * torch.log1p(-probabilities)
        ).sum() / num_boxes
        losses = {"loss_ce": loss}
        if log and idx[0].numel() > 0:
            target_classes = torch.cat(
                [target["labels"][target_index] for target, (_, target_index) in zip(targets, indices)]
            ).view(-1)
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes)[0]
        return losses
    
    def loss_labels_fg(self, outputs, targets, indices, indices_filted, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if 'pred_logits_fg' in outputs:
            src_logits_fg = outputs['pred_logits_fg']

            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits_fg.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits_fg.device)

            target_classes[idx] = target_classes_o.squeeze().long()
            # trans to fg 2 class label
            # 大于0的为前景，等于0的为背景
            target_classes = target_classes > 0
            target_classes = target_classes.long()

            target_classes_onehot = torch.zeros([src_logits_fg.shape[0], src_logits_fg.shape[1], src_logits_fg.shape[2]+1],
                                                dtype=src_logits_fg.dtype, layout=src_logits_fg.layout, device=src_logits_fg.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce_fg = sigmoid_focal_loss(src_logits_fg, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits_fg.shape[1]
            losses = {'loss_ce_fg': loss_ce_fg}
            return losses
    
    def loss_labels_yolostesreo(self, outputs, targets, indices, indices_filted, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        loss_ce = self.loss_cls_2d(src_logits, target_classes_onehot).sum() / src_logits.shape[1] 
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, indices_filted, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices,  indices_filted, num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_sample_point(self, outputs, targets, indices,  indices_filted, num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        src_sample_point = outputs['pred_sample_points'][idx]
        target_sample_point = torch.cat([t['sample_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_sample_point = F.l1_loss(src_sample_point, target_sample_point, reduction='none')
        losses = {}
        losses['loss_sample_point'] = loss_sample_point.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, indices_filted, num_boxes):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths_stable(self, outputs, targets, indices, indices_filted, num_boxes):  
        bs, nq = outputs['pred_depth'].shape[:2]

        # filter out low iou pairs
        indices_final = []
        if self.depth_match_type == "filter":
            indices_final = indices_filted
        elif self.depth_match_type == "rematch":
            for batch_index in range(bs):
                out_bbox_i = outputs["pred_boxes"][batch_index]  # [batch_size * num_queries, 4]
                tgt_bbox_i = targets[batch_index]["boxes_3d"]
                all_iou = box_iou(box_cxcylrtb_to_xyxy(out_bbox_i), box_cxcylrtb_to_xyxy(tgt_bbox_i))[0].view(nq, -1) # (b, num_queries, ngt) 
                indices_i = indices[batch_index]
                # max_iou, max_idx = all_iou.max(dim=1)
                if all_iou.shape[-1] > 0:
                    max_iou, max_idx = torch.max(all_iou, dim=1)
                    iou_mask = max_iou >= 0.5
                    positive_list = torch.nonzero(iou_mask, as_tuple=False)
                    if len(positive_list.shape) > 1:
                        positive_list = positive_list.squeeze(1)
                    max_idx = max_idx[iou_mask]
                    indices_final.append([positive_list.detach().cpu(), max_idx.detach().cpu()])
                else:
                    indices_final.append([torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)])
        else:
                print("must in rematch or filter ")
        idx = self._get_src_permutation_idx(indices_final)
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices_final)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        _s = depth_input
 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        # depth_loss = F.l1_loss(depth_input, target_depths, reduction='none')
        losses = {}
        
        if len(idx[0]) > 0:
            losses['loss_depth'] = depth_loss.sum() / len(idx[0]) 
        else:
            losses['loss_depth'] = depth_loss.sum()*0.0
        return losses 

    def loss_axial_depth_iou(
        self, outputs, targets, indices, indices_filted, num_boxes
    ):
        """Depth-only 1D GIoU using detached box geometry as support.

        The predicted interval mirrors the G7 diagnostic: its half extent uses
        the current predicted dimensions and yaw, while the target interval
        uses KITTI dimensions and yaw.  All support geometry is detached, so
        this auxiliary objective can only update the predicted query depth.
        """

        del num_boxes
        indices_final = (
            indices_filted if self.depth_match_type == "filter" else indices
        )
        idx = self._get_src_permutation_idx(indices_final)
        predicted_depth = outputs["pred_depth"][idx][:, 0]
        if idx[0].numel() == 0:
            return {"loss_depth_axial_iou": predicted_depth.sum() * 0.0}

        required_output_keys = {
            "quality_calibs",
            "quality_img_sizes_ori",
        }
        missing_outputs = required_output_keys.difference(outputs)
        if missing_outputs:
            raise KeyError(
                "axial depth loss requires output metadata: {}".format(
                    sorted(missing_outputs)
                )
            )
        missing_target_keys = [
            (batch_index, key)
            for batch_index, target in enumerate(targets)
            for key in ("src_size_3d", "boxes_ry")
            if key not in target
        ]
        if missing_target_keys:
            raise KeyError(
                "axial depth target metadata missing: {}".format(
                    missing_target_keys
                )
            )

        target_depth = torch.cat(
            [
                target["depth"][target_index]
                for target, (_, target_index) in zip(
                    targets, indices_final
                )
            ],
            dim=0,
        ).reshape(-1)
        predicted_dimensions = outputs["pred_3d_dim"][idx]
        target_dimensions = torch.cat(
            [
                target["src_size_3d"][target_index]
                for target, (_, target_index) in zip(
                    targets, indices_final
                )
            ],
            dim=0,
        ).reshape(-1, 3)
        target_rotation_y = torch.cat(
            [
                target["boxes_ry"][target_index]
                for target, (_, target_index) in zip(
                    targets, indices_final
                )
            ],
            dim=0,
        ).reshape(-1)

        predicted_boxes = outputs["pred_boxes"][idx]
        predicted_alpha = decode_predicted_alpha(outputs["pred_angle"][idx])
        calibration_rows = []
        original_widths = []
        for batch_index, (source_index, _) in enumerate(indices_final):
            count = int(source_index.numel())
            if count == 0:
                continue
            calibration_rows.append(
                outputs["quality_calibs"][batch_index]
                .unsqueeze(0)
                .expand(count, -1, -1)
            )
            original_widths.append(
                outputs["quality_img_sizes_ori"][batch_index, 0]
                .reshape(1)
                .expand(count)
            )
        calibration = torch.cat(calibration_rows, dim=0).to(
            dtype=predicted_depth.dtype
        )
        original_width = torch.cat(original_widths, dim=0).to(
            dtype=predicted_depth.dtype
        )
        center_2d_x_normalized = (
            predicted_boxes[:, 0]
            + 0.5 * (predicted_boxes[:, 3] - predicted_boxes[:, 2])
        )
        image_x = center_2d_x_normalized * original_width
        focal_x = calibration[:, 0, 0].clamp_min(self.axial_depth_eps)
        principal_x = calibration[:, 0, 2]
        predicted_rotation_y = predicted_alpha + torch.atan2(
            image_x - principal_x, focal_x
        )
        predicted_rotation_y = torch.atan2(
            torch.sin(predicted_rotation_y),
            torch.cos(predicted_rotation_y),
        )

        loss = depth_only_axial_giou_loss(
            predicted_depth=predicted_depth,
            target_depth=target_depth,
            predicted_dimensions_hwl=predicted_dimensions,
            target_dimensions_hwl=target_dimensions,
            predicted_rotation_y=predicted_rotation_y,
            target_rotation_y=target_rotation_y,
            min_half_extent=self.axial_depth_min_half_extent,
            eps=self.axial_depth_eps,
        )
        if not torch.isfinite(loss).all():
            raise FloatingPointError("axial depth GIoU produced NaN or Inf")
        return {"loss_depth_axial_iou": loss.mean()}

    def loss_query_distribution(
        self, outputs, targets, indices, indices_filted, num_boxes
    ):
        """Soft-bin supervision for the query-level local depth distribution."""
        indices_final = indices_filted if self.depth_match_type == "filter" else indices
        idx = self._get_src_permutation_idx(indices_final)
        probabilities = outputs["pred_depth_dist_probs"][idx]
        bins = outputs["pred_depth_dist_bins"][idx]
        if idx[0].numel() == 0:
            return {"loss_query_dist": probabilities.sum() * 0.0}

        target_depth = torch.cat(
            [target["depth"][target_index] for target, (_, target_index) in zip(targets, indices_final)],
            dim=0,
        ).view(-1, 1)
        target_distribution = F.softmax(
            -(bins - target_depth).abs()
            / max(self.query_dist_target_temperature, 1e-6),
            dim=-1,
        )
        loss = -(
            target_distribution * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        return {"loss_query_dist": loss.mean()}

    def loss_depth_gate(
        self, outputs, targets, indices, indices_filted, num_boxes
    ):
        """Train the geometry gate to prefer the lower-error depth source."""
        indices_final = indices_filted if self.depth_match_type == "filter" else indices
        idx = self._get_src_permutation_idx(indices_final)
        gate = outputs["pred_depth_gate"][idx].view(-1)
        if idx[0].numel() == 0:
            return {"loss_depth_gate": gate.sum() * 0.0}

        stereo_depth = outputs["pred_stereo_depth"][idx].view(-1)
        geometry_depth = outputs["pred_geometry_depth"][idx].view(-1)
        target_depth = torch.cat(
            [target["depth"][target_index] for target, (_, target_index) in zip(targets, indices_final)],
            dim=0,
        ).view(-1)
        stereo_error = (stereo_depth - target_depth).abs()
        geometry_error = (geometry_depth - target_depth).abs()
        oracle_gate = torch.sigmoid(
            (geometry_error - stereo_error)
            / max(self.gate_oracle_temperature, 1e-6)
        ).detach()
        loss = F.binary_cross_entropy(
            gate.clamp(1e-6, 1.0 - 1e-6), oracle_gate
        )
        return {"loss_depth_gate": loss}

    def loss_teacher_distill(
        self, outputs, targets, indices, indices_filted, num_boxes
    ):
        """Distill cached teacher depth into the student's local distribution."""
        teacher_depth = torch.stack(
            [target["teacher_depth"] for target in targets], dim=0
        )
        teacher_confidence = torch.stack(
            [target["teacher_confidence"] for target in targets], dim=0
        )
        query_points = outputs["pred_query_points"].detach()
        sampling_grid = query_points.mul(2.0).sub(1.0)
        sampled_teacher_depth = F.grid_sample(
            teacher_depth,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.query_align_corners,
        ).permute(0, 2, 3, 1)
        sampled_teacher_confidence = F.grid_sample(
            teacher_confidence,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.query_align_corners,
        ).permute(0, 2, 3, 1)
        point_weights = outputs["pred_query_point_weights"].detach()
        teacher_depth_per_query = (
            sampled_teacher_depth * point_weights
        ).sum(dim=2)
        confidence_per_query = (
            sampled_teacher_confidence * point_weights
        ).sum(dim=2)

        indices_final = indices_filted if self.depth_match_type == "filter" else indices
        idx = self._get_src_permutation_idx(indices_final)
        student_probabilities = outputs["pred_depth_dist_probs"][idx]
        student_bins = outputs["pred_depth_dist_bins"][idx]
        teacher_depth_matched = teacher_depth_per_query[idx].view(-1, 1)
        confidence_matched = confidence_per_query[idx].view(-1)
        if idx[0].numel() == 0:
            return {"loss_teacher_distill": student_probabilities.sum() * 0.0}

        valid = (
            (confidence_matched >= self.teacher_confidence_threshold)
            & torch.isfinite(teacher_depth_matched.view(-1))
            & (teacher_depth_matched.view(-1) > 0.0)
        )
        if not valid.any():
            return {"loss_teacher_distill": student_probabilities.sum() * 0.0}

        teacher_distribution = F.softmax(
            -(student_bins - teacher_depth_matched).abs()
            / max(self.teacher_target_temperature, 1e-6),
            dim=-1,
        )
        cross_entropy = -(
            teacher_distribution
            * student_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        weights = confidence_matched * valid.to(confidence_matched.dtype)
        loss = (cross_entropy * weights).sum() / weights.sum().clamp_min(1.0)
        return {"loss_teacher_distill": loss}

    def loss_depths(self, outputs, targets, indices, indices_filted, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
   
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        # depth_loss = F.l1_loss(depth_input, target_depths, reduction='none')
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses 
    
    def loss_dims(self, outputs, targets, indices, indices_filted, num_boxes):  
        if self.depth_match_type == "filter":
            indices_final = indices_filted
        else:
            indices_final = indices
        idx = self._get_src_permutation_idx(indices_final)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices_final)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / len(idx[0]) 
        return losses

    def loss_angles(self, outputs, targets, indices, indices_filted, num_boxes):  
        if self.depth_match_type == "filter":
            indices_final = indices_filted
        else:
            indices_final = indices
        idx = self._get_src_permutation_idx(indices_final)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices_final)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices_final)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / len(idx[0])  
        return losses

    def loss_depth_map(self, outputs, targets, indices, indices_filted, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']
        _, _, H_depth_map, W_depth_map = depth_map_logits.shape
        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([W_depth_map, H_depth_map, W_depth_map, H_depth_map], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
        gt_boxes3d = torch.cat([t['boxes_3d'] for t in targets], dim=0)* \
            torch.tensor([W_depth_map, H_depth_map, W_depth_map, W_depth_map, H_depth_map, H_depth_map], device='cuda')
        
        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, gt_boxes3d, num_gt_per_img, gt_center_depth)
        return losses
    
    def loss_disp_map(self, outputs, targets, indices, indices_filted, num_boxes):
        disp_map_pre = outputs['pred_disp']
        losses = dict()
        if self.disparity_supervision_mode == 'multi_candidate':
            missing = [
                index
                for index, target in enumerate(targets)
                if 'disp_candidates' not in target
                or 'disp_candidate_weights' not in target
            ]
            if missing:
                raise KeyError(
                    'multi-candidate disparity targets are missing for batch '
                    'items {}'.format(missing)
                )
            candidates = torch.stack(
                [target['disp_candidates'] for target in targets], dim=0
            )
            candidate_weights = torch.stack(
                [
                    target['disp_candidate_weights']
                    for target in targets
                ],
                dim=0,
            )
            losses["loss_disp_map"] = self.multi_candidate_disparity_loss(
                disp_map_pre, candidates, candidate_weights
            )
        else:
            disp_maps = torch.cat(
                [t['disp'] for t in targets], dim=0
            ).squeeze(dim=1)
            losses["loss_disp_map"] = self.disparity_loss(
                disp_map_pre, disp_maps
            )
        return losses

    def loss_quality_ranking(
        self, outputs, targets, indices, indices_filted, num_boxes
    ):
        del indices, indices_filted, num_boxes
        quality_targets = build_3d_iou_quality_targets(outputs, targets)
        quality_logits = outputs["pred_quality_logits"]
        losses = {
            "loss_quality_point": pointwise_quality_loss(
                quality_logits,
                quality_targets,
                negative_threshold=float(
                    self.quality_ranking_cfg.get("negative_threshold", 0.1)
                ),
                negative_weight=float(
                    self.quality_ranking_cfg.get("negative_weight", 0.1)
                ),
            )
        }
        if self.quality_pairwise_enabled:
            predicted_labels = outputs["pred_logits"].detach().sigmoid().argmax(dim=-1)
            losses["loss_quality_pair"] = pairwise_quality_loss(
                quality_logits,
                quality_targets,
                predicted_labels,
                margin=float(self.quality_ranking_cfg.get("pairwise_margin", 0.1)),
                max_pairs_per_class=int(
                    self.quality_ranking_cfg.get("max_pairs_per_class", 32)
                ),
            )
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def set_training_epoch(self, epoch):
        self.geometry_epoch = int(epoch)

    def loss_geometry_alignment(
        self, outputs, targets, indices, indices_filted, num_boxes
    ):
        schedule = progressive_weight(
            self.geometry_epoch,
            int(self.geometry_alignment_cfg.get("start_epoch", 1)),
            int(self.geometry_alignment_cfg.get("ramp_epochs", 4)),
        )
        zero = outputs["pred_boxes"].sum() * 0.0
        losses = {}
        if self.geometry_corner_enabled:
            losses["loss_geometry_corner"] = zero
        if self.geometry_projection_enabled:
            losses["loss_geometry_projection"] = zero
        if schedule <= 0.0:
            return losses
        required = {
            "geometry_calibs", "geometry_img_sizes",
            "geometry_img_sizes_ori", "geometry_img_sizes_upper",
        }
        missing = required.difference(outputs)
        if missing:
            raise KeyError("geometry metadata missing: {}".format(sorted(missing)))
        indices_final = indices_filted if self.depth_match_type == "filter" else indices
        corner_terms = []
        projection_terms = []
        for batch_index, (source_index, target_index) in enumerate(indices_final):
            if source_index.numel() == 0:
                continue
            device = outputs["pred_boxes"].device
            source_index = source_index.to(device=device, dtype=torch.long)
            target_index = target_index.to(device=targets[batch_index]["boxes_3d"].device, dtype=torch.long)
            count = source_index.numel()
            projection = outputs["geometry_calibs"][batch_index].unsqueeze(0).expand(count, -1, -1)
            cropped_size = outputs["geometry_img_sizes"][batch_index].unsqueeze(0).expand(count, -1)
            original_size = outputs["geometry_img_sizes_ori"][batch_index].unsqueeze(0).expand(count, -1)
            upper = outputs["geometry_img_sizes_upper"][batch_index].reshape(1).expand(count)
            predicted_boxes = outputs["pred_boxes"][batch_index, source_index]
            predicted_depth = outputs["pred_depth"][batch_index, source_index, 0]
            predicted_dimensions = outputs["pred_3d_dim"][batch_index, source_index]
            predicted_alpha = decode_predicted_alpha(outputs["pred_angle"][batch_index, source_index])
            target_boxes = targets[batch_index]["boxes_3d"][target_index]
            target_depth = targets[batch_index]["depth"][target_index].reshape(-1)
            target_dimensions = targets[batch_index]["size_3d"][target_index]
            target_alpha = decode_target_alpha(
                targets[batch_index]["heading_bin"][target_index],
                targets[batch_index]["heading_res"][target_index],
            )
            predicted_geometry = decode_camera_geometry(
                predicted_boxes, predicted_depth, predicted_dimensions,
                predicted_alpha, projection, cropped_size, original_size, upper,
            )
            target_geometry = decode_camera_geometry(
                target_boxes, target_depth, target_dimensions,
                target_alpha, projection, cropped_size, original_size, upper,
            )
            predicted_corners = cuboid_corners(
                predicted_geometry["center"], predicted_geometry["dimensions"],
                predicted_geometry["rotation_y"],
            )
            target_corners = cuboid_corners(
                target_geometry["center"], target_geometry["dimensions"],
                target_geometry["rotation_y"],
            )
            if self.geometry_corner_enabled:
                corner_terms.append(corner_alignment_loss(predicted_corners, target_corners))
            if self.geometry_projection_enabled:
                points, visible = project_corners(predicted_corners, projection)
                # PyTorch 2.0 Tensor.all only accepts one dimension. Flatten
                # the two point dimensions so the same check also works in
                # the AutoDL torch 2.0.1 reproduction environment.
                finite_points = torch.isfinite(points).flatten(start_dim=1).all(dim=1)
                valid = visible.all(dim=1) & finite_points
                if valid.any():
                    projected_boxes = projected_enclosing_boxes(points[valid], original_size[valid])
                    target_projection_boxes = encoded_boxes_to_original_xyxy(
                        target_boxes[valid], cropped_size[valid], original_size[valid], upper[valid]
                    )
                    projection_terms.append(projection_alignment_loss(
                        projected_boxes, target_projection_boxes,
                        boundary_weight=float(self.geometry_alignment_cfg.get("projection_boundary_weight", 0.25)),
                    ))
        if self.geometry_corner_enabled and corner_terms:
            losses["loss_geometry_corner"] = torch.cat(corner_terms).mean() * schedule
        if self.geometry_projection_enabled and projection_terms:
            losses["loss_geometry_projection"] = torch.cat(projection_terms).mean() * schedule
        return losses

    def get_loss(self, loss, outputs, targets, indices, indices_filted, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'labels_fg': self.loss_labels_fg,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths_stable,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'sample_point': self.loss_sample_point,
            'depth_map': self.loss_depth_map,
            'disp_map': self.loss_disp_map,
            'query_distribution': self.loss_query_distribution,
            'depth_gate': self.loss_depth_gate,
            'teacher_distill': self.loss_teacher_distill,
            'quality_ranking': self.loss_quality_ranking,
            'geometry_alignment': self.loss_geometry_alignment,
            'axial_depth_iou': self.loss_axial_depth_iou,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, indices_filted, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if self.quality_only:
            return self.loss_quality_ranking(
                outputs, targets, None, None, None
            )

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, indices_filted = self.matcher(outputs_without_aux, targets, group_num=group_num)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            #ipdb.set_trace()
            losses.update(self.get_loss(loss, outputs, targets, indices, indices_filted, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, indices_filted = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:

                    if loss in ['depth_map', 'labels_fg', 'disp_map', 'dims','depths', 'angles', 'center', 'sample_point', 'depth_gate', 'teacher_distill', 'quality_ranking', 'geometry_alignment', 'axial_depth_iou']:
                    # if loss in ['depth_map', 'labels_fg', 'disp_map']:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                        
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, indices_filted, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_StereoDETR(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    if cfg['decoder_type'] == "depthaware":
        depthaware_transformer = build_depthaware_transformer_stereo(cfg)
    elif cfg['decoder_type'] == "detr":
        depthaware_transformer = build_detr_transformer_stereo(cfg)
    elif cfg['decoder_type'] == "anchor_detr":
        depthaware_transformer = build_anchor_transformer_stereo(cfg)
    # depth prediction module
    depth_predictor = LightStereoDepthPredictor(cfg)
    model = StereoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        with_center_refine=cfg['with_center_refine'],
        two_stage=cfg['two_stage'],
        init_box=cfg['init_box'],
        use_dab = cfg['use_dab'],
        group_num=cfg['group_num'],
        two_stage_dino=cfg['two_stage_dino'],
        depth_pre_type=cfg['depth_pre_type'],
        depth_sample_mode=cfg['depth_sample_mode'],
        fusion_mode=cfg['fusion_mode'],
        num_scale=cfg['num_scale'],
        FPN_type=cfg['FPN_type'],
        query_depth_cfg=cfg.get('query_depth', {}),
        quality_ranking_cfg=cfg.get('quality_ranking', {}),
        geometry_alignment_cfg=cfg.get('geometry_alignment', {}),
        dynamic_depth_upsampling_cfg=cfg.get(
            'dynamic_depth_upsampling', {}
        ),
        groupwise_correlation_cfg=cfg.get(
            'groupwise_correlation', {}
        ),
        geometry_depth_residual_cfg=cfg.get(
            'geometry_depth_residual', {}
        ),
        axial_depth_iou_cfg=cfg.get('axial_depth_iou', {}))

    # matcher
    matcher = build_fg_Stablematcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_sample_point'] = cfg['sample_point_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_disp_map'] = cfg['disp_map_loss_coef']
    query_depth_cfg = cfg.get('query_depth', {})
    quality_ranking_cfg = cfg.get('quality_ranking', {})
    geometry_alignment_cfg = cfg.get('geometry_alignment', {})
    axial_depth_iou_cfg = cfg.get('axial_depth_iou', {})
    if query_depth_cfg.get('enabled', False):
        weight_dict['loss_query_dist'] = float(
            query_depth_cfg.get('distribution_loss_coef', 1.0)
        )
        if query_depth_cfg.get('geometry_gate', False):
            weight_dict['loss_depth_gate'] = float(
                query_depth_cfg.get('gate_loss_coef', 0.2)
            )
        if query_depth_cfg.get('offline_teacher', False):
            weight_dict['loss_teacher_distill'] = float(
                query_depth_cfg.get('teacher_loss_coef', 0.5)
            )
    if geometry_alignment_cfg.get('enabled', False):
        if geometry_alignment_cfg.get('corner_enabled', False):
            weight_dict['loss_geometry_corner'] = float(
                geometry_alignment_cfg.get('corner_loss_coef', 1.0)
            )
        if geometry_alignment_cfg.get('projection_enabled', False):
            weight_dict['loss_geometry_projection'] = float(
                geometry_alignment_cfg.get('projection_loss_coef', 1.0)
            )
    if axial_depth_iou_cfg.get('enabled', False):
        weight_dict['loss_depth_axial_iou'] = float(
            axial_depth_iou_cfg.get('loss_coef', 0.5)
        )
    quality_loss_enabled = bool(
        quality_ranking_cfg.get('enabled', False)
        and quality_ranking_cfg.get('loss_enabled', True)
    )
    if quality_loss_enabled:
        weight_dict['loss_quality_point'] = float(
            quality_ranking_cfg.get('point_loss_coef', 10.0)
        )
        if quality_ranking_cfg.get('pairwise_enabled', False):
            weight_dict['loss_quality_pair'] = float(
                quality_ranking_cfg.get('pairwise_loss_coef', 0.5)
            )
    
    # dn loss
    if cfg['use_dn']:
        weight_dict['tgt_loss_ce']= cfg['cls_loss_coef']
        weight_dict['tgt_loss_bbox'] = cfg['bbox_loss_coef']
        weight_dict['tgt_loss_giou'] = cfg['giou_loss_coef']
        weight_dict['tgt_loss_angle'] = cfg['angle_loss_coef']
        weight_dict['tgt_loss_center'] = cfg['3dcenter_loss_coef']

    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'sample_point', 'depth_map', 'disp_map']
    if query_depth_cfg.get('enabled', False):
        losses.append('query_distribution')
        if query_depth_cfg.get('geometry_gate', False):
            losses.append('depth_gate')
        if query_depth_cfg.get('offline_teacher', False):
            losses.append('teacher_distill')
    if quality_loss_enabled:
        losses.append('quality_ranking')
    if geometry_alignment_cfg.get('enabled', False):
        losses.append('geometry_alignment')
    if axial_depth_iou_cfg.get('enabled', False):
        losses.append('axial_depth_iou')
    
    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        group_num=cfg['group_num'],
        depth_sort_reverse=cfg['depth_sort_reverse'],
        depth_bg=cfg['depth_bg'],
        shrink_ratio=cfg['shrink_ratio'], 
        align_by_3d_center=cfg['align_by_3d_center'],
        query_depth_cfg=query_depth_cfg,
        disparity_supervision_cfg=cfg.get(
            'disparity_supervision', {}
        ),
        quality_ranking_cfg=quality_ranking_cfg,
        geometry_alignment_cfg=geometry_alignment_cfg,
        axial_depth_iou_cfg=axial_depth_iou_cfg)

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
