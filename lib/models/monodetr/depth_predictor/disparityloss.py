import torch
import torch.nn as nn

import torch.nn.functional as F


def isNaN(x):
    return x != x


class Disp2Prob(object):
    """
    Convert disparity map to matching probability volume
        Args:
            maxDisp, (int): the maximum of disparity
            gtDisp, (torch.Tensor): in (..., Height, Width) layout
            start_disp (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index

        Outputs:
            probability, (torch.Tensor): in [BatchSize, maxDisp, Height, Width] layout


    """
    def __init__(self, maxDisp:int, gtDisp:torch.Tensor, start_disp:int=0, dilation:int=1):

        if not isinstance(maxDisp, int):
            raise TypeError('int is expected, got {}'.format(type(maxDisp)))

        if not torch.is_tensor(gtDisp):
            raise TypeError('tensor is expected, got {}'.format(type(gtDisp)))

        if not isinstance(start_disp, int):
            raise TypeError('int is expected, got {}'.format(type(start_disp)))

        if not isinstance(dilation, int):
            raise TypeError('int is expected, got {}'.format(type(dilation)))

        if gtDisp.dim() == 2:  # single image H x W
            gtDisp = gtDisp.view(1, 1, gtDisp.size(0), gtDisp.size(1))

        if gtDisp.dim() == 3:  # multi image B x H x W
            gtDisp = gtDisp.view(gtDisp.size(0), 1, gtDisp.size(1), gtDisp.size(2))

        if gtDisp.dim() == 4:
            if gtDisp.size(1) == 1:  # mult image B x 1 x H x W
                gtDisp = gtDisp
            else:
                raise ValueError('2nd dimension size should be 1, got {}'.format(gtDisp.size(1)))

        self.gtDisp = gtDisp
        self.maxDisp = maxDisp
        self.start_disp = start_disp
        self.dilation = dilation
        self.end_disp = start_disp + maxDisp - 1
        self.disp_sample_number = (maxDisp + dilation -1) // dilation
        self.eps = 1e-40

    def getProb(self):
        # [BatchSize, 1, Height, Width]
        b, c, h, w = self.gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        index = torch.linspace(self.start_disp, self.end_disp, self.disp_sample_number)
        index = index.to(self.gtDisp.device)

        # [BatchSize, maxDisp, Height, Width]
        self.index = index.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()

        # the gtDisp must be (start_disp, end_disp), otherwise, we have to mask it out
        mask = (self.gtDisp > self.start_disp) & (self.gtDisp < self.end_disp)
        mask = mask.detach().type_as(self.gtDisp)
        self.gtDisp = self.gtDisp * mask

        probability = self.calProb()

        # let the outliers' probability to be 0
        # in case divide or log 0, we plus a tiny constant value
        probability = probability * mask + self.eps

        # in case probability is NaN
        if isNaN(probability.min()) or isNaN(probability.max()):
            print('Probability ==> min: {}, max: {}'.format(probability.min(), probability.max()))
            print('Disparity Ground Truth after mask out ==> min: {}, max: {}'.format(self.gtDisp.min(),
                                                                                      self.gtDisp.max()))
            raise ValueError(" \'probability contains NaN!")

        return probability

    def kick_invalid_half(self):
        distance = self.gtDisp - self.index
        invalid_index = distance < 0
        # after softmax, the valid index with value 1e6 will approximately get 0
        distance[invalid_index] = 1e6
        return distance

    def calProb(self):
        raise NotImplementedError


class LaplaceDisp2Prob(Disp2Prob):
    # variance is the diversity of the Laplace distribution
    def __init__(self, maxDisp, gtDisp, variance=1, start_disp=0, dilation=1):
        super(LaplaceDisp2Prob, self).__init__(maxDisp, gtDisp, start_disp, dilation)
        self.variance = variance

    def calProb(self):
        # 1/N * exp( - (d - d{gt}) / var), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        scaled_distance = ((-torch.abs(self.index - self.gtDisp)) / self.variance)
        probability = F.softmax(scaled_distance, dim=1)

        return probability

class StereoFocalLoss(object):
    """
    Under the same start disparity and maximum disparity, calculating all estimated cost volumes' loss
        Args:
            max_disp, (int): the max of Disparity. default: 192
            start_disp, (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index, it mainly used in gt probability volume generation
            weights, (list of float or None): weight for each scale of estCost.
            focal_coefficient, (float): stereo focal loss coefficient, details please refer to paper. default: 0.0
            sparse, (bool): whether the ground-truth disparity is sparse, for example, KITTI is sparse, but SceneFlow is not. default: False

        Inputs:
            estCost, (Tensor or list of Tensor): the estimated cost volume, in (BatchSize, max_disp, Height, Width) layout
            gtDisp, (Tensor): the ground truth disparity map, in (BatchSize, 1, Height, Width) layout.
            variance, (Tensor or list of Tensor): the variance of distribution, details please refer to paper, in (BatchSize, 1, Height, Width) layout.

        Outputs:
            loss, (dict), the loss of each level

        ..Note:
            Before calculate loss, the estCost shouldn't be normalized,
              because we will use softmax for normalization
    """

    def __init__(self, max_disp=192, start_disp=0, dilation=1, weights=None, focal_coefficient=0.0, sparse=False):
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.dilation = dilation
        self.weights = weights
        self.focal_coefficient = focal_coefficient
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, estCost, gtDisp, variance, dilation):
        N, C, H, W = estCost.shape
        scaled_gtDisp = gtDisp.clone()
        scale = 1.0
        if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
            # compute scale per level and scale gtDisp
            scale = gtDisp.shape[-1] / (W * 1.0)
            scaled_gtDisp = gtDisp.clone() / scale

            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))

        # mask for valid disparity
        # (start_disp, max disparity / scale)
        # Attention: the invalid disparity of KITTI is set as 0, be sure to mask it out
        lower_bound = self.start_disp
        upper_bound = lower_bound + int(self.max_disp/scale)
        mask = (scaled_gtDisp > lower_bound) & (scaled_gtDisp < upper_bound)
        mask = mask.detach_().type_as(scaled_gtDisp)
        if mask.sum() < 1.0:
            print('Stereo focal loss: there is no point\'s '
                  'disparity is in [{},{})!'.format(lower_bound, upper_bound))
            scaled_gtProb = torch.zeros_like(estCost)  # let this sample have loss with 0
        else:
            # transfer disparity map to probability map
            mask_scaled_gtDisp = scaled_gtDisp * mask
            scaled_gtProb = LaplaceDisp2Prob(int(self.max_disp/scale), mask_scaled_gtDisp, variance=variance,
                                             start_disp=self.start_disp, dilation=dilation).getProb()

        # stereo focal loss
        estProb = F.log_softmax(estCost, dim=1)
        weight = (1.0 - scaled_gtProb).pow(-self.focal_coefficient).type_as(scaled_gtProb)
        loss = -((scaled_gtProb * estProb) * weight * mask.float()).sum(dim=1, keepdim=True).mean()

        return loss

    def __call__(self, estCost, gtDisp, variance):
        if not isinstance(estCost, (list, tuple)):
            estCost = [estCost]

        if self.weights is None:
            self.weights = 1.0

        if not isinstance(self.weights, (list, tuple)):
            self.weights = [self.weights] * len(estCost)

        if not isinstance(self.dilation, (list, tuple)):
            self.dilation = [self.dilation] * len(estCost)

        if not isinstance(variance, (list, tuple)):
            variance = [variance] * len(estCost)

        # compute loss for per level
        loss_all_level = []
        for est_cost_per_lvl, var, dt in zip(estCost, variance, self.dilation):
            loss_all_level.append(
                self.loss_per_level(est_cost_per_lvl, gtDisp, var, dt))

        # re-weight loss per level
        loss = 0
        for i, loss_per_level in enumerate(loss_all_level):
            loss += self.weights[i] * loss_per_level

        return loss

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Dilation rate: {}\n'.format(self.dilation)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Focal coefficient: {}\n'.format(self.focal_coefficient)
        repr_str += ' ' * 4 + 'Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'StereoFocalLoss'

class DisparityLoss(nn.Module):
    """Some Information about DisparityLoss"""
    def __init__(self, maxdisp:int=64):
        super(DisparityLoss, self).__init__()
        #self.register_buffer("disp",torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])))
        self.criterion = StereoFocalLoss(maxdisp)

    def forward(self, x:torch.Tensor, label:torch.Tensor)->torch.Tensor:
        #x = torch.softmax(x, dim=1)
        label = label.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        loss = self.criterion(x, label, variance=0.5)
        #mask = (label > 0) * (label < 64)
        #loss = nn.functional.smooth_l1_loss(disp[mask], label[mask])
        return loss


class MultiCandidateDisparityLoss(nn.Module):
    """Probability-mass supervision from all valid disparities in a 4x4 cell.

    Candidate values are linearly splatted into the 96 disparity bins, then
    lightly smoothed with the same Laplace scale used by the original scalar
    loss.  Optional left-right confidence changes only relative pixel/candidate
    importance; its mean is normalized so the global loss scale stays close to
    the original StereoDETR setting.
    """

    def __init__(
        self,
        maxdisp: int = 96,
        variance: float = 0.5,
        smoothing_radius: int = 4,
        use_soft_confidence: bool = False,
    ):
        super(MultiCandidateDisparityLoss, self).__init__()
        if maxdisp <= 1:
            raise ValueError("maxdisp must be greater than one")
        if variance <= 0:
            raise ValueError("variance must be positive")
        if smoothing_radius < 0:
            raise ValueError("smoothing_radius must be non-negative")
        self.maxdisp = int(maxdisp)
        self.variance = float(variance)
        self.smoothing_radius = int(smoothing_radius)
        self.use_soft_confidence = bool(use_soft_confidence)

    def _smooth_probability(self, probability: torch.Tensor) -> torch.Tensor:
        if self.smoothing_radius == 0:
            return probability
        offsets = torch.arange(
            -self.smoothing_radius,
            self.smoothing_radius + 1,
            device=probability.device,
            dtype=probability.dtype,
        )
        kernel = torch.exp(-offsets.abs() / self.variance)
        kernel = kernel / kernel.sum().clamp_min(1.0e-12)
        smoothed = F.conv3d(
            probability.unsqueeze(1),
            kernel.view(1, 1, -1, 1, 1),
            padding=(self.smoothing_radius, 0, 0),
        ).squeeze(1)
        return smoothed / smoothed.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

    def build_target_probability(
        self,
        candidates: torch.Tensor,
        candidate_weights: torch.Tensor,
        output_dtype: torch.dtype,
    ):
        if candidates.ndim != 4:
            raise ValueError(
                "candidates must have [B, K, H, W] shape, got {}".format(
                    tuple(candidates.shape)
                )
            )
        if candidate_weights.shape != candidates.shape:
            raise ValueError(
                "candidate_weights shape {} does not match candidates {}".format(
                    tuple(candidate_weights.shape), tuple(candidates.shape)
                )
            )
        candidates = candidates.to(dtype=output_dtype)
        candidate_weights = candidate_weights.to(dtype=output_dtype)
        valid = (
            torch.isfinite(candidates)
            & torch.isfinite(candidate_weights)
            & (candidates > 0.0)
            & (candidates < float(self.maxdisp))
            & (candidate_weights > 0.0)
        )
        weights = torch.where(
            valid,
            candidate_weights.clamp(min=0.0, max=1.0),
            torch.zeros_like(candidate_weights),
        )
        mass = weights.sum(dim=1, keepdim=True)
        valid_cell = mass > 0.0
        normalized_weights = weights / mass.clamp_min(1.0e-12)

        safe_candidates = torch.where(
            valid, candidates, torch.zeros_like(candidates)
        )
        lower = safe_candidates.floor().long().clamp(
            min=0, max=self.maxdisp - 1
        )
        upper = (lower + 1).clamp(max=self.maxdisp - 1)
        fraction = (safe_candidates - lower.to(output_dtype)).clamp(0.0, 1.0)

        batch, _, height, width = candidates.shape
        probability = candidates.new_zeros(
            (batch, self.maxdisp, height, width), dtype=output_dtype
        )
        probability.scatter_add_(
            1, lower, normalized_weights * (1.0 - fraction)
        )
        probability.scatter_add_(
            1, upper, normalized_weights * fraction
        )
        probability = self._smooth_probability(probability)
        probability = probability * valid_cell.to(output_dtype)

        valid_count = valid.sum(dim=1, keepdim=True).to(output_dtype)
        cell_confidence = weights.sum(dim=1, keepdim=True) / valid_count.clamp_min(1.0)
        cell_confidence = cell_confidence * valid_cell.to(output_dtype)
        return probability.detach(), valid_cell.detach(), cell_confidence.detach()

    def forward(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
        candidate_weights: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != self.maxdisp:
            raise ValueError(
                "logits must have [B, {}, H, W] shape, got {}".format(
                    self.maxdisp, tuple(logits.shape)
                )
            )
        candidates = candidates.to(device=logits.device, dtype=logits.dtype)
        candidate_weights = candidate_weights.to(
            device=logits.device, dtype=logits.dtype
        )
        if candidates.shape[0] != logits.shape[0] or candidates.shape[-2:] != logits.shape[-2:]:
            raise ValueError(
                "candidate/logit spatial shapes differ: {} vs {}".format(
                    tuple(candidates.shape), tuple(logits.shape)
                )
            )
        target, valid_cell, confidence = self.build_target_probability(
            candidates, candidate_weights, logits.dtype
        )
        cross_entropy = -(target * F.log_softmax(logits, dim=1)).sum(
            dim=1, keepdim=True
        )
        reliability = valid_cell.to(logits.dtype)
        if self.use_soft_confidence and valid_cell.any():
            mean_confidence = confidence.sum() / valid_cell.sum().clamp_min(1)
            reliability = confidence / mean_confidence.clamp_min(1.0e-6)
            reliability = reliability.clamp(max=2.0) * valid_cell.to(
                logits.dtype
            )
        # Match the original loss convention: invalid pixels contribute zero,
        # while the denominator remains B*H*W.
        return (cross_entropy * reliability).mean()
