import torch.nn as nn
import torch.nn.functional as F

from .refinement import StereoDRNetRefinement

from .feature_extractor import FeatureExtractor
from .cost import CostVolumePyramid
from .aggregation import AdaptiveAggregation
from .estimation import DisparityEstimationPyramid

class StereoMatchingNetwork(nn.Module):
    def __init__(self, max_disp,
                 in_channels=3,
                 num_downsample=2,
                 no_mdconv=False,
                 feature_similarity='correlation',
                 num_scales=3,
                 num_fusions=6,
                 deformable_groups=2,
                 mdconv_dilation=2,
                 no_intermediate_supervision=False,
                 num_stage_blocks=1,
                 num_deform_blocks=3,
                 refine_channels=None):
        super(StereoMatchingNetwork, self).__init__()

        refine_channels = in_channels if refine_channels is None else refine_channels
        self.num_downsample = num_downsample
        self.num_scales = num_scales

        # Feature extractor
        self.feature_extractor = FeatureExtractor(in_channels=in_channels)
        max_disp = max_disp // 3

        # Cost volume construction
        self.cost_volume_constructor = CostVolumePyramid(max_disp, feature_similarity=feature_similarity)

        # Cost aggregation
        self.aggregation = AdaptiveAggregation(max_disp=max_disp,
                                               num_scales=num_scales,
                                               num_fusions=num_fusions,
                                               num_stage_blocks=num_stage_blocks,
                                               num_deform_blocks=num_deform_blocks,
                                               no_mdconv=no_mdconv,
                                               mdconv_dilation=mdconv_dilation,
                                               deformable_groups=deformable_groups,
                                               intermediate_supervision=not no_intermediate_supervision)

        # Disparity estimation
        self.disparity_estimation = DisparityEstimationPyramid(max_disp)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(num_downsample):
            refine_module_list.append(StereoDRNetRefinement(img_channels=refine_channels))

        self.refinement = refine_module_list

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1. / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(left_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
                curr_right_img = F.interpolate(right_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
            inputs = (disparity, curr_left_img, curr_right_img)
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid

    def forward(self, left_img, right_img):
        left_feature = self.feature_extractor(left_img)
        right_feature = self.feature_extractor(right_img)
        cost_volume = self.cost_volume_constructor(left_feature, right_feature)
        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_estimation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])

        return disparity_pyramid
