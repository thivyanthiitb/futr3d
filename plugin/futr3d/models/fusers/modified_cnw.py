import torch
from torch import nn
import numpy as np

# from mmdet3d.models.builder import FUSERS
# __all__ = ["DynamicFusion"]
# @FUSERS.register_module()

class ModifiedCNW(nn.Sequential):
    def __init__(self, num_channels, input_shape, use_spatial_adaptive_fusion=False):
        super().__init__()
        self.num_channels = num_channels
        self.input_shape = input_shape
        self.use_spatial_adaptive_fusion = use_spatial_adaptive_fusion
        self.camera_weights = nn.Parameter(torch.rand(self.num_channels))
        self.lidar_weights = nn.Parameter(torch.rand(self.num_channels))
        if self.use_spatial_adaptive_fusion:
            self.spatial_fusion = nn.Sequential(
                nn.Conv2d(self.num_channels, self.num_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(True)
            )
            self.adaptive_feature_selection = nn.Sequential(
                nn.AdaptiveAvgPool2d(self.input_shape),
                nn.Conv2d(num_channels, num_channels, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, features): # [lidar, camera]
        feature_shape = features[0].shape
        H, W = self.input_shape
        intermediate_shape = (feature_shape[0], self.num_channels, H, W)
        # Fusion using Channel Normalised Weights
        norm_weight_lidar  = torch.softmax(self.lidar_weights,  dim=0).view(1, 1, -1) 
        norm_weight_camera = torch.softmax(self.camera_weights, dim=0).view(1, 1, -1) 
        x = features[0] * norm_weight_lidar + features[1] * norm_weight_camera
        if not self.use_spatial_adaptive_fusion:
            out = x
        else:
            x = x.reshape(intermediate_shape)
            x = self.spatial_fusion(x)
            w = self.adaptive_feature_selection(x)
            out = w * x
            out = out.reshape(feature_shape)
        return out
