import torch
from torch import nn
import numpy as np

# from mmdet3d.models.builder import FUSERS
# __all__ = ["DynamicFusion"]
# @FUSERS.register_module()

class ModifiedCNW(nn.Sequential):
    def __init__(
            self, 
            num_channels, 
            input_shape, 
            use_adaptive_fusion=False,):
        super().__init__()
        self.num_channels = num_channels
        self.input_shape = input_shape
        self.use_adaptive_fusion = use_adaptive_fusion

        self.camera_weights  = nn.Parameter(torch.rand(self.num_channels))
        self.lidar_weights   = nn.Parameter(torch.rand(self.num_channels))
        
        if self.use_adaptive_fusion:
            self.feature_enhancer = nn.Sequential(
                nn.Linear(self.num_channels, 2 * self.num_channels),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Linear(2 * self.num_channels, self.num_channels),
            )

            self.channel_weights = nn.Parameter(torch.rand(self.num_channels))

    def forward(self, features): # [lidar, camera]
        norm_lidar_weight   = torch.softmax(self.lidar_weights,  dim=0).view(1, 1, -1) 
        norm_camera_weight  = torch.softmax(self.camera_weights, dim=0).view(1, 1, -1) 
        norm_channel_weight = torch.softmax(self.camera_weights, dim=0).view(1, 1, -1) 

        x = features[0] * norm_lidar_weight + features[1] * norm_camera_weight

        if self.use_adaptive_fusion:
            x = self.feature_enhancer(x)
            x = x * norm_channel_weight

        return x
