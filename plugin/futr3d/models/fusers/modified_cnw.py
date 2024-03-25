import torch
from torch import nn

# from mmdet3d.models.builder import FUSERS
# __all__ = ["DynamicFusion"]
# @FUSERS.register_module()

class ModifiedCNW(nn.Sequential):
    def __init__(self, num_channels, input_shape):
        super().__init__()
        self.num_channels = num_channels
        self.input_shape = input_shape

        self.camera_weights = nn.Parameter(torch.rand(self.num_channels))
        self.lidar_weights = nn.Parameter(torch.rand(self.num_channels))

        self.conv3x3 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bnorm = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(True)

        self.avg = nn.AdaptiveAvgPool2d(input_shape)
        self.conv1x1 = nn.Conv2d(num_channels, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, features): # [lidar, camera]
        # Fusion using Channel Normalised Weights
        norm_weight_camera = torch.softmax(self.camera_weights, dim=0).view(1, 1, -1) 
        norm_weight_lidar = torch.softmax(self.lidar_weights, dim=0).view(1, 1, -1) 

        out = features[0] * norm_weight_lidar + features[1] * norm_weight_camera

        # x = input.reshape(-1, self.input_shape[0], self.input_shape[1], self.num_channels)
        # x = self.conv3x3(x)
        # x = self.bnorm(x)
        # x = self.relu(x)

        # x1 = self.avg(x)
        # x1 = self.conv1x1(x1)
        # x1 = self.sigmoid(x1)

        # out = x * x1
        
        return out
