import torch.nn as nn
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)
    
class Hourglass(nn.Module):
    def __init__(self, depth, num_features, num_blocks):
        super().__init__()
        self.depth = depth
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.upper_branch = self._make_branch(num_blocks, num_features, num_features)
        self.lower_branch = self._make_branch(num_blocks, num_features, num_features)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if self.depth > 1:
            self.inner_hourglass = Hourglass(depth - 1, num_features, num_blocks)
        else:
            self.inner_residual = self._make_branch(num_blocks, num_features, num_features)

    def _make_branch(self, num_blocks, in_channels, out_channels):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        upper_out = self.upper_branch(x)
        pooled = self.pool(x)
        if self.depth > 1:
            inner_out = self.inner_hourglass(pooled)
        else:
            inner_out = self.inner_residual(pooled)
        lower_out = self.lower_branch(inner_out)
        upsampled = self.upsample(lower_out)
        return upsampled + upper_out
    
class StackedHourglass(nn.Module):
    def __init__(self, num_stacks, num_blocks, num_classes, in_channels=3):
        super().__init__()
        self.num_stacks = num_stacks
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pre_processing = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256)
        )
        self.hourglass_stack = nn.ModuleList([
            Hourglass(depth=4, num_features=256, num_blocks=num_blocks)
            for _ in range(num_stacks)
        ])
        self.heatmap_heads = nn.ModuleList([
            self._make_head(256, num_classes) for _ in range(num_stacks)
        ])
        self.feature_remaps = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1) for _ in range(num_stacks - 1)
        ])
        self.prediction_remaps = nn.ModuleList([
            nn.Conv2d(num_classes, 256, kernel_size=1) for _ in range(num_stacks - 1)
        ])

    def _make_head(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        features = self.pre_processing(x)
        all_heatmaps = []
        for i in range(self.num_stacks):
            hourglass_out = self.hourglass_stack[i](features)
            heatmap = self.heatmap_heads[i](hourglass_out)
            all_heatmaps.append(heatmap)
            if i < self.num_stacks - 1:
                features_remap = self.feature_remaps[i](hourglass_out)
                preds_remap = self.prediction_remaps[i](heatmap)
                features = features + features_remap + preds_remap

        return all_heatmaps
    

class CelebAHeatmapDataset(Dataset):
    def __init__(self, crop_faces, input_size, output_size):
        self.crop_faces = crop_faces
        self.input_size = input_size
        self.output_size = output_size
    def __len__(self):
        return len(self.crop_faces)

    def __getitem__(self, idx):
        
        return self.crop_faces[idx][0], self.crop_faces[idx][1], self.crop_faces[idx][2]
    