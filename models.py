"""
Module: models.py

This module defines a Slow Fusion LRCN model for video classification using ResNet-50.
It uses two separate 2D CNN backbones (ResNet-50 for RGB frames and ResNet-50 for Optical Flow),
extracts spatial and motion feature vectors independently (2048 dims each), concatenates them at
an intermediate stage (4096 total dims), and models joint temporal dynamics using a shared LSTM.
"""

import torch
import torch.nn as nn
from torchvision import models


class Identity(nn.Module):
    """Placeholder identity layer to strip classification heads from ResNet backbones."""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SlowFusionLRCN(nn.Module):
    """
    Slow Fusion LRCN for video classification with pretrained ResNet-50 backbones.
    """

    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet50', in_channels=5):
        super(SlowFusionLRCN, self).__init__()

        # --- Stream 1: RGB Feature Backbone (3 channels) ---
        if cnn_model == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            rgb_cnn = models.resnet50(weights=weights)
        elif cnn_model == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            rgb_cnn = models.resnet34(weights=weights)
        elif cnn_model == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            rgb_cnn = models.resnet18(weights=weights)
        else:
            raise ValueError('Unsupported CNN backbone variant.')

        if pretrained:
            # Freeze initial layers, but unfreeze layer4 for deep action feature adaptation
            for param in rgb_cnn.parameters():
                param.requires_grad = False
            for param in rgb_cnn.layer4.parameters():
                param.requires_grad = True

        num_features_rgb = rgb_cnn.fc.in_features  # 2048 for ResNet-50
        rgb_cnn.fc = Identity()
        self.rgb_backbone = rgb_cnn

        # --- Stream 2: Optical Flow Feature Backbone (2 channels: u, v) ---
        if cnn_model == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            flow_cnn = models.resnet50(weights=weights)
        elif cnn_model == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            flow_cnn = models.resnet34(weights=weights)
        elif cnn_model == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            flow_cnn = models.resnet18(weights=weights)

        if pretrained:
            # Freeze initial layers, but unfreeze layer4 for deep motion feature adaptation
            for param in flow_cnn.parameters():
                param.requires_grad = False
            for param in flow_cnn.layer4.parameters():
                param.requires_grad = True

        # Adapt conv1 for 2 optical flow channels (u, v displacement fields)
        old_flow_conv = flow_cnn.conv1
        flow_cnn.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=old_flow_conv.out_channels,
            kernel_size=old_flow_conv.kernel_size,
            stride=old_flow_conv.stride,
            padding=old_flow_conv.padding,
            bias=old_flow_conv.bias is not None
        )

        # Initialize 2 flow channels with average weights from original RGB conv1
        with torch.no_grad():
            flow_cnn.conv1.weight[:, :2, :, :] = old_flow_conv.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)

        for param in flow_cnn.conv1.parameters():
            param.requires_grad = True

        num_features_flow = flow_cnn.fc.in_features  # 2048 for ResNet-50
        flow_cnn.fc = Identity()
        self.flow_backbone = flow_cnn

        # Intermediate Fusion Dimension (2048 + 2048 = 4096)
        total_fused_features = num_features_rgb + num_features_flow

        # --- Recurrent & Classification Head ---
        self.rnn = nn.LSTM(total_fused_features, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        Input x shape: (batch_size, time_steps, 5, height, width)
            - x[:, :, :3, :, :] -> RGB frames
            - x[:, :, 3:, :, :] -> Optical Flow maps (u, v)
        """
        bs, ts, c, h, w = x.shape

        # Split input tensor along channel dimension into RGB (3 channels) and Flow (2 channels)
        x_rgb = x[:, :, :3, :, :].contiguous().view(bs * ts, 3, h, w)
        x_flow = x[:, :, 3:, :, :].contiguous().view(bs * ts, 2, h, w)

        # Extract feature representations (2048-dim each)
        rgb_features = self.rgb_backbone(x_rgb)     # Shape: (bs * ts, 2048)
        flow_features = self.flow_backbone(x_flow)   # Shape: (bs * ts, 2048)

        # Intermediate Fusion: Concatenate feature vectors
        fused_features = torch.cat([rgb_features, flow_features], dim=1)  # Shape: (bs * ts, 4096)

        # Reshape to sequence form for temporal modeling
        fused_seq = fused_features.view(bs, ts, -1)   # Shape: (bs, ts, 4096)

        # Pass combined sequence into shared LSTM
        rnn_out, (hn, cn) = self.rnn(fused_seq)
        last_step_features = rnn_out[:, -1, :]

        out = self.dropout(last_step_features)
        out = self.fc(out)
        return out


# Alias to maintain script compatibility with LRCN class name in run.py
LRCN = SlowFusionLRCN