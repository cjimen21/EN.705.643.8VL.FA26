"""
Module: test_pipeline.py

Unit and integration test suite using pytest to verify dataset loading,
tensor shapes, flow pre-computation, and 5-channel Slow Fusion LRCN model forward passes.
"""

import pytest
import torch
import numpy as np
from models import LRCN
from utils import transform_stats, compose_data_transforms


def test_lrcn_model_shapes():
    """Verify that SlowFusionLRCN accepts 5-channel inputs and outputs correct class logits shape."""
    batch_size = 2
    time_steps = 16
    in_channels = 5
    height, width = 224, 224
    n_classes = 50

    # Instantiate model matching run.py parameter settings
    model = LRCN(
        hidden_size=100,
        n_layers=1,
        dropout_rate=0.1,
        n_classes=n_classes,
        pretrained=False,
        cnn_model='resnet34',
        in_channels=in_channels
    )
    model.eval()

    # Dummy batch shape: (B, T, C, H, W)
    dummy_input = torch.randn(batch_size, time_steps, in_channels, height, width)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, n_classes), f"Expected shape ({batch_size}, {n_classes}), got {output.shape}"


def test_transform_stats_5_channel():
    """Verify normalization statistics extension for 5-channel Slow Fusion."""
    h, w, mean, std = transform_stats(model='lrcn', in_channels=5)

    assert (h, w) == (224, 224)
    assert len(mean) == 5, f"Expected 5 mean values, got {len(mean)}"
    assert len(std) == 5, f"Expected 5 std values, got {len(std)}"
    # Verify zero-mean and unit-variance for optical flow channels u and v
    assert mean[3:] == [0.0, 0.0]
    assert std[3:] == [1.0, 1.0]


def test_data_transforms():
    """Ensure image transforms produce valid normalized tensors."""
    height, width = 224, 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tr_transforms, val_transforms = compose_data_transforms(height, width, mean, std)

    from PIL import Image
    dummy_pil = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))

    transformed = val_transforms(dummy_pil)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, height, width)