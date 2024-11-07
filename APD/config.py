from dataclasses import dataclass
from typing import Tuple


@dataclass
class APDConfig:
    # Image settings
    image_size: Tuple[int, int] = (224, 224)
    num_channels: int = 3

    # Training settings
    batch_size: int = 32
    max_epochs: int = 10
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4

    # DBNet specific settings
    threshold: float = 0.5  # Added: threshold for binary map
    scale_range: float = 0.2
    rotation_range: int = 10
    aug_prob: float = 0.5

    # Model architecture settings
    backbone_name: str = 'resnet18'
    feature_channels: Tuple[int, ...] = (64, 128, 256, 512)
    fusion_channels: int = 256
