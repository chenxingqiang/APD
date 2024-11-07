import torch
import torch.nn as nn
from typing import  Dict


class APDModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Initialize DBNet components
        self.backbone = self._init_backbone(config)
        self.neck = self._init_neck(config)
        self.head = self._init_detection_head(config)

    def _init_backbone(self, config):
        return nn.Sequential(
            # Initial conv layer
            nn.Conv2d(config.num_channels, 64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsample layers
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def _init_neck(self, config):
        return nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def _init_detection_head(self, config):
        return nn.ModuleDict({
            'prob': nn.Conv2d(64, 1, kernel_size=1),
            'thresh': nn.Conv2d(64, 1, kernel_size=1),
            'binary': nn.Conv2d(64, 1, kernel_size=1)
        })

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Process image through backbone
        features = self.backbone(pixel_values)

        # Process through neck
        neck_features = self.neck(features)

        # Get detection outputs
        return {
            'prob_map': self.head['prob'](neck_features),
            'thresh_map': self.head['thresh'](neck_features),
            'binary_map': self.head['binary'](neck_features)
        }

    def get_loss(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the DBNet loss"""
        # Binary cross entropy for probability map
        prob_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred['prob_map'],
            target['prob_map']
        )

        # L1 loss for threshold map
        thresh_loss = torch.nn.functional.l1_loss(
            torch.sigmoid(pred['thresh_map']),
            target['thresh_map']
        )

        # Binary cross entropy for binary map
        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred['binary_map'],
            target['binary_map']
        )

        # Total loss
        total_loss = prob_loss + 0.5 * thresh_loss + 0.5 * binary_loss

        return total_loss
