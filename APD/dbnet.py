import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DynamicFeatureFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list),
                                     out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        # 将每个尺度的特征转换到相同的通道数
        aligned_features = [conv(feat)
                            for conv, feat in zip(self.convs, features)]

        # 调整特征图大小到相同尺寸（使用最大的特征图尺寸）
        target_size = aligned_features[0].shape[2:]
        resized_features = [
            F.interpolate(feat, size=target_size,
                          mode='bilinear', align_corners=True)
            for feat in aligned_features
        ]

        # 拼接所有特征
        fused = torch.cat(resized_features, dim=1)
        # 融合特征
        output = self.fusion_conv(fused)
        return output


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_channels=[64, 128, 256, 512]):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels

        # 创建多个卷积层，逐步降采样并增加通道数
        for out_channels in feature_channels:
            block = nn.Sequential(
                nn.Conv2d(current_channels, out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.layers.append(block)
            current_channels = out_channels

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

# 修改DBNet模型以包含新模块


class DBNet(nn.Module):
    def __init__(self, backbone_name='resnet18'):
        super().__init__()
        # 原有的backbone
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            pretrained=True
        )

        # 添加多尺度特征提取器
        self.multi_scale_extractor = MultiScaleFeatureExtractor()

        # 获取backbone的输出通道数
        backbone_channels = self.backbone.feature_info.channels()

        # 特征融合模块
        self.feature_fusion = DynamicFeatureFusion(
            in_channels_list=backbone_channels,
            out_channels=256
        )

        # 输出头部分保持不变
        self.prob_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.thresh_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.binary_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        # 获取backbone特征
        backbone_features = self.backbone(x)

        # 获取多尺度特征
        multi_scale_features = self.multi_scale_extractor(x)

        # 合并所有特征
        all_features = backbone_features + multi_scale_features

        # 特征融合
        fused_features = self.feature_fusion(all_features)

        # 生成各种map
        prob_map = self.prob_head(fused_features)
        thresh_map = self.thresh_head(fused_features)
        binary_map = self.binary_head(fused_features)

        return {
            'prob_map': prob_map,
            'thresh_map': thresh_map,
            'binary_map': binary_map
        }
