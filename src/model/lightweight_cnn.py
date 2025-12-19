

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块：
    1) depthwise: 对每个通道单独做卷积
    2) pointwise: 1x1 卷积做通道混合
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # depthwise
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # pointwise
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


class LightweightCNN(nn.Module):
    """
    轻量 CNN 模型：
    输入: 
    输出: (B, num_classes) 的 logits
    """
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()

        # 第一层普通卷积（提取低级特征）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 对 bicoherence：第一层不做池化，保留低频耦合结构
            # nn.MaxPool2d(kernel_size=2, stride=2)   # 64x64 -> 32x32
        )

        # 后面几层用轻量 depthwise separable conv
        self.dsconv2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 32x32 -> 16x16
        )

        self.dsconv3 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 16x16 -> 8x8
        )

        self.dsconv4 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 8x8 -> 4x4
        )

        # 全局平均池化：4x4 -> 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x: (B, 1, 64, 64)
        """
        x = self.conv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.dsconv4(x)

        x = self.global_pool(x)      # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 128)

        logits = self.classifier(x)  # (B, num_classes)
        return logits


if __name__ == "__main__":
    # 简单自测
    model = LightweightCNN(num_classes=5, in_channels=1)
    x = torch.randn(2, 1, 64, 64)  # batch_size=2
    out = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("参数量:", sum(p.numel() for p in model.parameters()) )
    print("LightweightCNN 自测完成 ✅")
