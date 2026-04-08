import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)), inplace=True)
        x = F.relu(self.bn2(self.pointwise(x)), inplace=True)
        return x


class LightweightCNN(nn.Module):
    """OG 架构，通道加宽至 32→64→128→256（无 CoordAtt，无 Dropout）"""

    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dsconv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2)
        )
        self.dsconv3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2)
        )
        self.dsconv4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.dsconv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = LightweightCNN(num_classes=2)
    x = torch.randn(2, 1, 32, 64)
    out = model(x)
    print("输入:", x.shape, "→ 输出:", out.shape)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e3:.2f}K")
