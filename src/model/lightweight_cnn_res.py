import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class ResBlock(nn.Module):
    """DSConv + CoordAtt + MaxPool，带残差 shortcut"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主路径
        self.main = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # CoordAtt
            CoordAtt(out_channels, out_channels),
            # 下采样
            nn.MaxPool2d(2),
        )
        # shortcut：1x1 conv 对齐通道 + MaxPool 对齐空间
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return F.relu(self.main(x) + self.shortcut(x), inplace=True)


class LightweightCNNRes(nn.Module):
    """LightweightCNN + CoordAtt + 残差连接"""

    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = ResBlock(32, 64)
        self.layer3 = ResBlock(64, 128)
        self.layer4 = ResBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = LightweightCNNRes(num_classes=2)
    x = torch.randn(2, 1, 32, 64)
    out = model(x)
    print("输入:", x.shape, "→ 输出:", out.shape)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e3:.2f}K")
