import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAtt(nn.Module):
    """
    Coordinate Attention 模块
    分别在水平（时间）和垂直（频率）方向进行特征编码，保留精确的位置信息。
    """
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 1. 分别在高度和宽度方向进行池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # 2. 拼接并进行特征融合
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        # 3. 拆分回两个方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 4. 生成注意力权重并应用
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w

class DepthwiseSeparableConv(nn.Module):
    """
    带 Coordinate Attention 的深度可分离卷积
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 集成 CoordAtt
            CoordAtt(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class LightweightCNN(nn.Module):
    """
    升级版轻量级 CNN (PhysioNet 2016 二分类专用)
    """
    def __init__(self, num_classes=2, in_channels=1, dropout=0.3):
        super().__init__()

        # 第一层：普通卷积 (16 -> 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 后面几层通道数：32 -> 64 -> 128 -> 256
        self.layer2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2)  # 64x64 -> 32x32
        )

        self.layer3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2)  # 32x32 -> 16x16
        )

        self.layer4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # 模拟输入：BatchSize=2, 通道=1, Mel频带=64, 时间帧=64
    model = LightweightCNN(num_classes=2, in_channels=1)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e3:.2f} k")
    print("LightweightCNN (CoordAtt版) 自测完成 ✅")