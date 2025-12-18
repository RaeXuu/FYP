import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    标准 Transformer 里的位置编码（可复用）
    输入 shape: [seq_len, batch_size, d_model]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer，不参与梯度
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class AudioTransformer(nn.Module):
    """
    Tiny Audio Transformer
    适配你当前的特征: (B, 1, 64, 64) 的 Log-Mel 图像
    设计思路：
      - 把 Mel 看成序列: time=64 作为 token 长度
      - 每个 time step 的 freq=64 作为特征向量
      - 送入 TransformerEncoder
      - 对整个序列做 mean pooling 得到全局表示
    """
    def __init__(
        self,
        num_classes: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # 输入投影层: freq_dim(64) -> d_model
        self.input_proj = nn.Linear(64, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # 我们使用 [S, B, E] 形式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 序列聚合: 用 mean pooling（也可以改成 CLS token 或 attention pooling）
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 分类 head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 64, 64]
            B: batch_size
            1: channel (mel 单通道)
            64: freq_bins
            64: time_steps
        """
        # 去掉 channel 维度: [B, 1, 64, 64] -> [B, 64, 64]
        # 这里我们约定: x[:, 0, freq, time]，你现在的特征是 (1, 64, 64)，freq 和 time 的具体顺序
        # 如果你的 Mel 是 (1, n_mels, n_frames)，通常 n_mels=64, n_frames=64
        x = x.squeeze(1)  # [B, 64, 64]

        # 约定: time 维度 = 最后一维，freq 维度 = 倒数第二维
        # 也就是说: x[b, freq, time]
        # 我们需要把 time 当成序列长度 S，把 freq 当成特征维度
        # 先转成 [B, time, freq]
        x = x.permute(0, 2, 1)  # [B, 64(time), 64(freq)]

        # 线性投影到 d_model 维: [B, S, d_model]
        x = self.input_proj(x)  # [B, 64, d_model]

        # Transformer 期望输入: [S, B, E]
        x = x.transpose(0, 1)  # [64, B, d_model]

        # 加位置编码
        x = self.pos_encoder(x)  # [S, B, d_model]

        # 经过 Transformer Encoder
        x = self.transformer_encoder(x)  # [S, B, d_model]

        # 再转回 [B, S, d_model]
        x = x.transpose(0, 1)  # [B, S, d_model]

        # 对序列做 mean pooling: [B, S, d_model] -> [B, d_model]
        # 这里用的是对 S 维做平均
        x = x.transpose(1, 2)  # [B, d_model, S]
        x = self.pool(x).squeeze(-1)  # [B, d_model]

        # 分类
        logits = self.classifier(x)  # [B, num_classes]
        return logits


if __name__ == "__main__":
    # 自测代码：运行本文件，检查维度和参数量是否正常
    model = AudioTransformer(num_classes=5)
    dummy = torch.randn(2, 1, 64, 64)  # batch_size=2
    out = model(dummy)
    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {num_params}")
