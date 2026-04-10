import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from torchviz import make_dot
from src.model.lightweight_cnn import LightweightCNN

model = LightweightCNN(num_classes=2, in_channels=1)
x = torch.randn(1, 1, 64, 64)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_arch", format="png")
print("model_arch.png 已保存")
