import torch
from torchviz import make_dot

model = YourCNNModel()
x = torch.randn(1, 3, 224, 224)  # 根据输入尺寸调整
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_arch", format="png")