import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"显卡是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    # 尝试做一个简单的计算来触发内核加载
    a = torch.randn(100).cuda()
    b = a * 2
    print("GPU 计算测试成功！")