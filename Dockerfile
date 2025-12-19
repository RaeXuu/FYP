# 1. 选择基础镜像（用户态环境）
FROM ubuntu:22.04

# 2. 避免 apt 交互卡死
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安装系统依赖 + Python
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 4. 设定工作目录（容器里的 /workspace）
WORKDIR /workspace

# 5. 默认进入 bash（方便调试）
CMD ["bash"]
