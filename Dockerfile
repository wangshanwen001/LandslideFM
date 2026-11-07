FROM determinedai/environments:cuda-11.8-base-gpu-0.31.1

# 设置工作目录（小写）
WORKDIR /workspace/landslidefm

RUN conda create -n venv python=3.12
ENV PATH=/opt/conda/envs/venv/bin:$PATH

# 先复制 requirements.txt
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码和预训练模型到镜像（路径也改为小写）
COPY . /workspace/landslidefm