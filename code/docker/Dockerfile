# ──────────────────────────────────────────────────────────────────────────────
# CPR  —  Cascade Patch Retrieval for anomaly detection
#
# Base   : PyTorch 2.4.1 + CUDA 12.1 + cuDNN 9  (Ubuntu 20.04)
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# RTX 3060 = Ampere (sm_86)
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Deterministic cublas
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# System deps — OpenCV headless needs libGL + libGLib
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip install \
    "torchvision>=0.19" \
    "numpy>=1.24,<2.0" \
    "Pillow>=9.5" \
    "opencv-python-headless>=4.8,<4.10" \
    "scipy>=1.10,<1.14" \
    "scikit-learn>=1.2,<1.6" \
    "scikit-image>=0.20,<0.23" \
    "timm==0.9.12" \
    "kornia==0.7.3" \
    "pandas>=1.5,<3.0" \
    "matplotlib>=3.5" \
    "tqdm>=4.64" \
    "tabulate>=0.9" \
    "loguru>=0.7" \
    "einops>=0.6" \
    "imgaug>=0.4" \
    "tensorboard>=2.14" \
    "torchmetrics>=0.10,<1.0" \
    "pynvml>=11.5"

WORKDIR /app

CMD ["python", "-c", \
     "import torch, torchvision, timm, kornia, numpy; \
      print(f'torch={torch.__version__}  cuda={torch.version.cuda}  ' \
            f'tv={torchvision.__version__}  timm={timm.__version__}  ' \
            f'kornia={kornia.__version__}  numpy={numpy.__version__}')"]
