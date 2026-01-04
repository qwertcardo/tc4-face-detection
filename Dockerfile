FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

# Sistema + Python + build deps (em um único layer)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake \
      libgl1 \
      libglib2.0-0 \
      libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Garante "python" e "pip"
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Atualiza pip e instala dependências Python
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      "numpy<2" \
      tf-keras \
      deepface \
      opencv-python-headless \
      tqdm \
      dlib \
      mediapipe

WORKDIR /workspace
CMD ["bash"]