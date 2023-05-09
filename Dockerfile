FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install apt packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget git python3 python3-venv \
    libgl1 libglib2.0-0 \
    libgoogle-perftools-dev \
    && \
    rm -rf /var/lib/apt/lists/*

# Workaround: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/6850
ENV LD_PRELOAD=libtcmalloc.so

# Workaround: https://gitlab.com/nvidia/container-images/cuda/-/issues/192
RUN cd /usr/local/cuda/targets/x86_64-linux/lib && \
    ln -sv libnvrtc.so.11.2 libnvrtc.so

# Setup user and pick 1000 as the default UID for huggingface compatiblity
ARG UID=1000
RUN useradd -m -u $UID user
USER user

# Copy Local Files to Container
COPY --chown=user . /webui

# Setup venv and pip cache
RUN python3 -m venv /webui/venv && \
    mkdir -p /webui/cache/pip
ENV PIP_CACHE_DIR=/webui/cache/pip

# Install dependencies (pip, wheel)
RUN . /webui/venv/bin/activate && \
    pip install -U pip wheel

# Install dependencies (torch)
RUN . /webui/venv/bin/activate && \
    pip install \
    torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies (requirements.txt)
RUN . /webui/venv/bin/activate && \
    pip install -r /webui/requirements.txt

# Clone repo and install dependencies (setup.py)
RUN cd /webui && \
    . /webui/venv/bin/activate && \
    python installer.py

STOPSIGNAL SIGINT
ENTRYPOINT [ "bash", "/webui/entrypoint.sh" ]