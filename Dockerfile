FROM runpod/pytorch:3.10-2.0.1-117-devel

# ARGS
ARG INSTALL_DIR="/webui" \
    DATA_DIR="/data" \
    UUID=1000

ENV INSTALL_DIR="$INSTALL_DIR" \
    DATA_DIR="$DATA_DIR" \
    UUID=$UUID \
    USERNAME="webui-user"

# Install apt packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget git python3 python3-venv \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install packages needed by extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libglfw3-dev libgles2-mesa-dev pkg-config libcairo2 libcairo2-dev && \
    rm -rf /var/lib/apt/lists/*

# Setup user which will run the service
RUN useradd -m -u $UUID "$USERNAME"
USER "$USERNAME"

# Copy repository into the docker image
COPY --chown="$USERNAME" . "$INSTALL_DIR"
WORKDIR "$INSTALL_DIR"

# Setup venv and pip cache
RUN python3 -m venv "$INSTALL_DIR/venv" && \
    mkdir -p "$INSTALL_DIR/cache/pip"
ENV PIP_CACHE_DIR="$INSTALL_DIR/cache/pip"

# Install dependencies (pip, wheel)
RUN . "$INSTALL_DIR/venv/bin/activate" && \
    pip install -U pip wheel

# Install automatic1111 dependencies (installer.py)
RUN . "$INSTALL_DIR/venv/bin/activate" && \
    python installer.py && \
    pip cache purge

# Start container as root in order to enable bind-mounts
USER root

STOPSIGNAL SIGINT
# In order to pass variables along to Exec Form Bash, we need to copy them explicitly
ENTRYPOINT ["/bin/bash", "-c", "${INSTALL_DIR}/entrypoint.sh $0 $@"]
