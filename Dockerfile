FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# ARGS
ARG UID=1000
ARG INSTALLDIR="/webui"
ENV RUNDIR ${INSTALLDIR}

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

RUN useradd -m -u $UID user
USER user

# Copy Local Files to Container
COPY --chown=user . $INSTALLDIR

# Setup venv and pip cache
RUN python3 -m venv $INSTALLDIR/venv && \
    mkdir -p $INSTALLDIR/cache/pip
ENV PIP_CACHE_DIR=$INSTALLDIR/cache/pip

# Install dependencies (pip, wheel)
RUN . $INSTALLDIR/venv/bin/activate && \
    pip install -U pip wheel

# Install dependencies (torch)
RUN . $INSTALLDIR/venv/bin/activate && \
    pip install \
    torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies (requirements.txt)
RUN . $INSTALLDIR/venv/bin/activate && \
    pip install -r $INSTALLDIR/requirements.txt

# Install automatic111 dependencies (installer.py)
RUN cd $INSTALLDIR && \
    . $INSTALLDIR/venv/bin/activate && \
    python installer.py

STOPSIGNAL SIGINT
# In order to pass variables along to Exec Form Bash, we need to copy them explicitly
ENTRYPOINT ["/bin/bash", "-c", "${RUNDIR}/entrypoint.sh $0 $@"]
