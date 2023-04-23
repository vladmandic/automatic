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

# Setup user
ARG UID=0
ARG GID=${UID}
RUN (if [ ${GID} -ne 0 ] ; then groupadd -g ${GID} webui ; fi) && \
    (if [ ${UID} -ne 0 ] ; then useradd -l -m -s /bin/bash -u ${UID} -g ${GID} webui ; fi)
RUN mkdir -p /webui && \
    chown ${UID}:${GID} -R /webui
USER ${UID}:${GID}

# Setup venv and pip cache
RUN python3 -m venv /webui/venv && \
    mkdir -p /webui/cache/pip
ENV PIP_CACHE_DIR=/webui/cache/pip

# Install dependencies (pip, wheel)
RUN --mount=type=cache,uid=${UID},gid=${GID},target=/webui/cache/pip \
    . /webui/venv/bin/activate && \
    pip install -U pip wheel

# Install dependencies (torch)
RUN --mount=type=cache,uid=${UID},gid=${GID},target=/webui/cache/pip \
    . /webui/venv/bin/activate && \
    pip install \
    torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies (requirements.txt)
ARG REPO=vladmandic/automatic
ARG BRANCH=master
ARG COMMIT=${BRANCH}
ADD --chown=${UID}:${GID} \
    https://raw.githubusercontent.com/${REPO}/${COMMIT}/requirements.txt \
    /webui/requirements.txt
RUN --mount=type=cache,uid=${UID},gid=${GID},target=/webui/cache/pip \
    . /webui/venv/bin/activate && \
    pip install -r /webui/requirements.txt && \
    rm -f /webui/requirements.txt

# Clone repo and install dependencies (setup.py)
ADD --chown=${UID}:${GID} \
    https://api.github.com/repos/${REPO}/git/refs/heads/${BRANCH} \
    /webui/version.json
RUN --mount=type=cache,uid=${UID},gid=${GID},target=/webui/cache/pip \
    rm -f /webui/version.json && \
    git clone https://github.com/${REPO}.git /webui/repo && \
    cd /webui/repo && \
    git checkout ${COMMIT} && \
    . /webui/venv/bin/activate && \
    python ./setup.py --noupdate

# Init webui environment
WORKDIR /webui/repo
RUN mkdir -p /webui/repo/data && \
    # Fix: data-dir owner mismatch
    git config --global --add safe.directory '*' && \
    # Fix: Disable quick mode on first startup
    mv setup.log setup.old.log

VOLUME /webui/repo/data
VOLUME /webui/repo/outputs
STOPSIGNAL SIGINT
COPY --chown=${UID}:${GID} entrypoint.sh /webui/entrypoint.sh
ENTRYPOINT [ "bash", "/webui/entrypoint.sh" ]
