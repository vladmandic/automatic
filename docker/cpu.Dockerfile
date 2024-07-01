FROM ubuntu:22.04

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --chmod=755 ./ ./

RUN echo Updating apt... && \
    apt update -y > /dev/null

RUN echo Upgrading apt... && \
    apt upgrade -y > /dev/null

RUN echo "Install apt packages" && \
    apt install -y git \
    python3.10 \
    pythonpy \
    python3.10-venv \
    python3-pip \
    curl \
    wget \
    aria2 > /dev/null

RUN python3 -m venv /python/venv

ENV venv_dir=/python/venv

CMD ["./webui.sh", "--uv", "--listen"]