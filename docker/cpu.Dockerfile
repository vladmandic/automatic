FROM ubuntu:22.04

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./ ./

RUN ./docker/setup.sh

ENV venv_dir=/python/venv

CMD ["./webui.sh", "--uv", "--listen", "--use-cpu=all"]