FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./ ./

RUN ./docker/setup.sh

ENTRYPOINT ["./docker/entrypoint.sh"]