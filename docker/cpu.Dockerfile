FROM ubuntu:22.04

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./ ./

RUN ./docker/setup.sh

ENTRYPOINT ["./docker/entrypoint.sh"]