FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./ ./

RUN ls -al

RUN ./docker/setup.sh

ENTRYPOINT ["./docker/entrypoint.sh"]

# CMD ["./webui.sh", "--uv", "--listen", "--use-cuda"]